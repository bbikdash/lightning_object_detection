#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import numpy as np
import torch
from torch import nn
import torchvision
# from torch.onnx.verification import VerificationOptions, GraphInfo, find_mismatch

import onnx
import onnxruntime
from onnxsim import simplify

from lit_yolox import LitYOLOX
from models.network_blocks import SiLU
from utils import replace_module



_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]
def meshgrid(*tensors):
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)
    
def decode_outputs(outputs, hw, arch_strides, dtype):
    # This is determined by the input size of the image.
    # Height/width of feature maps

    grids = []
    strides = []
    for (hsize, wsize), stride in zip(hw, arch_strides):
        yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(dtype)
    strides = torch.cat(strides, dim=1).type(dtype)

    outputs = torch.cat([
        (outputs[..., 0:2] + grids) * strides,
        torch.exp(outputs[..., 2:4]) * strides,
        outputs[..., 4:]
    ], dim=-1)
    return outputs


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):

    global IMAGE_PRED
    global BBOXES
    global CLASS_CONF
    global CLASS_PRED
    global SCORES
    global CONF_MASK
    global DETECTIONS

    # box_corner = prediction.new(prediction.shape)
    box_corner = torch.zeros_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    BBOXES = box_corner[0,:,:4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        IMAGE_PRED = image_pred

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        CLASS_CONF = class_conf
        CLASS_PRED = class_pred

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        CONF_MASK = conf_mask

        SCORES = image_pred[:, 4] * class_conf.squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        DETECTIONS = detections
    return detections




@logger.catch
def main(args):

    logger.info("args value: {}".format(args))

    arch_strides = args.arch_strides
    hw = []
    for a in arch_strides:
        hw.append((int(args.image_size[0] / a), int(args.image_size[1] / a)))

    # Initializations and hyperparameters
    device = torch.device("cuda:0") if args.device == "gpu" else torch.device("cpu")
    # Initialize model
    model = LitYOLOX.load_from_checkpoint(args.ckpt)
    model.to(device)

    model = replace_module(model, nn.SiLU, SiLU)
    model.model.head.decode_in_inference = args.decode_in_inference
    model.eval()

    dummy_input = torch.randn(args.batch_size, 3, args.image_size[0], args.image_size[1], device=device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
        decoded_output = decode_outputs(dummy_output.to(device), hw=hw, arch_strides=arch_strides, dtype=torch.float)
        post_proc_output = postprocess(decoded_output.to(device), args.num_classes, 0.01, 0.45, True)

    logger.info(f"Network Input: {dummy_input.shape}")
    logger.info(f"Network Output: {dummy_output.shape}")

    model.to_onnx(
        file_path=args.output_name,
        input_sample=dummy_input,
        export_params=True,
        do_constant_folding=True,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes={args.input: {0: 'batch'},
                      args.output: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset,
    )
    logger.success("Saved onnx model: {}".format(args.output_name))

    if not args.no_onnxsim:
        # use onnx-simplifier to reduce redundant model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model)
        onnx.checker.check_model(model_simp, True)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.success("Saved simplified onnx model: {}".format(args.output_name))


    ort_session = onnxruntime.InferenceSession(args.output_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # ONNX Runtime expects a numpy array
    # Compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # Compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(dummy_output), ort_outs[0], rtol=1e-03, atol=1e-03)
    logger.success("Exported model has been tested with ONNXRuntime, and the result looks good!")


    print()
    logger.info("For C++ runtime the following matrix sizes will be useful to know:")
    print(f"Expected Network Input Size: {dummy_input.shape}\n"
          f"Raw Network Output Size: {dummy_output.shape}\n"
          f"Decoding Step Input Size: {dummy_output.shape}\n"
          f"Decoding Step Output Size: {decoded_output.shape}\n"
          f"Post Processing Step Input Size: {decoded_output.shape}\n"
          f"Post Processing Step Output Size: {post_proc_output.shape}\n\n"
          f"C++ Preallocated Variables and Matrix Sizes:\n"
          f"\thw: {hw}\n"
          f"\timage_pred: {IMAGE_PRED.shape}\n"
          f"\tbboxes: {BBOXES.shape}\n"
          f"\tclass_conf: {CLASS_CONF.shape}\n"
          f"\tclass_pred: {CLASS_PRED.shape}\n"
          f"\tscores: {SCORES.shape}\n"
          f"\tconf_mask: {CONF_MASK.shape}\n"
          f"\tdetections: {DETECTIONS.shape}"
         )

    logger.success("Computed matrix sizes")


if __name__ == "__main__":

    # Process cmd line args
    parser = argparse.ArgumentParser(description="YOLOX ONNX Export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--output-name", type=str, default="yolox_eo_rgb_gray_finetune.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="input", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=16, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--num-classes", type=int, default=1, help="number of classes")
    parser.add_argument("--arch-strides", nargs="+", type=int, default=[8, 16, 32], help="YOLOX architecture hyperparameter. Convoluional stride during inference")
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--ckpt", type=str, default="./logs/yolox_s_eo_building/etg_finetune/gray/version_1++/checkpoints/epoch=115-step=17296.ckpt")

    parser.add_argument("--image_size", nargs="+", type=int, default=[640, 640], help="Resize the image before inference.")

    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )

    args = parser.parse_args()
    main(args)

