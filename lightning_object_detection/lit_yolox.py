# Copyright (c) April 2023 KEF Robotics in Pittsburgh, PA

"""
@author Bassam Bikdash, https://github.com/Megvii-BaseDetection/YOLOX


"""
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics import AveragePrecision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import albumentations as A
from typing import Dict, List, Optional, Any
import yaml
from bidict import bidict

from loguru import logger
import inspect

# Model
from models.build import create_yolox_model
from utils import postprocess, gpu_mem_usage, mem_usage, xyxy2cxcywh, vis

import pdb

class LitYOLOX(LightningModule):    # Name, Object of Interest, Special Format, Purpose

    def __init__(self,
                 architecture: str,
                 classes_path: str,
                 pretrained: bool=True,
                 confidence_threshold: float=0.25,
                 nms_threshold: float=0.45,
        ):
        """
        Args:
            architecture: YOLOX model architecture to use: yolox_<nano, tiny, s, m, l, x, darknet, custom>
            classes_path: Path to yaml file containing the classes ids and labels that the model will predict. Should match number of classes in dataset. DO NOT CHANGE
            pretrained: Load weights pretrained on COCO2017
            confidence_threshold: Confidence threshold for inference
            nms_threshold: Non-maximal suppression threshold for inference
        """
        super().__init__()

        self.architecture = architecture
        self.pretrained = pretrained
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold


        with open(classes_path, "r") as stream:
            try:
                classes = yaml.safe_load(stream)
            except yaml.YAMLError as ex:
                logger.error("Error while parsing YAML file. Could not load class ids and labels!")
                sys.exit(1)

        self.num_classes = len(classes)
        self.class_ids = list(classes.keys())
        self.class_labels = list(classes.values())

        self.model = create_yolox_model(self.architecture, self.pretrained, self.num_classes)

        # Save all hyperparameters to the checkpoint and a yaml file within the logging dir
        self.save_hyperparameters()

        # Create torchmetrics logging objects here
        self.map = MeanAveragePrecision(box_format='cxcywh', iou_type='bbox', compute_on_cpu=True)
        # Create any additional nn.Module loss functions here
        return
    
    @logger.catch
    def forward(self, x: torch.Tensor, y: torch.Tensor=None):
        x = self.model(x, targets=y)
        return x

    # NOTE: When using the PyTorch Lightning CLI, this is overriden by the optimizer and scheduler in the config file.
    def configure_optimizers(self):
        logger.info("Setting optimizer and lr scheduler from script")
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-04)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        return [optimizer], [scheduler]
    

    def on_train_start(self) -> None:
        print()
        logger.success("Training start ...")
        return

    def training_step(self, train_batch, batch_idx) -> None:
        # Get the image and the label
        inputs, target, _ = train_batch

        # Perform forward pass through the network
        # Automatically computes loss
        outputs = self(inputs, target)

        loss = outputs["total_loss"]

        # If you want to calculate epoch-level metrics and log them, use log().
        # Logging to TensorBoard by default
        self.log("train/total_loss", outputs['total_loss'], prog_bar=True)
        self.log("train/iou_loss", outputs["iou_loss"], prog_bar=True)
        self.log("train/l1_loss", outputs["l1_loss"], prog_bar=False)
        self.log("train/conf_loss", outputs["conf_loss"], prog_bar=False)
        self.log("train/class_loss", outputs["cls_loss"], prog_bar=False)
        self.log("train/num_fg", outputs["num_fg"], prog_bar=False)
        self.log("train/gpu_mem", gpu_mem_usage(), prog_bar=False, on_epoch=True)
        self.log("train/mem", mem_usage(), prog_bar=False, on_epoch=True)

        return loss


    def validation_step(self, val_batch, batch_idx) -> None:
        # Get the image and the label
        inputs, target, img_info = val_batch
        b, c, h, w = inputs.shape
        # pdb.set_trace()

        # Perform forward pass through the network. Ensure no gradients are computed. We don't want the validation set to affect the gradients.
        # Losses are only computed during training mode. Set model to train just to compute losses
        t0 = time.time()
        with torch.no_grad():
            self.model.train()
            losses = self(inputs, target)
            self.model.eval()
            decoded_output = self(inputs)   
            # Postprocess the outputs # detections: 
            postprocessed_output = postprocess(decoded_output, self.num_classes, self.confidence_threshold, self.nms_threshold)   # List with len=batch_size

        # Error checking: what if there are no detections? postprocessed_output is a list of Nones???
        # Convert detections to coco format
        t1 = time.time()
        preds = []
        targets = []
        for (sample, target_sample, og_height, og_width) in zip(postprocessed_output, target, img_info[0], img_info[1]):
            if sample is None:
                sample = torch.zeros((10, 7), device=self.device)
            # sample: [x1, y1, x2, y2, obj_conf, class_conf, class]
            # target_sample: [class, cx, cy, w, h]

            # Convert sample bboxes to coco format
            sample_bboxes = sample[:, 0:4]
            sample_bboxes = xyxy2cxcywh(sample_bboxes)

            preds.append(
                # Add 
                dict(
                    boxes=sample_bboxes,
                    scores=sample[:, 4] * sample[:, 5],
                    labels=sample[:, 6].to(torch.int)
                )
            )

            targets.append(
                dict(
                    boxes=target_sample[:, 1:5],
                    labels=target_sample[:, 0].to(torch.int)
                )
            )

        # Compute mean average precision
        self.map.update(preds, targets)
        # Save model based on highest mean average precision
        
        # Calling self.log will surface up scalars for you in TensorBoard
        # Log losses        
        self.log("val/total_loss", losses['total_loss'], prog_bar=True)
        self.log("val/iou_loss", losses["iou_loss"], prog_bar=False)
        self.log("val/l1_loss", losses["l1_loss"], prog_bar=False)
        self.log("val/conf_loss", losses["conf_loss"], prog_bar=False)
        self.log("val/class_loss", losses["cls_loss"], prog_bar=False)
        self.log("val/num_fg", losses["num_fg"], prog_bar=False)
        self.log("val/gpu_mem", gpu_mem_usage(), prog_bar=False, on_epoch=True)
        self.log("val/mem", mem_usage(), prog_bar=False, on_epoch=True)
        
        return losses["total_loss"]

    def on_validation_epoch_end(self) -> None:
        """TorchMetrics mAP output of forward and compute the metric returns the following:
        map_dict: A dictionary containing the following key-values:
            map: (Tensor)
            map_small: (Tensor)
            map_medium:(Tensor)
            map_large: (Tensor)
            mar_1: (Tensor)
            mar_10: (Tensor)
            mar_100: (Tensor)
            mar_small: (Tensor)
            mar_medium: (Tensor)
            mar_large: (Tensor)
            map_50: (Tensor) (-1 if 0.5 not in the list of iou thresholds)
            map_75: (Tensor) (-1 if 0.75 not in the list of iou thresholds)
            map_per_class: (Tensor) (-1 if class metrics are disabled)
            mar_100_per_class: (Tensor) (-1 if class metrics are disabled)"""
        # Log mAP
        # mAP compute times become very large (>20 seconds). So we will only compute the mAP metric once at the end of validation
        t0 = time.time()
        temp = self.map.compute()
        del temp['classes']
        self.log_dict(temp, prog_bar=False)
        self.map.reset()
        # print("\n")
        logger.success(f"MAP Compute time: {time.time() - t0}\n")
        return


    def test_step(self, test_batch, batch_idx) -> None:
        # Get the image and the label
        inputs, target, _, _ = test_batch

        # Perform forward pass through the network. Ensure no gradients are computed. We don't want the validation set to affect the gradients.
        # Losses are only computed during training mode. Set model to train just to compute losses
        with torch.no_grad():
            self.model.train()
            losses = self(inputs, target)
            self.model.eval()
            decoded_output = self(inputs)

        # Compute mean average precision and log it
        # Save model based on highest mean average precision
        # Log losses
        self.log("test/total_loss", losses['total_loss'], prog_bar=True)
        self.log("test/iou_loss", losses["iou_loss"], prog_bar=True)
        self.log("test/l1_loss", losses["l1_loss"], prog_bar=False)
        self.log("test/conf_loss", losses["conf_loss"], prog_bar=True)
        self.log("test/class_loss", losses["cls_loss"], prog_bar=True)
        self.log("test/num_fg", losses["num_fg"], prog_bar=False)
        self.log("test/gpu_mem", gpu_mem_usage(), prog_bar=False)
        self.log("test/mem", mem_usage(), prog_bar=False)
        return losses["total_loss"]


    def on_test_epoch_end(self) -> None:
        # Log mAP
        # mAP compute times become very large (>20 seconds). So we will only compute the mAP metric once at the end of validation
        t0 = time.time()
        self.log_dict(self.map.compute(), prog_bar=False)
        self.map.reset()
        print("\n")
        logger.info(f"MAP Compute time: {time.time() - t0}\n")
        return
    

    @logger.catch
    def inference(self,
                  image: np.ndarray,
                  confidence_threshold: float,
                  nms_threshold: float,
                  transforms: A.Compose=None,
                  normalize: bool=True) -> np.ndarray:
        """
        Single image prediction of YOLOX object detection

        Args:
            image: image to use for prediction. Will be resized to closest height/width multiples 32
            confidence_threshold: Confidence threshold for inference
            nms_threshold: Non-maximal suppression threshold for inference
            transforms: preprocessing image transforms (color correction, resize, crop, etc.) 
            normalize: normalize image b/t [0,1] or not
        """
        logger.info("Running single image inference")
        self.eval()

        # Preprocess image (crop, to_tensor)
        network_input = transforms(image=image)['image']    # Output size: [3, height, width]
        c, h, w = network_input.shape
        assert h % 32 == 0 and w % 32 == 0, "Network input height/width must be multiples of 32"

        network_input = network_input.unsqueeze(0).to(torch.float)  # Add batch dimension: [1, 3, height, width]
        network_input = network_input.to(self.device)   # Move image to the device

        # Allocate image for visualization (move to cpu, convert to numpy, remove batching dim, and transpose to: [height, width, 3])
        background = network_input.detach().cpu().numpy()[0].transpose(1,2,0).copy()

        # Apply prediction and decoding
        with torch.no_grad():
            outputs = self(network_input)
            detections = postprocess(
                outputs, self.num_classes, confidence_threshold,
                nms_threshold, class_agnostic=True
            )[0]
        
        if detections == None:
            logger.error("No detections!")
            return background
        else:
            logger.success(f"{len(detections)} object(s) detected!")
            bboxes = detections[:, 0:4]
            cls = detections[:, 6]
            scores = detections[:, 4] * detections[:, 5]
            # logger.info(bboxes)
            # Visualize bounding boxes in [0,255] range
            vis_res = vis(background, bboxes, scores, cls, confidence_threshold, self.class_labels)

            if normalize:
                # Normalize between 0 and 1
                vis_res = (vis_res - np.min(vis_res)) / (np.max(vis_res) - np.min(vis_res))
                
            # Return final visual
            return vis_res

if __name__ == "__main__":
    logger.info("Debugging model pathing and loading")
    LitYOLOX("yolox_s", num_classes=5, pretrained=True)
