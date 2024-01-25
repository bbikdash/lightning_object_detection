#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from loguru import logger
import yaml
from pathlib import Path

# Model
from . import YOLOX, YOLOFPN, YOLOPAFPN, YOLOXHead
from utils import load_ckpt


__all__ = [
    "create_yolox_model",
    "yolox_nano",
    "yolox_tiny",
    "yolox_s",
    "yolox_m",
    "yolox_l",
    "yolox_x",
    "yolov3",
    "yolox_custom"
]


_KEF_YOLOX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # Working directory for kef_yolox
_KEF_YOLOX_PRESETS_YAML = os.path.join(_KEF_YOLOX_ROOT, "config/yolox_architectures.yaml")     # Path relative to kef_yolox working directory
_PRETRAINED_CKPTS_DIR = os.path.join(_KEF_YOLOX_ROOT, "weights/pretrained_weights")
_PRETRAINED_CKPT_PATHS = {
    "yolox_nano": f"{_PRETRAINED_CKPTS_DIR}/yolox_nano.pth",
    "yolox_tiny": f"{_PRETRAINED_CKPTS_DIR}/yolox_tiny.pth",
    "yolox_s": f"{_PRETRAINED_CKPTS_DIR}/yolox_s.pth",
    "yolox_m": f"{_PRETRAINED_CKPTS_DIR}/yolox_m.pth",
    "yolox_l": f"{_PRETRAINED_CKPTS_DIR}/yolox_l.pth",
    "yolox_x": f"{_PRETRAINED_CKPTS_DIR}/yolox_x.pth",
    "yolov3": f"{_PRETRAINED_CKPTS_DIR}/yolox_darknet.pth",
}


@logger.catch
def create_yolox_model(architecture: str, pretrained: bool = True, num_classes: int = 80) -> nn.Module:
    """Creates and loads a YOLOX model.

    Args:
        architecture (str): name of model. for example, "yolox-s", "yolox-tiny" or "yolox_custom"
        if you want to load your own model.
        pretrained (bool): load pretrained weights into the model. Default to True.
        device (str): default device to for model. Default to None.
        num_classes (int): number of model classes. Default to 80.
        exp_path (str): path to your own experiment file. Required if name="yolox_custom"
        ckpt_path (str): path to your own ckpt. Required if name="yolox_custom" and you want to
            load a pretrained model


    Returns:
        YOLOX model (nn.Module)
    """

    assert architecture in _PRETRAINED_CKPT_PATHS or architecture == "yolox_custom", \
        f"user should use one of value in {_PRETRAINED_CKPT_PATHS.keys()} or \"yolox_custom\""
    
    # Get path to YOLOX model configuration presets (in .yaml file)
    logger.info(f"Building {architecture}")
    # Build YOLOX from preset
    # NOTE: Hardcoded path of yolox configurations relative to this YOLOX Lightning Module
    # Load yolox architecture presets from the YAML file
    with open(_KEF_YOLOX_PRESETS_YAML, "r") as stream:
        try:
            yolox_presets = yaml.safe_load(stream)
        except yaml.YAMLError as ex:
            logger.error("Error while parsing YAML file. Could not load YOLOX architecture description!")
            sys.exit(1)

    # Get classes
    backbone_class_ = globals()[yolox_presets[architecture]['backbone']['class']]
    head_class_ = globals()[yolox_presets[architecture]['head']['class']]
    
    # Extract key word arguments for backbone and head model instantiation
    backbone_kwargs = yolox_presets[architecture]['backbone']['init_args']
    head_kwargs = yolox_presets[architecture]['head']['init_args']
    
    backbone = backbone_class_(**backbone_kwargs)
    head = head_class_(*[num_classes], **head_kwargs)

    logger.info(f"\tNetwork backbone: {yolox_presets[architecture]['backbone']['class']}")
    logger.info(f"\tNetwork head: {yolox_presets[architecture]['head']['class']}")

    # Instantiate YOLOX model
    model = YOLOX(backbone, head)
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
    # Initialize weights/biases
    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)

    # Load pretrained weights if available
    if pretrained and architecture != "yolox_custom":
        logger.info("Loading pretrained weights for fine tuning")
        ckpt = torch.load(_PRETRAINED_CKPT_PATHS[architecture], map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        model = load_ckpt(model, ckpt)
    else:
        logger.warning("Attempting to load pretrained weights for custom YOLOX model. Check that weights exist")

    model.train()
    logger.success("YOLOX Built")
    return model


if __name__ == "main":
    logger.info("Debugging model pathing and loading")
    create_yolox_model("yolox_s", num_classes=5)
    