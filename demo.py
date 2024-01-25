#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2
import torch

from lit_yolox import LitYOLOX
from data.detection_data_module import DroneNetDataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pdb
import imageio

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def write2mp4(path, frames, fps=10):
    writer = imageio.get_writer(path, fps=fps)

    for f in frames:
        writer.append_data(f)
    writer.close()

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def main(args):
    # Initializations and hyperparameters
    device = torch.device("cuda") if args.device == "gpu" else torch.device("cpu")
    # Initialize model
    model = LitYOLOX.load_from_checkpoint(args.ckpt)
    model.to(device)


    vis_folder = os.path.dirname(os.path.dirname(args.ckpt))
    save_folder = os.path.join(vis_folder, "vis_results")

    if args.save_result:
        os.makedirs(save_folder, exist_ok=True)

    # Load image paths
    image_paths = []
    if args.mode == 'image':
        image_paths.append(args.path)
    elif args.mode == 'dir':
        image_paths.extend(sorted(get_image_list(args.image_dir)))

    frames = []
    # Run inference on all of the images
    for ip in image_paths:
        image = cv2.imread(ip, 1)

        if args.image_size[0] == -1 or args.image_size[1] == -1:
            # Use original image dimensions
            h,w,c = image.shape
        else:
            # Use the desired image dimensions
            h,w = args.image_size

        # Verify image dimensions
        if h % 32 != 0:
            h = int(round(h / 32) * 32)
        if w % 32 != 0:
            w = int(round(w / 32) * 32)
        
        transforms = A.Compose([
            # Geometric transformations
            # A.ToGray(always_apply=True),
            A.Resize(h, w, always_apply=True), 
            ToTensorV2(),
        ])
        
        # Perform model prediction
        pred = model.inference(image, args.conf, args.nms, transforms, False)
        frames.append(pred)

        logger.info(f"Saving {ip}")

        # Save the resulting image
        cv2.imwrite(os.path.join(save_folder, os.path.basename(ip)), pred)
    write2mp4(os.path.join(save_folder, "video.mp4"), frames, fps=5)

if __name__ == "__main__":
    # Process cmd line args
    parser = argparse.ArgumentParser(description="YOLOX Detection Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # PROGRAM level args
    # Program arguments (data_path, cluster_email, etc…)
    parser.add_argument_group("Program_Args", description="PROGRAM level arguments such as data paths, cluster email, etc.")
    parser.add_argument("--mode", type=str, default="dir", choices=["image", "dir"])
    # parser.add_argument("--path", type=str, default="/mnt/data/Datasets/UAV-Vehicle-Detection-Dataset/data/val/MunichStreet02-MOS91.png")
    parser.add_argument("--path", type=str, default="/mnt/data/Datasets/KEF_Vesper_Vehicle_Evaluation_Datasets/demo/187719784054.png")

    # parser.add_argument("-dir", "--image_dir", type=str, default="/mnt/data/Datasets/Vehicle_Detection/UAV-Vehicle-Detection-Dataset/data/val")
    # parser.add_argument("-dir", "--image_dir", type=str, default="/mnt/data/Datasets/KEF_Vesper_Vehicle_Evaluation_Datasets/demo")
    # parser.add_argument("-dir", "--image_dir", type=str, default="/mnt/data/Datasets/Person_Detection/Soccer_Player_Dataset/images/val")
    parser.add_argument("-dir", "--image_dir", type=str, default="/mnt/data/Datasets/NAMC_Object_Detection/images")

    parser.add_argument("-save", "--save_result", type=str, default="./logs")
    # parser.add_argument("--ckpt", type=str, default="./logs/yolox_m/version_0/checkpoints/epoch=1-step=6152.ckpt")
    parser.add_argument("--ckpt", type=str, default="./logs/yolox_m/version_4/checkpoints/epoch=26-step=107651.ckpt")

    parser.add_argument("--image_size", nargs="+", type=int, default=[640, 1120], help="Resize the image before inference.")
    # parser.add_argument("--image_size", nargs="+", type=int, default=[-1, -1], help="Resize the image before inference.")

    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu", help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--conf", default=0.30, type=float, help="test conf")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")

    args = parser.parse_args()
    main(args)
