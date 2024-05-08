#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2023 KEF Robotics in Pittsburgh, PA

"""
@author Bassam Bikdash
February 2023

This python module contains Pytorch Dataset classes to load the UAv vehicle detection dataset
provided here: https://github.com/jwangjie/UAV-Vehicle-Detection-Dataset

Aerial images are provided with bounding box annotations
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import pdb

import copy

import warnings
from typing import Tuple

from loguru import logger
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import xyxy2cxcywh

class ThermalBuildingDetection(Dataset):
    def __init__(self,
                 image_dir: str,
                 label_dir: str,
                 transforms: A.Compose,
                 class_ids: list,
                 class_labels: list,
                 max_labels: int = 50) -> None:
        """

        NOTE: We will use the cropped images for loading and crop/resize them via data augmentations during training
        The <train,val>_bboxes.csv in each csv_folder will always refer to images within <train,val>/images/ and not within cropped/.

        data_dir/
            image1.png
            image2.png
            ...
        label_dir/
            image1.txt
            image2.txt
            ...

        Args:
            data_dir: path to the directory containing images
            label_dir: path to the directory containing bounding box labels as text files
            transforms: Albumentations object of transformations to apply to the image
            class_ids: list of ids (ints) representing class predictions (0=person, 1=fish)
            class_labels: list of labels (strings) that correspond to the class ids and represent the name of the class
            max_labels: default tensor size for target. Bounding box labels are padded to [max_labels, 5] for efficient batching.
                        if you get the following error: `RuntimeError: stack expects each tensor to be equal size...`. Increase this number
        """
        
        # Find the images in the current images directory
        self.image_names = [ name for name in sorted(os.listdir(image_dir)) if name.endswith('png') or name.endswith('jpg') or name.endswith('JPG') ]
        self.image_paths = [ os.path.join(image_dir, image_name) for image_name in self.image_names ]
        self.label_names = [ name for name in sorted(os.listdir(label_dir)) if name.endswith('txt') ]
        self.label_paths = [ os.path.join(label_dir, image_name) for image_name in self.label_names ]

        logger.info(f"Number of images found: {len(self.image_names)}\tNumber of labels found: {len(self.label_names)}")
        assert(len(self.image_names) == len(self.label_names))

        self.class_ids = class_ids
        self.class_labels = class_labels
        self.name = "thermal_building_dataset"
        self.bbox_format = "yolo_normalized" # Bounding box format of the labels in each text file

        # A set of composed Albumentation transformations (can be for training or validation)
        self.transforms = transforms

        # Maximum number of labels per image
        self.max_labels = max_labels


    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.image_paths)


    def load_image(self, index: int):
        # Load image in color (to maintain 3 channels)
        image = cv2.imread(self.image_paths[index], 1)
        # Image will be converted to gray when transformations are applied
        assert image is not None, f"file named {self.image_paths[index]} not found"
        return image


    def load_annotation(self, index: int):
        """

        """
        # Load the annotations (class_id, x_center, y_center, width, height)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            label_data = np.loadtxt(self.label_paths[index], ndmin=2)

        if len(label_data) == 0:
            # ASSERT: File is empty
            # Bounding box and class labels are 0
            # print("No bounding boxes present in current image")
            return np.zeros((self.max_labels, 5))
        else:
            # ASSERT: File is NOT empty
            return label_data # [class_id, x_cent, y_cent, width, height]


    def pull_item(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the original image and target at an index for mixup

        Use this function separately before applying any transformations to the image or bounding boxes

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        image = self.load_image(index)          # [H,W,3]
        label = self.load_annotation(index)     # [MAX_LABELS, 5]. 1st column is class id, last 4 columns are bounding boxes in yolo format

        return image, copy.deepcopy(label), image.shape[:2] # Images in this dataset do not have a unique ID

    @logger.catch
    def _apply_augmentations(self, image: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Args:
            image: image to apply data augmentations to
            target: Nx5 numpy area [class_id, x_center, y_center, width, height] (normalized)
        albumentations expects the bboxes to be in this configuration
        bboxes = [
            [23, 74, 295, 388, 'dog'],
            [377, 294, 252, 161, 'cat'],
            [333, 421, 49, 49, 'sports ball'],
        ]"""

        labels = np.zeros_like(target[:,0]).astype(int) # All buildings have a label of 0 # label_data[:,0]
        bboxes = target[:,1:]
        bboxes = np.clip(target[:,1:], 0.001, 1.0)

        # Convert labels and bboxes to list format for albumentations
        albument_bboxes = bboxes.tolist()
        albument_labels = labels.tolist()

        transformed = self.transforms(image=image, bboxes=albument_bboxes, class_labels=albument_labels)
        transformed_image = transformed['image'].float()    # Torch.Tensor [channel, height, width]
        transformed_bboxes = np.array(transformed['bboxes'])
        transformed_class_labels = np.expand_dims(np.array(transformed['class_labels']), 1)

        if len(transformed_bboxes) == 0:
            # ASSERT: Transformations pushed all bounding boxes out of frame
            # logger.warning("Albumentations pushed all bboxes out of frame")
            transformed_bboxes = np.zeros((self.max_labels, 4))
            transformed_class_labels = np.zeros((self.max_labels, 1))

        # Concatenate class labels to the bboxes
        transformed_target = np.hstack((transformed_class_labels, transformed_bboxes))

        channel, height, width = transformed_image.shape
        # Dataset is already in normalized coordinates
        # Let's unnormalize them
        transformed_target[:,1] *= width
        transformed_target[:,2] *= height
        transformed_target[:,3] *= width
        transformed_target[:,4] *= height

        return transformed_image, transformed_target

    @logger.catch
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray, tuple, np.ndarray]:
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox unnormalized
                    w, h (float) : size of bbox unnormalized
            info_img : tuple of h, w.
                h, w (int): original shape of the image
        """
        img, target, img_info = self.pull_item(index)

        if self.transforms is not None:
            img, target = self._apply_augmentations(img, target)
        
        # Pad the array of bboxes to have a length of MAX_LABELS
        num_samples = len(target)
        if num_samples < self.max_labels:
            pad = np.zeros((self.max_labels - num_samples, 5))
            target = np.vstack((target, pad))
        else:
            # Truncate target to keep randomized maximum number of samples
            ind_to_remove = np.random.choice(len(target), size=(num_samples - self.max_labels), replace=False)
            target = np.delete(target, ind_to_remove, axis=0)

        return img, target, img_info


if __name__ == "__main__":
    logger.info("Debugging data loading, data augmentation, and labels")
    img_size = 608

    transforms = A.Compose([
        # Geometric transformations
        A.RandomResizedCrop(img_size, img_size, scale=(0.75, 1.0), always_apply=True), # Crop the image first. Reduces computation time when applying optical or elastic distortion
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.5),

        # Color transformations
        A.OneOf([
            A.Sequential([
                A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, p=0.5),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
                A.ISONoise(color_shift=(0.01, 0.3), intensity=(0.1, 0.7), p=0.5),
            ], p=0.3),
            A.Sequential([
                A.RandomGamma(gamma_limit=(60,100), p=0.5),
            ], p=0.7)
        ]),

        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.2), contrast_limit=(-0.2, 0.7), p=0.5),
        A.CLAHE(p=0.15),

        # Brightness transformations
        # A.Blur(blur_limit=(1,5), p=0.5),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', min_area=512, min_visibility=0.4, label_fields=['class_labels']))

    # NOTE: The annotations are in yolo format so it's representing it as bounding box centers, heights, and widths
    # dataset
    dataset = ThermalBuildingDetection(image_dir="/mnt/data/Datasets/Building_Detection/thermal/val/val_images",
                                       label_dir="/mnt/data/Datasets/Building_Detection/thermal/val/val_labels",
                                       transforms=transforms,
                                       class_ids=[0],
                                       class_labels=['building'],
                                       max_labels=10)


    for i in np.arange(82, 85):
        img, target, img_info = dataset[i]
        logger.info(f"Image Shape: {img.shape}\nTarget Shape: {target.shape}\nImage Info: {img_info}\nImage ID: {img_id}")
        img = img.permute(1,2,0).detach().cpu().numpy()
        height, width, channel = img.shape
        class_id, bboxes = target[:,0], target[:,1:5]

        background = img.copy()
        for box in bboxes:
            # Assuming yolo format for visualization

            # Assuming unnormalized yolo format for visualization
            xc, yc, w, h = box
            col_min = int(xc - w/2)
            row_min = int(yc - h/2)
            col_max = int(xc + w/2)
            row_max = int(yc + h/2)
            background = cv2.rectangle(background, (col_min, row_min), (col_max, row_max), (0,0,255), 2)

        background = (background - np.min(background)) / (np.max(background) - np.min(background))
        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), frameon=True, layout="tight", dpi=300)
        ax.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB)) ; ax.set_title("Bounding Boxes") ; ax.set_axis_off()
        plt.show()
