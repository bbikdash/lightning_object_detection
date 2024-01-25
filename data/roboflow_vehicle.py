#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2023 KEF Robotics in Pittsburgh, PA

"""
@author Bassam Bikdash
February 2023

This python module contains Pytorch Dataset classes to load a soccer player detection dataset
provided here: https://github.com/newsdata/SoccerDB/tree/master/dataset/detection_dataset

The original class mapping is:
0: player
1: ball
2: goal

with the original bounding box labels in the following format: [class_id, center_x, center_y, width, height]
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
import json
import copy
from bidict import bidict

import warnings
from typing import Tuple

from loguru import logger
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import xyxy2cxcywh, visualize_bbox_augmentations, convert_label

class RoboflowVehicleDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transforms: A.Compose,
                 intermediate_class_mapping: dict,
                 target_class_mapping: bidict,
                 max_labels: int = 50) -> None:
        """
        
        The Inria Aerial Dataset has the following directory structure. 
        NOTE: We will use the cropped images for loading and crop/resize them via data augmentations during training
        The <train,val>_bboxes.csv in each csv_folder will always refer to images within <train,val>/images/ and not within cropped/.

        root/
            images/
                train/ 
                    sample_0.jpg
                    sample_1.jpg
                    ...
                val/
                    sample_815.jpg
                    sample_816.jpg
                    ...
            annotations/
                train/
                    sample_0.txt
                    sample_1.txt
                    ...
                val/
                    sample_815.txt
                    sample_816.txt
                    ...

        Args:
            root: path to the directory containing uav images and labels
            transforms: Albumentations object of transformations to apply to the image
            class_ids: list of ids (ints) representing class predictions (0=person, 1=fish)
            class_labels: list of labels (strings) that correspond to the class ids and represent the name of the class
            max_labels: default tensor size for target. Bounding box labels are padded to [max_labels, 5] for efficient batching.
                        if you get the following error: `RuntimeError: stack expects each tensor to be equal size...`. Increase this number
        """
        self.root = root

        self.image_dir = os.path.join(root, f"{split}/images/")
        self.annotation_dir = os.path.join(root, f"{split}/labels")

        self.image_names = [ name for name in sorted(os.listdir(self.image_dir)) if name.endswith('png') or name.endswith('jpg') or name.endswith('JPG') ]
        self.image_paths = [ os.path.join(self.image_dir, image_name) for image_name in self.image_names ]
        self.label_names = [ name for name in sorted(os.listdir(self.annotation_dir)) if name.endswith('txt') ]
        self.label_paths = [ os.path.join(self.annotation_dir, image_name) for image_name in self.label_names ]
        assert(len(self.image_names) == len(self.label_names))

        # Training or validation: <train, val>
        self.split = split

        self.SOURCE_CLASS_BIMAP = bidict({
            0: 'big bus',
            1: 'big truck',
            2: 'bus-l',
            3: 'bus-s',
            4: 'car',
            5: 'mid truck',
            6: 'small bus',
            7: 'small truck',
            8: 'truck-l',
            9: 'truck-m',
            10: 'truck-s',
            11: 'truck-xl'
        })
        self.intermediate_class_mapping = intermediate_class_mapping
        self.target_class_mapping = target_class_mapping
        self.name = "soccer_player_dataset"

        # A set of composed Albumentation transformations (can be for training or validation)SoccerPlayerDataset
        self.transforms = transforms

        # Maximum number of labels per image
        self.max_labels = max_labels


    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.label_paths)
    

    def load_image(self, index: int) -> np.ndarray:
        # Load image in color (to maintain 3 channels)
        image = cv2.imread(self.image_paths[index], 1)

        # Image will be converted to gray when transformations are applied
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert image is not None, f"file named {self.image_paths[index]} not found"
        return image
    

    def load_annotation(self, index: int) -> np.ndarray:
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
            return np.zeros((1, 5))
        else:
            # ASSERT: File is NOT empty
            # Convert source labels to target labels 
            label_data = convert_label(label_data,
                                       self.SOURCE_CLASS_BIMAP,
                                       self.intermediate_class_mapping,
                                       self.target_class_mapping)
            
            return label_data   # [class_id, x_center, y_center, width, height]


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

        return image, copy.deepcopy(label), image.shape[:2]     # Images in this dataset do not have a unique ID
    
    @logger.catch
    def _apply_augmentations(self, image: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:

        """
        Args:
            image: image to apply data augmentations to
            target: Nx5 numpy area [class_id, x_center, y_center, width, height] (unnormalized)
        albumentations expects the bboxes to be in this configuration
        bboxes = [
            [23, 74, 295, 388, 'dog'],
            [377, 294, 252, 161, 'cat'],
            [333, 421, 49, 49, 'sports ball'],
        ]"""

        labels = target[:,0].astype(int) # All buildings have a label of 0
        bboxes = target[:,1:]
        bboxes = np.clip(bboxes, np.finfo(np.float32).tiny, 1.0)

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

        # Unnormalize coords for yolox
        transformed_bboxes[:,0] = transformed_bboxes[:,0] * transformed_image.shape[2]
        transformed_bboxes[:,1] = transformed_bboxes[:,1] * transformed_image.shape[1]
        transformed_bboxes[:,2] = transformed_bboxes[:,2] * transformed_image.shape[2]
        transformed_bboxes[:,3] = transformed_bboxes[:,3] * transformed_image.shape[1]

        # Concatenate class labels to the bboxes
        transformed_target = np.hstack((transformed_class_labels, transformed_bboxes))

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
        A.RandomResizedCrop(img_size, img_size, scale=(0.5, 1.0), always_apply=True), # Crop the image first. Reduces computation time when applying optical or elastic distortion
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30),
        # A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.5),

        # Color transformations
        A.ChannelShuffle(p=0.5),
        A.OneOf([
            A.Sequential([
                A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, p=0.5),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
                # A.ISONoise(color_shift=(0.01, 0.3), intensity=(0.1, 0.7), p=0.5),
            ]),
            A.Sequential([
                # A.ToGray(p=0.5),
                A.RandomGamma(gamma_limit=(60,100), p=0.5),
            ])
        ]),

        # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.1, 0.5), p=0.5),
        # A.CLAHE(p=0.15),

        # Brightness transformations
        # A.Blur(blur_limit=(1,5), p=0.5),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', min_area=256, min_visibility=0.1, label_fields=['class_labels']))
    
    # NOTE: The annotations are in yolo format so it's representing it as bounding box centers, heights, and widths
    # dataset
    target_class_mapping = bidict({
        0: 'building',
        1: 'vehicle',
        2: 'person',
        3: 'window',
        4: 'door'
    })
    intermediate_class_mapping = {
        'big bus': 'vehicle',
        'big truck': 'vehicle',
        'bus-l': 'vehicle',
        'bus-s': 'vehicle',
        'car': 'vehicle',
        'mid truck': 'vehicle',
        'small bus': 'vehicle',
        'small truck': 'vehicle',
        'truck-l': 'vehicle',
        'truck-m': 'vehicle',
        'truck-s': 'vehicle',
        'truck-xl': 'vehicle'
    }
    dataset = RoboflowVehicleDataset(root="/mnt/data/Datasets/Vehicle_Detection/RoboFlow_Vehicles",
                                  transforms=transforms,
                                  split="val",
                                  intermediate_class_mapping=intermediate_class_mapping,
                                  target_class_mapping=target_class_mapping,
                                  max_labels=30)

    if not hasattr(dataset, "SOURCE_CLASS_BIMAP"):
        logger.error("Dataset MUST have SOURCE_CLASS_BIMAP attribute mapping integers to class names (str)")
        sys.exit(1)

    for i in np.arange(len(dataset)):
        img, target, img_info = dataset[i]
        assert target.shape[0] == 30
        # logger.info(f"Image Shape: {img.shape}\nTarget Shape: {target.shape}\nImage Info: {img_info}")
        # visualize_bbox_augmentations(dataset, i, 1)
    