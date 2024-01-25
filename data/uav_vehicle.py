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
import sys
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import pdb
from bidict import bidict

import copy

import warnings
from typing import Tuple

from loguru import logger
from utils import xyxy2cxcywh, visualize_bbox_augmentations, convert_label


class UAVVehicleDetection(Dataset):
    def __init__(self,
                 root: str,
                 split:str,
                 transforms: A.Compose,
                 intermediate_class_mapping: dict,
                 target_class_mapping: bidict,
                 max_labels: int = 50) -> None:
        """
        
        Args:
            data_dir: path to the directory containing images
            label_dir: path to the directory containing bounding box labels as text files
            transforms: Albumentations object of transformations to apply to the image
            class_ids: list of ids (ints) representing class predictions (0=person, 1=fish)
            class_labels: list of labels (strings) that correspond to the class ids and represent the name of the class
            max_labels: default tensor size for target. Bounding box labels are padded to [max_labels, 5] for efficient batching.
                        if you get the following error: `RuntimeError: stack expects each tensor to be equal size...`. Increase this number
        """
        self.root = root
        self.image_dir = os.path.join(root, f"data/{split}")
        self.label_dir = os.path.join(root, f"data/{split}")        # Find the images in the current images directory
        
        self.image_names = [ name for name in sorted(os.listdir(self.image_dir)) if name.endswith('png') or name.endswith('jpg') or name.endswith('JPG') ]
        self.image_paths = [ os.path.join(self.image_dir, image_name) for image_name in self.image_names ]
        self.label_names = [ name for name in sorted(os.listdir(self.label_dir)) if name.endswith('txt') ]
        self.label_paths = [ os.path.join(self.label_dir, image_name) for image_name in self.label_names ]
        assert(len(self.image_names) == len(self.label_names))

        self.SOURCE_CLASS_BIMAP = bidict({
            0: 'car',
            1: 'truck',
            2: 'bus',
            3: 'trailer'
        })
        self.intermediate_class_mapping = intermediate_class_mapping
        self.target_class_mapping = target_class_mapping
        self.name = "uav_aerial_dataset"

        # A set of composed Albumentation transformations (can be for training or validation)
        self.transforms = transforms

        # Maximum number of labels per image
        self.max_labels = max_labels


    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.image_paths)
    

    def load_image(self, index: int) -> np.ndarray:
        # Load image in color (to maintain 3 channels)
        image = cv2.imread(self.image_paths[index], 1)

        # Image will be converted to gray when transformations are applied
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
            label_data = np.zeros((1, 5))

        # Convert source labels to target labels 
        label_data = convert_label(label_data,
                                   self.SOURCE_CLASS_BIMAP,
                                   self.intermediate_class_mapping,
                                   self.target_class_mapping)

        return label_data
            

    def pull_item(self, index: int):
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
    

    def _apply_augmentations(self, image: np.ndarray, target: np.ndarray):

        """ albumentations expects the bboxes to be in this configuration
        bboxes = [
            [23, 74, 295, 388, 'dog'],
            [377, 294, 252, 161, 'cat'],
            [333, 421, 49, 49, 'sports ball'],
        ]"""

        labels = target[:,0].astype(int) # label_data[:,0]
        bboxes = target[:,1:]
        bboxes = np.clip(bboxes, np.finfo(np.float32).tiny, 1.0)

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

        # UAV Dataset is already in normalized coordinates
        # Let's unnormalize them
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
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45),

        # Color transformations
        # A.CLAHE(),
        # A.RandomGamma(gamma_limit=(60,80), p=0.5),
        # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.5), p=0.5),
        # A.Blur(blur_limit=(1,5), p=0.5),
        # A.ISONoise(color_shift=(0.01, 0.3), intensity=(0.1, 0.7), p=0.5),
        # A.PixelDropout(p=0.1),
        # A.OneOf([
        #     A.RandomRain(blur_value=1, brightness_coefficient=0.9),
        #     A.RandomSnow(brightness_coeff=1.0),
        #     A.RandomShadow(),
        # ]),
        A.RandomResizedCrop(img_size, img_size, scale=(0.04, 1), always_apply=True),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo',  min_area=256, min_visibility=0.1, label_fields=['class_labels']))
    
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
        'car': 'vehicle',
        'truck': 'vehicle',
        'bus': 'vehicle',
        'trailer': 'vehicle',
    }
    dataset = UAVVehicleDetection(root='/mnt/data/Datasets/Vehicle_Detection/UAV-Vehicle-Detection-Dataset',
                                  split='train',
                                  transforms=transforms,
                                  intermediate_class_mapping=intermediate_class_mapping,
                                  target_class_mapping=target_class_mapping,
                                  max_labels=50)

    if not hasattr(dataset, "SOURCE_CLASS_BIMAP"):
        logger.error("Dataset MUST have SOURCE_CLASS_BIMAP attribute mapping integers to class names (str)")
        sys.exit(1)

    for i in np.arange(len(dataset)):
        img, target, img_info = dataset[i]
        logger.info(f"Image Shape: {img.shape}\nTarget Shape: {target.shape}\nImage Info: {img_info}")
        visualize_bbox_augmentations(dataset, i, 1)
        