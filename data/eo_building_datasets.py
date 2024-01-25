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
import json
import copy
from bidict import bidict

import warnings
from typing import Tuple

from loguru import logger
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import xyxy2cxcywh, visualize_bbox_augmentations, convert_label

class InriaBuildingDetection(Dataset):
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
            train/  
                csv_folder/
                    cropped_train_bboxes.csv
                    train_bboxes.csv
                gt/
                    austin1.tif
                    ...
                    cropped/
                        austin1_0.tif
                        ...
                images/
                    austin1.tif
                    austin2.tif
                    ...
                    cropped/
                        austin1_0.tif
                        austin1_1.tif
                        ...
            val/
                csv_folder/
                    val_bboxes.csv
                gt/
                images/
                    vienna1.tif
                    ...
                    cropped/
                        vienna1_0.tif
            test/
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
        self.image_dir = os.path.join(root, f"{split}/images")
        self.cropped_image_dir = os.path.join(root, f"{split}/images/cropped")
        self.csv_path = os.path.join(root, f"{split}/csv_folder/{split}_bboxes.csv")
        self.cropped_csv_path = os.path.join(root, f"{split}/csv_folder/cropped_{split}_bboxes.csv")

        # Load the bboxes from Inria root
        # Use the names to refer to associate names for images
        csv_data = np.loadtxt(self.cropped_csv_path, delimiter=',', dtype=str)
        self.format = csv_data[0]
        csv_data = csv_data[1:]

        self.annotations = self._process_annotations(csv_data)

        # Training or validation: <train, val>
        self.split = split

        self.SOURCE_CLASS_BIMAP = bidict({
            0: 'building',
        })
        self.intermediate_class_mapping = intermediate_class_mapping
        self.target_class_mapping = target_class_mapping
        self.name = "inria_satellite_dataset"

        # A set of composed Albumentation transformations (can be for training or validation)
        for t in transforms:
            if isinstance(t, A.RandomResizedCrop):
                t.scale = (0.15, 0.7)
            elif isinstance(t, A.RandomRotate90):
                t.p = 0.5
        self.transforms = transforms

        # Maximum number of labels per image
        self.max_labels = max_labels


    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.annotations)
    

    def _process_annotations(self, csv_bboxes: List[List[str]]) -> np.ndarray:
        annotations = {}
        for line in csv_bboxes:
            image_name, bbox = line[1], line[2:].astype(int)    # string, [xmin, ymin, xmax, ymax]

            current_bboxes = annotations.get(image_name)
            if current_bboxes is None:
                # Image sample has not been encountered before. Add it to the dictionary
                annotations[image_name] = bbox
            else:
                annotations[image_name] = np.vstack((current_bboxes, bbox))

        return annotations  # Nx4 np.ndarray
    

    def load_image(self, index: int) -> np.ndarray:
        # Get the name of the image
        k = list(self.annotations.keys())[index]
        # Create the image to the image
        image_path = os.path.join(self.cropped_image_dir, f"{k}.tif")

        # Load image in color (to maintain 3 channels)
        image = cv2.imread(image_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Image will be converted to gray when transformations are applied
        assert image is not None, f"file named {image_path} not found"
        return image
    

    def load_annotation(self, index: int) -> np.ndarray:
        # Get the name of the image
        k = list(self.annotations.keys())[index]
        bboxes = self.annotations[k].copy()  # [xmin, ymin, xmax, ymax]

        if len(bboxes.shape) == 1:
            # Array is 1-dimensional. Add a leading dimension
            bboxes = np.expand_dims(bboxes, axis=0)

        # Convert bbox into yolo format (unnormalized)
        bboxes = xyxy2cxcywh(bboxes)
        
        # Append the class ids to the annotations (all class_ids are 0 since there is only 1 class)
        label_data = np.hstack((np.zeros((len(bboxes),1)), bboxes))

        # Convert source labels to target labels 
        label_data = convert_label(label_data,
                                   self.SOURCE_CLASS_BIMAP,
                                   self.intermediate_class_mapping,
                                   self.target_class_mapping)

        return label_data   # [class_id, xmin, ymin, xmax, ymax]


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
            target: Nx5 numpy area [class_id, x_center, y_center, width, height] (unnormalized)
        albumentations expects the bboxes to be in this configuration
        bboxes = [
            [23, 74, 295, 388, 'dog'],
            [377, 294, 252, 161, 'cat'],
            [333, 421, 49, 49, 'sports ball'],
        ]"""

        labels = target[:,0].astype(int) # All buildings have a label of 0
        bboxes = target[:,1:]

        # Normalize the bounding box coordinates for yolo style augmentations
        bboxes[:,0] = bboxes[:,0] / image.shape[1]
        bboxes[:,1] = bboxes[:,1] / image.shape[0]
        bboxes[:,2] = bboxes[:,2] / image.shape[1]
        bboxes[:,3] = bboxes[:,3] / image.shape[0]
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
            img_id (int): same as the input index. Used for evaluation.
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



class KEFETGBuildingDetection(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transforms: A.Compose,
                 class_ids: list,
                 class_labels: list,
                 max_labels: int = 50) -> None:
        """
        
        The KEF ETG Building Dataset has the following directory structure. 
        NOTE: The JSON file contains the file names, file locations, and bounding boxes for each subfolder in the root directory (all
        relative to the root directory as well). We will use this prior knowledge to hardcode file paths.

        root/
            train/
                11_17_08/  
                    11_17_08_img1.png
                    11_17_08_img2.png
            val/

            etg.json

        Args:
            root: path to the directory containing uav images and labels
            transforms: Albumentations object of transformations to apply to the image
            class_ids: list of ids (ints) representing class predictions (0=person, 1=fish)
            class_labels: list of labels (strings) that correspond to the class ids and represent the name of the class
            max_labels: default tensor size for target. Bounding box labels are padded to [max_labels, 5] for efficient batching.
                        if you get the following error: `RuntimeError: stack expects each tensor to be equal size...`. Increase this number
        """
        self.root = root

        self.image_dir = os.path.join(root, split)
        self.json_path = os.path.join(root, "etg.json")

        # Find the images in the current images directory and subdirectories
        self.image_paths, self.image_names = self._get_image_paths(self.image_dir)

        # Load the bboxes from ETG root
        # Use the names to refer to associate names for images
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            raise

        # Returns a dictionary {file_name: nd.array of bboxes (Nx4)}
        self.annotations = self._process_annotations(data)

        # Training or validation: <train, val>
        self.split = split

        self.class_ids = class_ids
        self.class_labels = class_labels
        self.name = "kef_etg_dataset"

        # A set of composed Albumentation transformations (can be for training or validation)
        self.transforms = transforms

        # Maximum number of labels per image
        self.max_labels = max_labels


    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.image_paths)
    
    def _get_image_paths(self, directory):
        # Initialize lists to store file paths and file names
        file_paths = []
        file_names = []
        for root, _, files in os.walk(directory):
            for filename in files:
                # Check if the file has a .png or .jpg extension
                if filename.lower().endswith(('.png', '.jpg')):
                    # Get the absolute file path
                    file_path = os.path.join(root, filename)

                    # Get the file name with extension (without parent directories)
                    file_name = os.path.relpath(file_path, directory)

                    # Append the file path and file name to the respective lists
                    file_paths.append(file_path)
                    file_names.append(file_name)
        return file_paths, file_names
    

    def _process_annotations(self, json_data):
        """
        
        annotations: dict, key,value => [img_file_path (str): bboxes (np.ndarray)]
        """
        annotations = {}
        for image_filepath, image_data in json_data.items():
            # Extract the relevant fields
            objects = image_data.get("objects", [])

            # File name of the image is the key    
            image_filename = os.path.basename(image_filepath)

            # Iterate through every bbox dictionary in the list
            for bbox_obj in objects:

                # Retrieve unnormalized yolo style bbox coords
                bbox = np.array([bbox_obj.get('xcent'),
                                 bbox_obj.get('ycent'),
                                 bbox_obj.get('width'),
                                 bbox_obj.get('height')]) # string, [xcent, ycent, width, height]
                
                # Check if key has been added to the annotations dict
                current_bboxes = annotations.get(image_filename)
                if current_bboxes is None:
                    # Image sample has not been encountered before. Add it to the dictionary
                    annotations[image_filename] = bbox
                else:
                    # Initial bbox was already added. Add new one to the list
                    annotations[image_filename] = np.vstack((current_bboxes, bbox))
        return annotations
    

    def load_image(self, index: int) -> np.ndarray:
        # Get the path to the image
        image_path = self.image_paths[index]

        # Load image in color (to maintain 3 channels)
        image = cv2.imread(image_path, 1)
        # Image will be converted to gray when transformations are applied
        assert image is not None, f"file named {image_path} not found"

        return image
    

    def load_annotation(self, index: int):
        # Get the name of the image
        k = os.path.basename(self.image_paths[index])
        bboxes = self.annotations.get(k,
                                      np.zeros((self.max_labels, 4)) ) # [xcent, ycent, width, height]
    
        if len(bboxes.shape) == 1:
            # Array is 1-dimensional. Add a leading dimension
            bboxes = np.expand_dims(bboxes, axis=0)

        # Append the class ids to the annotations (all class_ids are 0 since there is only 1 class)
        label_data = np.hstack((np.zeros((len(bboxes),1)), bboxes))   # [class_id, xmin, ymin, xmax, ymax]

        return label_data


    def pull_item(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            target: Nx5 numpy area [class_id, x_center, y_center, width, height] (unnormalized)
        albumentations expects the bboxes to be in this configuration
        bboxes = [
            [23, 74, 295, 388, 'dog'],
            [377, 294, 252, 161, 'cat'],
            [333, 421, 49, 49, 'sports ball'],
        ]"""

        labels = target[:,0].astype(int) # label_data[:,0]
        bboxes = target[:,1:]

        # Normalize the bounding box coordinates for yolo style augmentations
        bboxes[:,0] = bboxes[:,0] / image.shape[1]
        bboxes[:,1] = bboxes[:,1] / image.shape[0]
        bboxes[:,2] = bboxes[:,2] / image.shape[1]
        bboxes[:,3] = bboxes[:,3] / image.shape[0]
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
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int], np.ndarray]:
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
            img_id (int): same as the input index. Used for evaluation.
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
    img_size = 640
    
    transforms = A.Compose([
        # Geometric transformations
        A.RandomResizedCrop(img_size, img_size, scale=(0.3, 0.8), always_apply=True), # Crop the image first. Reduces computation time when applying optical or elastic distortion
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30),
        # A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.5),

        # Color transformations
        # A.OneOf([
        #     A.Sequential([
        #         A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, p=0.5),
        #         A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
        #         A.ISONoise(color_shift=(0.01, 0.3), intensity=(0.1, 0.7), p=0.5),
        #     ]),
        #     A.Sequential([
        #         A.ToGray(p=0.5),
        #         A.RandomGamma(gamma_limit=(60,100), p=0.5),
        #     ])
        # ]),

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
        'building': 'building'
    }
    dataset = InriaBuildingDetection(root="/mnt/data/Datasets/Building_Detection/eo/Inria_Aerial_Image_Labeling_Dataset",
                                     transforms=transforms,
                                     split="train",
                                     intermediate_class_mapping=intermediate_class_mapping,
                                     target_class_mapping=target_class_mapping,
                                     max_labels=30)
    # dataset = KEFETGBuildingDetection(root="/mnt/data/Datasets/Building_Detection/eo/ETG",
    #                                 transforms=transforms,
    #                                 split="val",
    #                                 class_ids=[0],
    #                                 class_labels=['building'],
    #                                 max_labels=15)

    if not hasattr(dataset, "SOURCE_CLASS_BIMAP"):
        logger.error("Dataset MUST have SOURCE_CLASS_BIMAP attribute mapping integers to class names (str)")
        sys.exit(1)

    for i in np.arange(len(dataset)):
        img, target, img_info = dataset[i]
        # assert target.shape[0] == 30
        # logger.info(f"Image Shape: {img.shape}\nTarget Shape: {target.shape}\nImage Info: {img_info}")
        # visualize_bbox_augmentations(dataset, i, 1)
