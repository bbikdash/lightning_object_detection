
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Copyright (c) 2023 KEF Robotics in Pittsburgh, PA

"""
@author Bassam Bikdash
November 2023

This python module contains Pytorch Dataset classes for loading COCO using PyCOCOTools and is based
on Megvii's COCO Dataset class which was published in their YOLOX repo:
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/datasets/coco.py


NOTE: I adjusted pycocotools (coco.py) to have an additional parameter (`inclusive`: bool) in getImgIds().
This allows the function to return the union of the sets (or) when specifying categories to get
image ids of instead of the intersection (and).
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
import copy

import warnings
from typing import Tuple
from bidict import bidict

from loguru import logger
from tools.custom_coco_parser import COCO
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import xyxyCOCO2cxcywh, visualize_bbox_augmentations, convert_label



def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)

class COCODataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str, # train_2014, val_2014, train_
                 categories_to_include: list,
                 transforms: A.Compose,
                 intermediate_class_mapping: dict,
                 target_class_mapping: bidict,
                 max_labels: int = 50) -> None:
        """
        
        I've organized COCO 2014 and 2017 . 
        NOTE: We will use the cropped images for loading and crop/resize them via data augmentations during training
        The <train,val>_bboxes.csv in each csv_folder will always refer to images within <train,val>/images/ and not within cropped/.

        root/
            coco2014/
                annotations/
                images/

            coco2017/
                annotations/
                images/

        Args:
            root: path to the directory containing uav images and labels
            transforms: Albumentations object of transformations to apply to the image
            class_ids: list of ids (ints) representing class predictions (0=person, 1=fish)
            class_labels: list of labels (strings) that correspond to the class ids and represent the name of the class
            max_labels: default tensor size for target. Bounding box labels are padded to [max_labels, 5] for efficient batching.
                        if you get the following error: `RuntimeError: stack expects each tensor to be equal size...`. Increase this number
        """
        if (split != "train2014" and split != "val2014" and
            split != "train2017" and split != "val2017"):
            logger.error("coco_type must be <train2014 | val2014 | train2017 | val2017>")
            raise ValueError

        self.root = root
        self.split = split
        self.SOURCE_CLASS_BIMAP = bidict({
            1: 'person',
            2: 'bicycle',
            3: 'car',
            4: 'motorcycle',
            5: 'airplane',
            6: 'bus',
            7: 'train',
            8: 'truck',
            9: 'boat',
            10: 'traffic light',
            11: 'fire hydrant',
            12: 'street sign',
            13: 'stop sign',
            14: 'parking meter',
            15: 'bench',
            16: 'bird',
            17: 'cat',
            18: 'dog',
            19: 'horse',
            20: 'sheep',
            21: 'cow',
            22: 'elephant',
            23: 'bear',
            24: 'zebra',
            25: 'giraffe',
            26: 'hat',
            27: 'backpack',
            28: 'umbrella',
            29: 'shoe',
            30: 'eye glasses',
            31: 'handbag',
            32: 'tie',
            33: 'suitcase',
            34: 'frisbee',
            35: 'skis',
            36: 'snowboard',
            37: 'sports ball',
            38: 'kite',
            39: 'baseball bat',
            40: 'baseball glove',
            41: 'skateboard',
            42: 'surfboard',
            43: 'tennis racket',
            44: 'bottle',
            45: 'plate',
            46: 'wine glass',
            47: 'cup',
            48: 'fork',
            49: 'knife',
            50: 'spoon',
            51: 'bowl',
            52: 'banana',
            53: 'apple',
            54: 'sandwich',
            55: 'orange',
            56: 'broccoli',
            57: 'carrot',
            58: 'hot dog',
            59: 'pizza',
            60: 'donut',
            61: 'cake',
            62: 'chair',
            63: 'couch',
            64: 'potted plant',
            65: 'bed',
            66: 'mirror',
            67: 'dining table',
            68: 'window',
            69: 'desk',
            70: 'toilet',
            71: 'door',
            72: 'tv',
            73: 'laptop',
            74: 'mouse',
            75: 'remote',
            76: 'keyboard',
            77: 'cell phone',
            78: 'microwave',
            79: 'oven',
            80: 'toaster',
            81: 'sink',
            82: 'refrigerator',
            83: 'blender',
            84: 'book',
            85: 'clock',
            86: 'vase',
            87: 'scissors',
            88: 'teddy bear',
            89: 'hair drier',
            90: 'toothbrush',
            91: 'hair brush'
        })

        self.intermediate_class_mapping = intermediate_class_mapping
        self.target_class_mapping = target_class_mapping

        # initialize COCO api for instance annotations
        annotation_file = f"{root}/coco{split[-4:]}/annotations/instances_{split}.json"
        
        # 
        self.coco = COCO(annotation_file)
        remove_useless_info(self.coco)

        # get all images containing given categories, select one at random
        # Check if category is a valid COCO class, if not skip it
        for cate in categories_to_include:
            if cate not in self.SOURCE_CLASS_BIMAP.values():
                logger.error(f"{cate} is not a valid category in COCO. Skipping...")
    
        # Get all of the image ids for that category
        category_ids = self.coco.getCatIds(catNms=categories_to_include)
        self.img_ids = self.coco.getImgIds(catIds=category_ids, inclusive=True)

        # A set of composed Albumentation transformations (can be for training or validation)
        new_transforms = [A.CropAndPad(percent=(0.0, 1.25), p=0.1)]
        for t in transforms:
            if isinstance(t, A.RandomResizedCrop):
                t.scale = (0.4, 1.0)
            elif isinstance(t, A.RandomRotate90):
                t.p = 0.00001
            elif isinstance(t, A.ShiftScaleRotate):
                t.shift_limit = (-0.1, 0.1)
                t.scale_limit = (1.0, 1.0)
                t.rotate_limit = (-5, 5)
            new_transforms.append(t)
        self.transforms = A.Compose(new_transforms, bbox_params=transforms.processors['bboxes'].params)

        # Maximum number of labels per image
        self.max_labels = max_labels

    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.img_ids)
    

    def load_image(self, index: int) -> np.ndarray:
        # Get the file name from the current img_id
        image_name = self.coco.loadImgs(self.img_ids[index])[0].get('file_name')
        image_path = f"{self.root}/coco{self.split[-4:]}/images/{self.split}/{image_name}"

        # Load image in color (to maintain 3 channels)
        image = cv2.imread(image_path, 1)
        assert image is not None, f"file named {image_path} not found"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    

    def load_annotation(self, index: int) -> np.ndarray:
        # Get the name of the image
        annIds = self.coco.getAnnIds(imgIds=self.img_ids[index])
        anns = self.coco.loadAnns(annIds)

        label_data = np.zeros((len(anns), 5))
        for i in range(len(anns)):
            id = np.array(anns[i].get("category_id"), ndmin=2)
            bbox = np.array(anns[i].get("bbox"), ndmin=2)

            # Convert bbox type
            bbox = xyxyCOCO2cxcywh(bbox)
            label_data[i] = np.hstack((id, bbox))

        # Convert src labels to target labels
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
        bboxes = np.clip(bboxes, 1e-6, 1.0)
        # print(bboxes)

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



if __name__ == "__main__":
    logger.info("Debugging data loading, data augmentation, and labels")
    img_size = 608
    
    transforms = A.Compose([
        # Geometric transformations
        A.RandomResizedCrop(img_size, img_size, scale=(0.4, 1.0), always_apply=True), # Crop the image first. Reduces computation time when applying optical or elastic distortion
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30),
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
    
    target_class_mapping = bidict({
        0: 'building',
        1: 'vehicle',
        2: 'person',
        3: 'window',
        4: 'door'
    })
    intermediate_class_mapping = {
        'person': 'person',
        'bicycle': 'vehicle',
        'car': 'vehicle',
        'motorcycle': 'vehicle',
        'airplane': 'vehicle',
        'bus': 'vehicle',
        'train': 'vehicle',
        'truck': 'vehicle',
        'boat': 'vehicle'
    }
    dataset = COCODataset(root="/mnt/data/Datasets/COCO",
                          split='train2017',
                          categories_to_include=['person',
                                                 'bicycle',
                                                 'car',
                                                 'motorcycle',
                                                 'airplane',
                                                 'bus',
                                                 'train',
                                                 'truck',
                                                 'bus'],
                          transforms=transforms,
                          intermediate_class_mapping=intermediate_class_mapping,
                          target_class_mapping=target_class_mapping,
                          max_labels=30)

    if not hasattr(dataset, "SOURCE_CLASS_BIMAP"):
        logger.error("Dataset MUST have SOURCE_CLASS_BIMAP attribute mapping integers to class names (str)")
        sys.exit(1)


    for i in np.arange(len(dataset)):
        img, target, img_info = dataset[i]
        assert target.shape[0] == 30
        logger.info(f"Image Shape: {img.shape}\nTarget Shape: {target.shape}\nImage Info: {img_info}")
        visualize_bbox_augmentations(dataset, i, 1)
    
