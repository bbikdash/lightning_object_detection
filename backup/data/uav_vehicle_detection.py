


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
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import utils.visualization as u
from typing import Dict, List, Optional, Any
import albumentations as A


class UAVVehicleDetection(Dataset):
    def __init__(self, data_dir: str or List, label_dir: str or List, transforms: A.Compose) -> None:
        """
        
        Args:
            data_dir: path to the image directory. The folder should contain .jpg files
            label_dir: path to the label directory. The folder should contain .txt files
            transforms: Albumentations object of transformations to apply to the image

        NOTE: It is okay if the labels and the images are in the same directory. Images will be searched for
        by using .jpg or .png extensions and the labels will be searched for with the .txt extension.
        NOTE: Each image and corresponding label should have the same name
        """

        # Convert data_dir to list so I don't have the change the code below
        if data_dir is str:
            data_dir = [data_dir]

        self.image_names = []
        self.image_paths = []
        for folder in data_dir:
            # Find the images in the current images directory
            current_image_names = [ name for name in sorted(os.listdir(folder)) if name.endswith('png') or name.endswith('jpg') ]
            current_image_paths = [ os.path.join(folder, image_name) for image_name in current_image_names ]
            # Add these paths to the running list of names and paths
            self.image_names.extend(current_image_names)
            self.image_paths.extend(current_image_paths)

        assert(len(self.image_names) == len(self.image_paths))

        self.label_names = []
        self.label_paths = []
        for folder in label_dir:
            # Find the labels in the current labels directory
            current_label_names = [ name for name in sorted(os.listdir(folder)) if name.endswith('txt') ]
            current_label_paths = [ os.path.join(folder, image_name) for image_name in current_label_names ]
            # Add these paths to the running list of names and paths for labels
            self.label_names.extend(current_label_names)
            self.label_paths.extend(current_label_paths)

        assert(len(self.image_names) == len(self.label_names))
        # print(self.image_names[0][-20:], self.label_names[0][-20:])
        assert(self.image_names[0][-10:] == self.label_names[0][-10:])

        # A set of composed Albumentation transformations (can be for training or validation)
        self.transforms = transforms

    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Loads and returns a sample from the dataset at the given index idx
        See: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        """
        # Load image and it's corresponding label
        image = cv2.imread(self.image_paths[idx], 0)

        # TODO: Ask Eric, does building detection require a grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.array(image)

        # Apply the transformations to the image and label
        if self.transforms:
            transformed = self.transforms(image=image)
            transformed_image = transformed['image']

        # Convert the transformed label to hesse normal form in numpy for computation in the loss function and convert back to tensor
        return transformed_image, label # 3xHxW, int 

