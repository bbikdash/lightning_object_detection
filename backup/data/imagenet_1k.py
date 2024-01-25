# Copyright (c) 2023 KEF Robotics in Pittsburgh, PA

"""
@author Bassam Bikdash
February 2023

This python module contains Pytorch Dataset classes to load the ImageNet dataset.
"""

import os
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from typing import Dict, List, Optional, Any
import albumentations as A


class ImageNet(Dataset):
    def __init__(self, data_dir: str, transforms: A.Compose) -> None:
        """
        This dataset class is used to load the training portion of the data from ImageNet 2017.
        Here I will modify the Lightning AI method for ImageNet of allocating some of the training set
        for validation. Instead, I will use the training set in it's entirety and use some of the
        validation set for testing. The test set provided in ImageNet 2017 from Kaggle is unlabeled and pretty useless
        for training.

        Initialize the directory containing the images, the annotations file, and both transforms.
        
        The user specified the number of classes to use for training and the number of images per
        class for validation.

        Args:
            data_dir: path to the ImageNet training data .../Data/CLS-LOC/train
            transforms: Albumentations object of transformations to apply to the image
        """
        # https://pytorch.org/vision/main/generated/torchvision.data.imageFolder.html
        # (3xHxW Pil image, int label) = dataset[index]. `label` corresponds to the folder name in the directory
        self.dataset = ImageFolder(data_dir)
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
        # Images loaded with ImageFolder are PIL images
        image, label = self.dataset[idx]
        image = np.array(image)

        # Apply the transformations to the image and label
        if self.transforms:
            transformed = self.transforms(image=image)
            transformed_image = transformed['image']

        # Convert the transformed label to hesse normal form in numpy for computation in the loss function and convert back to tensor
        return transformed_image, label # 3xHxW, int 


class ImageNetValidation(Dataset):
    """
    This dataset class is used to load the validation/testing portion of the data from ImageNet 2017.
    Loading the validation set requires it's own logic.

    As of March 6, 2023, this Dataset class is obsolete. I wrote a bash script that will preprocess the
    validation directory to have the exact same directory structure as the training set. The regular
    ImageNet() class above can be used the validation set with greater ease.
    
    Args:
        data_dir: path to the directory containing the validation images. 
        val_solution_csv: path to the solution CSV file for the validation set. Contains
            image names and their corresponding labels (and bounding boxes)
        transforms: Albumentations object of transformations to apply to the image
    """

    def __init__(self, data_dir: str, transforms: A.Compose, val_solution_csv: str, class_mapping: dict) -> None:
        super().__init__()

        self.data_dir = data_dir
        val_solution_dataframe = pd.read_csv(val_solution_csv)
        
        self.image_names = list(val_solution_dataframe['ImageId'])
        self.image_paths = [ os.path.join(self.data_dir, image_name) + '.JPEG' for image_name in self.image_names ]

        prediction_strings = list(val_solution_dataframe["PredictionString"])
        self.label_names = [ string.split(' ')[0] for string in prediction_strings ]
        self.labels = [ class_mapping[name] for name in self.label_names ]

            # A set of composed Albumentation transformations (can be for training or validation)
        self.transforms = transforms


    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset
        """
        return len(self.image_names)

    def __getitem__(self, index: int):
        """
        Loads and returns a sample from the dataset at the given index
        See: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        """
        # Load image and it's corresponding label (HxWx3)
        image = cv2.imread(self.image_paths[index], 1)

        # Convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[index]

        # Apply the transformations to the image and label
        if self.transforms:
            transformed = self.transforms(image=image)
            transformed_image = transformed['image']

        # Convert the transformed label to hesse normal form in numpy for computation in the loss function and convert back to tensor
        return transformed_image, label # 3xHxW, int 


class TinyImageNet(Dataset):
    def __init__(self, data_dir: str, transforms: A.Compose) -> None:
        """
        Args:
            data_dir: path to the TinyImageNet training data .../tiny-imagenet-200/train
            transforms: Albumentations object of transformations to apply to the image
        """

        self.label_names = os.listdir(data_dir)
        self.labels = []
        for l, i in enumerate(self.label_names):
            
            path_to_label_dir = os.path.join(data_dir, l, 'images')



            self.labels.append(i)

        # TODO: Prepare backup plan for training DarkNet19 to become a somewhat decent feature extractor
        
        # https://pytorch.org/vision/main/generated/torchvision.data.imageFolder.html
        # (3xHxW Pil image, int label) = dataset[index]. `label` corresponds to the folder name in the directory
        self.dataset = ImageFolder(data_dir)
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
        # Images loaded with ImageFolder are PIL images
        image, label = self.dataset[idx]
        image = np.array(image)

        # Apply the transformations to the image and label
        if self.transforms:
            transformed = self.transforms(image=image)
            transformed_image = transformed['image']

        # Convert the transformed label to hesse normal form in numpy for computation in the loss function and convert back to tensor
        return transformed_image, label # 3xHxW, int
    