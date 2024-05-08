#!/usr/bin/env python
"""
@author Bassam Bikdash

Lightning Data Module
Combines datasets, defines data augmentations, creates dataloaders for detection task
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
from lightning.pytorch import LightningDataModule
from typing import List, Tuple, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from bidict import bidict
from loguru import logger
import yaml

from lightning_object_detection.utils import visualize_bbox_augmentations
from lightning_object_detection.data import *

def check_type(iterable, tp):
    return all(isinstance(item, tp) for item in iterable)

# Verify image dimensions
def verify_image_resize(original_size:List[int], target_size: List[int]):
    """
    original_size: [height, width] of image
    target_size: [height, width] of resized image. Image dimensions must be multiple of 32 for input into YOLOX
    """
    h,w = original_size
    if target_size[0] == -1:
        if h % 32 != 0:
            h = int(round(h / 32) * 32)
    if target_size[1] == -1:
        if w % 32 != 0:
            w = int(round(w / 32) * 32)
    return h,w

class FakeModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()


class DroneNetDataModule(LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self,
                 train_datasets: List[Dict],
                 val_datasets: List[Dict],
                 test_datasets: List[Dict],
                 train_batch_size: int,
                 val_batch_size: int,
                 train_image_size: list,
                 val_image_size: list,
                 num_workers: int,
                 verify: bool,
                 target_class_mapping: str):
        """
        Args:
            nas_training: TODO
            train_datasets: TODO
            val_datasets: TODO
            test_datasets: TODO
            train_batch_size: TODO
            val_batch_size: TODO
            train_image_size: TODO
            val_image_size: TODO
            num_workers: TODO
            verify: TODO
            classes_path: TODO
        """
        super().__init__()

        self.train_datasets = train_datasets
        self.valid_datasets = val_datasets
        self.test_datasets = test_datasets

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.train_image_size = train_image_size
        self.val_image_size = val_image_size
        self.num_workers = num_workers
        self.verify = verify
        
        with open(target_class_mapping, "r") as stream:
            try:
                classes = yaml.safe_load(stream)
            except yaml.YAMLError as ex:
                logger.error("Error while parsing YAML file. Could not load class ids and labels!")
                sys.exit(1)

        self.num_classes = len(classes)
        self.target_class_mapping = bidict(classes)

        """
        Executive decision: Define transforms here.
        Previous versions used a separate script to save transformations into a yaml format in config/ which was then
        loaded here. However, this approach was clunky and resulted in image size conflicts: the image size defined
        in the generage_augmentations.py was/could be different from the one defined in config.yaml and it's easy to
        get them confused. It's best to just have the transformations be initialized here.
        """
        # Define default image transformations for training and validation. Individual datasets will override transformations for their custom purposes
        self.train_transforms = A.Compose([
            # Geometric transformations
            A.RandomResizedCrop(self.train_image_size[0], self.train_image_size[1], scale=(0.40, 1.0), always_apply=True), # Crop the image first. Reduces computation time when applying optical or elastic distortion
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.0001),
            A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.01, rotate_limit=1, p=0.5),
            # A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),

            # Color transformations
            A.OneOf([
                A.Sequential([
                    A.ChannelShuffle(p=0.10),
                    A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, p=0.2),
                    A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.25),
                    A.ISONoise(color_shift=(0.01, 0.3), intensity=(0.1, 0.7), p=0.1),
                ]),
                A.Sequential([
                    A.RandomGamma(gamma_limit=(70,100), p=0.5),
                ])
            ]),

            A.ToGray(p=0.05),


            # Brightness transformations
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.5), p=0.5),
            A.CLAHE(p=0.15),
            # A.Blur(blur_limit=(1,5), p=0.5),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', min_area=512, min_visibility=0.4, label_fields=['class_labels']))

        self.valid_transforms = A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.RandomResizedCrop(self.val_image_size[0], self.val_image_size[1], scale=(0.75, 1.0), always_apply=True), # Crop the image first. Reduces computation time when applying optical or elastic distortion
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', min_area=512, min_visibility=0.4, label_fields=['class_labels']))

        self.test_transforms = A.Compose([
            # Geometric transformations
            A.RandomResizedCrop(self.val_image_size[0], self.val_image_size[1], scale=(0.75, 1.0), always_apply=True), # Crop the image first. Reduces computation time when applying optical or elastic distortion
            # A.CLAHE(always_apply=True),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', min_area=512, min_visibility=0.4, label_fields=['class_labels']))

    
        # Initialize the datasets to None
        self.train = None
        self.valid = None
        self.test = None

        return


    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally. 
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    @logger.catch
    def _load_datasets(self, datasets: List[Dict], current_transforms: A.Compose) -> List[Dataset]:
        """
        Args:
            datasets:
            current_transforms:
        """
        loaded_datasets = []
        for d in datasets:
            for dataset_name, dirs in d.items():
                # pdb.set_trace()
                try:
                    class_ = globals()[dataset_name]
                except KeyError:
                    logger.error(f"Dataset class name <{dataset_name}> not found! Double check configuration file or imports")
                    sys.exit(1)
                logger.info(f"Using {class_}")

                # NOTE: The data directory arguments from config.yaml are merged with the data augmentations, target class_ids/class_labels
                dirs.update({
                    "transforms": current_transforms,
                    "target_class_mapping": self.target_class_mapping,
                })
                
                dataset_kwargs = dirs

                # Instantiate dataset and check that it has a source int-to-string class mapping
                instantiated_dataset = class_(**dataset_kwargs)
                if not hasattr(instantiated_dataset, "SOURCE_CLASS_BIMAP"):
                    logger.error("Dataset MUST have SOURCE_CLASS_BIMAP attribute mapping integers to class names (str)")
                    sys.exit(1)

                loaded_datasets.append(instantiated_dataset)
        return loaded_datasets


    @logger.catch
    def setup(self, stage: str = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and
        prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Lists that store the Dataset classes for training, validation, and testing. They will concatenated into a single training, validation, or testing dataset
        training_datasets = []
        valid_datasets = []
        test_datasets = []
        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here
        if stage == "fit" or stage is None:
            
            # Stores PyTorch datasets in a list
            logger.info("Loading training/validation datasets")
            training_datasets = self._load_datasets(self.train_datasets, self.train_transforms)
            valid_datasets = self._load_datasets(self.valid_datasets, self.valid_transforms)
            
            # Concatenates all of the datasets into a single one for training and validation
            self.train = ConcatDataset(training_datasets)
            self.valid = ConcatDataset(valid_datasets)

            verification_sets = training_datasets

            # Check network input size
            self._check_network_input(training_datasets[0][0][0])   # List[Dataset[Tuple[img, target, img_info, id]]]
            self._check_network_input(valid_datasets[0][0][0])      # List[Dataset[Tuple[img, target, img_info, id]]]


        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            logger.info("Loading testing datasets")
            test_datasets = self._load_datasets(self.test_datasets, self.test_transforms)
            self.test = ConcatDataset(test_datasets)                # List[Dataset[Tuple[img, target, img_info, id]]]

            self._check_network_input(test_datasets[0][0][0])
            verification_sets = test_datasets

        
        # Verify data augmentations
        if self.verify:
            logger.info("Verifying data augmentations. See GUI")
            for v in verification_sets:
                # Use verification set for visualization because the ConcatDataset only contains samples and labels
                # It doesn't contain any additional fields from the original dataset classes (like the source and target mappings)
                visualize_bbox_augmentations(v, idx=0, samples=3)
            # visualize_bbox_augmentations(valid_datasets[0], idx=10, samples=3)

        logger.success("Data Loaded")

        
    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet. 
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=1, shuffle=False, num_workers=self.num_workers)
    
    @logger.catch
    def _check_network_input(self, network_input: torch.Tensor):
        c, h, w = network_input.shape
        if h % 32 != 0 or w % 32 != 0:
            logger.error("Network input height/width must be multiples of 32. Check config file for proper target image size!!")
            sys.exit(1)



if __name__ == "__main__":
    # Use for debugging data loading
    logger.info("Debugging data loading")

    target_class_mapping = bidict({
        0: 'building',
        1: 'vehicle',
        2: 'person',
        3: 'window',
        4: 'door'
    })
    dm = DroneNetDataModule(
        nas_training=False,
        train_datasets=[
            {'SoccerPlayerDataset': 
                {'root': "/mnt/data/Datasets/Person_Detection/Soccer_Player_Dataset", 
                 'split': "train",
                 'intermediate_class_mapping': {'player': 'person'},
                 'max_labels': 30}
            },
        ],
        val_datasets=[
            {'SoccerPlayerDataset': 
                {'root': "/mnt/data/Datasets/Person_Detection/Soccer_Player_Dataset", 
                 'split': "val",
                 'intermediate_class_mapping': {'player': 'person'},
                 'max_labels': 30}
            },
            {'COCODataset':
                {'root': '/mnt/data/Datasets/COCO',
                 'split': 'val2017',
                 'categories_to_include': ['person',
                                           'bicycle',
                                           'car',
                                           'motorcycle',
                                           'airplane',
                                           'bus',
                                           'train',
                                           'truck',
                                           'bus'],
                 'intermediate_class_mapping': {'person': 'person',
                                                'bicycle': 'vehicle',
                                                'car': 'vehicle',
                                                'motorcycle': 'vehicle',
                                                'airplane': 'vehicle',
                                                'bus': 'vehicle',
                                                'train': 'vehicle',
                                                'truck': 'vehicle',
                                                'boat': 'vehicle'},
                }
            },
        ],
        test_datasets=[],
        train_batch_size=8,
        val_batch_size=8,
        train_image_size=[640, 640],
        val_image_size=[608, 608],
        num_workers=8,
        verify=True,
        target_class_mapping='/home/bbikdash/Development/1_object_detection/kef_yolox/config/namc_classes.yaml'
    )

    dm.setup('fit')

