

import os
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
import lightning.pytorch as pl
import argparse
from typing import List, Optional, Dict
import albumentations as A
from utils.visualization import visualize_augmentations

class DroneNetDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self,
                 train_datasets: List[Dict],
                 val_datasets: List[Dict],
                 test_datasets: List[Dict],
                 transforms: List[str],
                 batch_size: int,
                 image_size: list,
                 verify: bool):
        super().__init__()
        self.train_datasets = train_datasets
        self.valid_datasets = val_datasets
        self.test_datasets = test_datasets

        self.train_transforms_path = transforms[0]
        self.valid_transforms_path = transforms[1]
        self.test_transforms_path = transforms[2]

        self.input_size = image_size
        self.batch_size = batch_size
        self.verify = verify

        # Define image transformations for training and validation
        self.train_transforms = A.load(self.train_transforms_path, data_format='yaml')
        self.valid_transforms = A.load(self.valid_transforms_path, data_format='yaml')
        self.test_transforms  = A.load(self.test_transforms_path, data_format='yaml')
        
        # Initialize the datasets to None
        self.train = None
        self.valid = None
        self.test = None
        # self.prepare_data_per_node = False # Needed to prevent "AttributeError: 'PowerLineDataModule' object has no attribute 'prepare_data_per_node'"
        
    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally. 
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass


    def _load_datasets(self, datasets: List[Dict], current_transforms):
        loaded_datasets = []
        for d in datasets:
            for dataset, dirs in d.items():
                class_ = getattr(pl_datasets, dataset)
                print("Using ", class_)
                loaded_datasets.append(class_(dirs["image_dir"], dirs["label_dir"], current_transforms))
        return loaded_datasets


    def setup(self, stage: str = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
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
            training_datasets = self._load_datasets(self.train_datasets, self.train_transforms)
            valid_datasets = self._load_datasets(self.valid_datasets, self.valid_transforms)
            
            # Concatenates all of the datasets into a single one for training and validation
            self.train = ConcatDataset(training_datasets)
            self.valid = ConcatDataset(valid_datasets)

            # Verify data augmentations
            if self.verify:
                visualize_augmentations(training_datasets[0], idx=0, samples=5)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_datasets = self._load_datasets(self.test_datasets, self.test_transforms)
            self.test = ConcatDataset(test_datasets)

        
    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet. 
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count()-4)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count()-4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, shuffle=False, num_workers=os.cpu_count()-4)
