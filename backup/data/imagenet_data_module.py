"""
@author Bassam Bikdash
From here: https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/datamodules/imagenet_datamodule.py
Modified by Bassam Bikdash to use Albumentations for data augmentations and changed the loading scheme of the
ImageNet dataset (made it simpler and better). Freakin amateurs

"""

import numpy as np
import os
from typing import Any, List, Dict

from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split

import albumentations as A
from utils.visualization import parse_synset_mapping, generate_synset_to_int_mapping, visualize_augmentations

from data.imagenet_1k import ImageNet, ImageNetValidation

class ImagenetDataModule(LightningDataModule):
    """
    PyTorch Lightning Data Module for 2017 ImageNet Dataset from Kaggle.
    Specs:
        - 1000 classes
        - Each image is (3 x varies x varies) (here we default to 3 x 224 x 224)

    This module Imagenet train, val and test dataloaders.
    The train set is the imagenet train.
    The val set is taken from the train set with `num_imgs_per_val_class` images per class.
    For example if `num_imgs_per_val_class=2` then there will be 2,000 images in the validation set.
    The test set is the official imagenet validation set.
    Example::
        from pl_bolts.datamodules import ImagenetDataModule
        dm = ImagenetDataModule(IMAGENET_PATH)
        model = LitModel()
        Trainer().fit(model, datamodule=dm)

    We assume the following directory structure from the 2017 ImageNet Dataset downloaded from Kaggle.
    
    ILSVRC/
        Annotations/
            CLS_LOC/
                    train/
                        n01440764/
                            n01440764_18.xml
                            ... *.xml
                        n01443537/
                            ... *.xml
        Data/
            CLS_LOC/
                test/

                train/
                    n01440764/
                        n01440764_18.JPEG
                        ... *.JPEG
                    n01443537/
                        ... *.JPEG
                    ...
                val/
                    ILSVRC2012_val_00000001.JPEG
                    ... *.JPEG
        ImageSets/
            CLS_LOC/
                test.txt
                train_cls.txt
                train_loc.txt
                val.txt

        LOC_sample_submission.csv
        LOC_synset_mapping.txt
        LOC_train_solution.csv
        LOC_val_solution.csv
    
    The user only needs to specify the root folder for the imagenet dataset `/ILSVRC/` and the rest of the folder locations
    are hardcoded based on that. We assume that the user has not made any changes to that directory structure. The alternative
    to hardcoding these values would be to specify the following as arguments to our datamodule constructor:
        - class mapping file
        - train data solution file
        - validation data solution file
        - path to directory of training images
        - path to directory of validation images
        - any other paths to relevant files that the user needs
    """

    name = "imagenet"

    def __init__(
        self,
        imagenet_root: str,
        image_size: List[int],
        transforms: List[str],
        num_workers: int = 8,
        batch_size: int = 8,
        verify: bool = True, 
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset file
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)
    
        self.imagenet_root = imagenet_root
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verify = verify

        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # Hard coded training, validation, and testing directory locations based on imagenet_root
        self.train_dir = os.path.join(imagenet_root, "Data/CLS-LOC/train")
        self.valid_dir = os.path.join(imagenet_root, "Data/CLS-LOC/val")
        # self.train_dir = os.path.join(imagenet_root, "train")
        # self.valid_dir = os.path.join(imagenet_root, "val")
        # self.test_dir = os.path.join(imagenet_root, "Data/CLS-LOC/test") # Redundant, not needed
        # Hard coded solution csv files based on imagenet_root
        self.train_solution_csv = os.path.join(imagenet_root, "LOC_train_solution.csv")
        self.valid_solution_csv = os.path.join(imagenet_root, "LOC_val_solution.csv")

        # Hard coded location of the class mapping
        class_mapping_file = os.path.join(imagenet_root, "LOC_synset_mapping.txt")
        # Construct the class mapping {"n01478985": "great white shark"}
        self.class_mapping = parse_synset_mapping(class_mapping_file)
        # print(self.class_mapping)
        # Construct the class mapping to integer labels {"n01234789": 536}
        self.class_to_int_mapping = generate_synset_to_int_mapping(self.class_mapping)

        # Define image transformations for training and validation
        self.train_transforms = A.load(transforms[0], data_format='yaml')
        self.valid_transforms = A.load(transforms[1], data_format='yaml')
        self.test_transforms  = A.load(transforms[2], data_format='yaml')

        # Initialize the datasets to None
        self.train = None
        self.valid = None
        self.test = None


    @property
    def num_classes(self) -> int:
        """
        Return:
            1000
        """
        return 1000


    def prepare_data(self) -> None:
        """
        This method already assumes you have imagenet2012 downloaded. It validates the data using the meta.bin.
        .. warning:: Please download imagenet on your own first.
        """
        pass
        

    def setup(self, stage: str = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Lists that store the Dataset classes for training, validation, and testing. They will concatenated into a single training, validation, or testing dataset

        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here
        if stage == "fit" or stage is None:

            # Load the datasets
            self.train = ImageNet(self.train_dir, self.train_transforms)
            self.valid = ImageNet(self.valid_dir, self.valid_transforms)
            # self.validate = ImageNetValidation(self.valid_dir, self.valid_transforms, self.valid_solution_csv, self.class_to_int_mapping)
            # Perform any kind of dataset splitting here

            # Verify data augmentations
            if self.verify:
                visualize_augmentations(self.train, idx=100, samples=5)
                visualize_augmentations(self.valid, idx=50, samples=5)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = ImageNet(self.valid_dir, self.test_transforms)
            # self.test = ImageNetValidation(self.valid_dir, self.valid_transforms, self.valid_solution_csv, self.class_mapping)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
    