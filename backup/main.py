# Copyright (c) 2022 KEF Robotics in Pittsburgh, PA

"""
@author Bassam Bikdash
Feb. 2023

This file can train various Yolo based object detection models related to the EOTACs project at KEF. The original training code was
written by one of KEF's first full-time employees, Avery 
Loads a training, validation, and test dataset. Applies the appropriate data augmentation to each. Logs IoU and F1 score during
training and validation. By default, the three weights with the lowest validation loss are saved. Finally, evaluates the best set of weights on
the test set.

This ML pipeline was recently refactored to take advantage of the PyTorch Lightning Command Line Interface (CLI) and it is recommended
to run training, validation, and testing using that command line interface. Specify datasets and hyperparameters for training in `config.yaml`.

To run training and validation, run
`./run_train.sh` or `python3 main.py fit --config config.yaml`
 
To test, run
`python3 main.py test --config config.yaml`

"""

import os
import sys
# sys.path.insert(1, "/home/bbikdash/Development/10_bassam_devel_utils/")
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import Callback
from typing import List
import albumentations as A
import utils.visualization as u


class AlbumentTransformSavingCallback(Callback):
    """Saves the data augmentations used for this round of training, validation, or testing
    Args:
        transforms_path (list of strings): Paths to the .yaml files containing serialized albumentation
            data augmentations. Each file corresponds to a single albumentations.Compose() object. 
    """
    def __init__(self, transforms_path: List[str]) -> None:
        super().__init__()
        self.train_transforms_path = transforms_path[0]
        self.valid_transforms_path = transforms_path[1]
        self.test_transforms_path = transforms_path[2]
        
        self.train_transforms = A.load(self.train_transforms_path, data_format='yaml')
        self.valid_transforms = A.load(self.valid_transforms_path, data_format='yaml')
        self.test_transforms  = A.load(self.test_transforms_path, data_format='yaml')

    def on_sanity_check_start(self, trainer, pl_module):
        A.save(self.train_transforms, os.path.join(trainer.logger.save_dir, trainer.logger.name, f"version_{trainer.logger.version}/train_transforms.yaml"), data_format='yaml')
        A.save(self.valid_transforms, os.path.join(trainer.logger.save_dir, trainer.logger.name, f"version_{trainer.logger.version}/valid_transforms.yaml"), data_format='yaml')
        A.save(self.test_transforms,  os.path.join(trainer.logger.save_dir, trainer.logger.name, f"version_{trainer.logger.version}/test_transforms.yaml"),  data_format='yaml')
        print(f"\nTransformations saved")


def main():
    
    cli = LightningCLI()
    # Docs: https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html#instantiation-only-mode
    
if __name__ == '__main__':
    # Process cmd line args
    main()

