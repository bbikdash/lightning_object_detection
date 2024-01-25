# Copyright (c) 2022 KEF Robotics in Pittsburgh, PA

"""
@author Bassam Bikdash
November 2022

This file trains a powerline detection dilated Convolutional Neural Network described in: https://www.ri.cmu.edu/app/uploads/2017/08/root.pdf
Loads a training, validation, and test dataset. Applies the appropriate data augmentation to each. Logs IoU and F1 score during
training and validation. The two weights with the lowest validation loss are saved. Finally, evaluates the best set of weights on
the test set.

Can be run directly from the command line or from the VSCode UI.
"""

import os
import sys
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from powerline_model_1 import PowerlineSeg
from KEF_POWERLINE_DATASETS import Mithril_Carla_Dataset, AIRLab_Powerline_Dataset, Junk_Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import utils.visualization as u

def main():
    
    # Define image transformations for training and validation
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, p=0.5),
        A.AdvancedBlur(blur_limit=(1,7), noise_limit=(0.9, 1.1), p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.ISONoise(p=0.5),
        A.InvertImg(p=0.5),
        A.Resize(args.input_size[0], args.input_size[1], always_apply=True),
        A.SmallestMaxSize(max_size=args.input_size[0], always_apply=True),
        A.OneOf([
            A.CropNonEmptyMaskIfExists(args.input_size[0], args.input_size[1], p=0.65),
            A.RandomResizedCrop(args.input_size[0], args.input_size[1], scale=(0.5, 1.0), p=0.35),
        ], p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    valid_transforms = A.Compose([
        A.SmallestMaxSize(max_size=args.input_size[0], always_apply=True),
        # A.CropNonEmptyMaskIfExists(args.input_size[0], args.input_size[1], always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Prepare datasets
    train_real_dataset = AIRLab_Powerline_Dataset("/home/bbikdash/Datasets/AIRLab_Wire_Detection_Dataset/wires_usf/train", train_transforms, args.input_size)
    valid_real_dataset = AIRLab_Powerline_Dataset("/home/bbikdash/Datasets/AIRLab_Wire_Detection_Dataset/wires_usf/val", valid_transforms, args.input_size)
    test_real_dataset = AIRLab_Powerline_Dataset("/home/bbikdash/Datasets/AIRLab_Wire_Detection_Dataset/wires_usf/test", valid_transforms, args.input_size)

    train_dataset = ConcatDataset([train_real_dataset])
    valid_dataset = ConcatDataset([valid_real_dataset])
    test_dataset = ConcatDataset([test_real_dataset])

    # Verify data augmentations
    if args.verify:
        u.visualize_powerline_augmentations(train_real_dataset, idx=0, samples=5)

    # Create dataloaders
    num_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_cpu-4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_cpu-4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_cpu-4)

    # Initialize model
    model = PowerlineSeg(args)

    # Modify checkpoint behavior
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="val_loss", mode='min', every_n_epochs=1)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Create training object
    trainer = Trainer(
        logger = TensorBoardLogger,
        callbacks=[checkpoint_callback, lr_monitor, TQDMProgressBar(refresh_rate=20),
            SavingCallback(train_transforms=train_transforms, valid_transforms=valid_transforms, test_transforms=valid_transforms)],
        accelerator="gpu",
        devices=-1,
        max_epochs=5,
    )

    # Train and validate
    if False:
        # Activate Pytorch-Lightnings learning rate finder to suggest an initial learning rate
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model, train_dataloader)
        print(f"\nAuto-find suggested learning rate: {lr_finder.suggestion()}\n")

        # For plotting, running with VSCode debugger
        fig = lr_finder.plot(suggest=True)
        fig.show()
    else:
        trainer.fit(model, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    # Process cmd line args
    main()
