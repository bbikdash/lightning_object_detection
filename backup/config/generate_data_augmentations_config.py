"""
Saves training, validation, and testing data augmentations as yaml files to the current working directory.
These will be loaded using the PyTorch Lightning CLI when training the detection network. 
This script just provides a quick way to programmatically adjust a data augmentation pipeline
and save it as a YAML file as opposed to editing to YAML file directly (which can still be done of course).

"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

input_size = [224, 224]

train_transforms = A.Compose([
	# Geometric transformations
    A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.RandomRotate90(p=0.5),
    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45),

    # Color transformations
    # A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=0.3, p=0.5),
    # A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, p=0.5),
    # A.AdvancedBlur(blur_limit=(1,7), noise_limit=(0.9, 1.1), p=0.5),
    # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
    # A.ISONoise(p=0.5),
    A.RandomResizedCrop(input_size[0], input_size[1], scale=(0.75, 1), always_apply=True),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

valid_transforms = A.Compose([
    A.Resize(input_size[0] + 32, input_size[1] + 32, always_apply=True),
    A.CenterCrop(input_size[0], input_size[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

test_transforms = A.Compose([
    A.Resize(input_size[0] + 32, input_size[1] + 32, always_apply=True),
    A.CenterCrop(input_size[0], input_size[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


A.save(train_transforms, "./train_transforms.yaml", data_format='yaml')
A.save(valid_transforms, "./valid_transforms.yaml", data_format='yaml')
A.save(test_transforms,  "./test_transforms.yaml",  data_format='yaml')
print("Transformations saved")

