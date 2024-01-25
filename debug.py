# Copyright (c) 2022 KEF Robotics in Pittsburgh, PA

"""
@author Bassam Bikdash
April 2023

This file is best used to debug the full ML training pipeline for YOLOX anchor free object detection.
Loads a training, validation, and test dataset. Applies the appropriate data augmentation to each. Logs IoU and F1 score during
training and validation. The three weights with the lowest validation loss are saved. Finally, evaluates the best set of weights on
the test set.

Can be run directly from the command line or from the VSCode UI.

Recommended to just use this script for debugging model loading, data loading, data augmentation, imports, visualization,
and checkpoint saving. To run extensive training, use the Lightning CLI and config.yaml 
"""

import os
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from lit_yolox import LitYOLOX
from data.detection_data_module import DroneNetDataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentTransformSavingCallback(Callback):
    def __init__(self, train_transforms: A.Compose, valid_transforms: A.Compose, test_transforms: A.Compose) -> None:
        super().__init__()
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.test_transforms = test_transforms

    def on_sanity_check_start(self, trainer, pl_module):
        A.save(self.train_transforms, os.path.join(trainer.logger.save_dir, trainer.logger.name, f"version_{trainer.logger.version}/train_transforms.yaml"), data_format='yaml')
        A.save(self.valid_transforms, os.path.join(trainer.logger.save_dir, trainer.logger.name, f"version_{trainer.logger.version}/valid_transforms.yaml"), data_format='yaml')
        A.save(self.test_transforms,  os.path.join(trainer.logger.save_dir, trainer.logger.name, f"version_{trainer.logger.version}/test_transforms.yaml"),  data_format='yaml')
        print(f"\nTransformations saved")



def main():
    # Hyperparameters
    save_dir = "./debug_logs"
    log_name = "yolox_m_vehicle"
    image_size = [640,640]
    num_workers = 4

    batch_size = 4
    accelerator = "gpu"
    max_epochs = 5
    verify = True
    ckpt = None
    auto_lr_find = False

    datasets = [
        dict(
            UAVVehicleDetection=dict(
                image_dir="/mnt/data/Datasets/UAV-Vehicle-Detection-Dataset/data/val",
                label_dir="/mnt/data/Datasets/UAV-Vehicle-Detection-Dataset/data/val",
            )
        )
    ]

    architecture = "yolox_m"
    classes_path = "./config/vehicle_classes.yaml"
    pretrained = True
    confidence_threshold = 0.3
    nms_threshold = 0.45

    # Define image transformations for training and validation
    transforms = A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45),

        # Color transformations
        A.CLAHE(),
        A.RandomGamma(gamma_limit=(60,80), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.5), p=0.5),
        # A.Blur(blur_limit=(1,5), p=0.5),
        A.ISONoise(color_shift=(0.01, 0.3), intensity=(0.1, 0.7), p=0.5),
        A.PixelDropout(p=0.1),
        A.OneOf([
            A.RandomRain(blur_value=1, brightness_coefficient=0.9),
            A.RandomSnow(brightness_coeff=1.0),
            A.RandomShadow(),
        ]),
        A.RandomResizedCrop(640, 640, scale=(0.25, 1), always_apply=True),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo',  min_area=256, min_visibility=0.1, label_fields=['class_labels']))


    # Create logger
    logger = TensorBoardLogger(save_dir=save_dir, name=log_name, default_hp_metric=False)
    
    # Create Lightning data module
    data_module = DroneNetDataModule(train_datasets=datasets,
                                     val_datasets=datasets,
                                     test_datasets=datasets,
                                    #  transforms=transforms, 
                                     train_batch_size=batch_size,
                                     val_batch_size=batch_size,
                                     train_image_size=image_size,
                                     val_image_size=image_size,
                                     num_workers=num_workers,
                                     verify=verify,
                                     target_class_mapping=classes_path)

    # Initialize model
    model = LitYOLOX(architecture=architecture,
                     classes_path=classes_path,
                     pretrained=pretrained,
                     confidence_threshold=confidence_threshold,
                     nms_threshold=nms_threshold)


    # Modify checkpoint behavior
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="val/total_loss", mode='min', every_n_epochs=1)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Create training object
    trainer = Trainer(accelerator=accelerator,
                      max_epochs=max_epochs,
                      callbacks=[checkpoint_callback, lr_monitor, TQDMProgressBar(refresh_rate=20),
                                 AlbumentTransformSavingCallback(transforms, transforms, transforms)],
                      logger = logger,
    )

    # Train and validate
    if auto_lr_find:
        # Activate Pytorch-Lightnings learning rate finder to suggest an initial learning rate
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model, data_module.train_dataloader())
        print(f"\nAuto-find suggested learning rate: {lr_finder.suggestion()}\n")

        # For plotting, running with VSCode debugger
        fig = lr_finder.plot(suggest=True)
        fig.show()
    else:
        trainer.fit(model, datamodule=data_module)

    # Test
    # trainer.test(dataloaders=test_dataloader, ckpt_path="best")

if __name__ == '__main__':
    main()

