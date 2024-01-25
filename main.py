# Copyright (c) 2022 KEF Robotics in Pittsburgh, PA

"""
@author Bassam Bikdash
April 2023

This file can train various Yolo based object detection models related to the EOTACs project at KEF. The original training code was
written by one of KEF's first full-time employees, Avery Horvath.
Loads a training, validation, and test dataset. Applies the appropriate data augmentation to each. Logs metrics during
training and validation. By default, the three weights with the lowest validation loss are saved.

This ML pipeline was recently refactored to take advantage of the PyTorch Lightning Command Line Interface (CLI) and it is recommended
to run training, validation, and testing using that command line interface. Specify datasets and hyperparameters for training in `config.yaml`.

To run training and validation, run
`./run_train.sh` or `python3 main.py fit --config config.yaml`

To test, run
`python3 main.py test --config config.yaml`
"""

import sys
import os
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import Callback, BasePredictionWriter
from typing import Any, Dict, List, Literal, Optional, Sequence
import albumentations as A
from loguru import logger
import pdb

# May use one day: Python relative imports: https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time

class AlbumentTransformSavingCallback(Callback):
    """Saves the data augmentations used for this round of training, validation, or testing"""

    def on_sanity_check_end(self, trainer, pl_module):
        """Saves the data augmentations used for this round of training, validation, or testing
        Args:
            trainer: the current Trainer instance
            pl_module: the current LightningModule instance
        """
        A.save(trainer.datamodule.train_transforms, os.path.join(trainer.logger.save_dir, trainer.logger.name, f"version_{trainer.logger.version}/train_transforms.yaml"), data_format='yaml')
        A.save(trainer.datamodule.valid_transforms, os.path.join(trainer.logger.save_dir, trainer.logger.name, f"version_{trainer.logger.version}/valid_transforms.yaml"), data_format='yaml')
        A.save(trainer.datamodule.test_transforms,  os.path.join(trainer.logger.save_dir, trainer.logger.name, f"version_{trainer.logger.version}/test_transforms.yaml"),  data_format='yaml')
        logger.info("Transformations saved")

class VerifyNumClasses(Callback):
    """Checks that the number of classes in the lightning module and the datamodule are the same"""
    def setup(self, trainer, pl_module, stage):
        """
        Args:
            trainer: the current Trainer instance
            pl_module: the current LightningModule instance
            stage: either 'fit', 'validate', 'test', or 'predict'
        """
        if pl_module.num_classes != trainer.datamodule.num_classes:
            logger.error(f"\nLightning Module Num Classes={pl_module.num_classes}\nLightning Data Module Num Classes={trainer.datamodule.num_classes}")
            logger.error("The number of classes specified for YOLOX output layer is different from that specified for data loading. Check config file!!")
            sys.exit(1)
        else:
            logger.success("Number of classes verified")

class SaveBestWeightDict(Callback):

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint) -> None:
        super().on_load_checkpoint(trainer, pl_module, checkpoint)
        logger.warning("Overwriting learning rate found in checkpoint")
        # logger.warning(f"Overwriting checkpoint optimizer to {optimizer} with {learning_rate_scheduler} learning rate scheduler") # TODO: Put this here eventually
        # pdb.set_trace() # Maybe: https://github.com/Lightning-AI/lightning/issues/1982
        # checkpoint['optimizer_states'][0]['param_groups'][0]['lr'] = 1e-03
        # checkpoint['lr_schedulers'][0]['step_size'] = 3
        # checkpoint['lr_schedulers'][0]['gamma'] = 0.9
        return

    """Saves the best_k_models dict containing the checkpoint paths with the corresponding scores to a YAML"""
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        NOTE: This callback is not run if the on_save_checkpoint hook is used. Leave as on_validation_epoch_end
        Args:
            trainer: the current Trainer instance
            pl_module: the current LightningModule instance
        """
        # Get the object for the ModelCheckpoint callback
        checkpoint_class = trainer.callbacks[-1]

        if os.path.exists(checkpoint_class.dirpath):
            # Save the metrics for the best weights only if checkpoint directory exists yet
            checkpoint_class.to_yaml()
            print("\n")
            logger.success(f"Saving metrics for best checkpoints")
        else:
            # The checkpoint directory doesn't exist yet
            print("\n")
            logger.error(f"Checkpoint path doesn't exist yet. Can't save top_k checkpoint metrics")
        return


class BBoxPredictionWriter(BasePredictionWriter):
    # TODO: Finish writing this
    # https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.BasePredictionWriter.html#lightning.pytorch.callbacks.BasePredictionWriter
    def __init__(self, write_interval: Literal['batch', 'epoch', 'batch_and_epoch'] = "batch") -> None:
        super().__init__(write_interval)

    def write_on_batch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           prediction: Any,
                           batch_indices: Sequence[int],
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int) -> None:
        return super().write_on_batch_end(trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)


def main():
    cli = LightningCLI()
    # Might be worth changing to in future: https://lightning.ai/docs/pytorch/latest/cli/lightning_cli_intermediate_2.html
    # Docs: https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html#instantiation-only-mode
    
if __name__ == '__main__':
    # Process cmd line args
    main()

