# Copyright (c) November 2022 KEF Robotics in Pittsburgh, PA

"""
@author Mithril Hugunin, Bassam Bikdash
November 2022

Trains a powerline detection dilated Convolutional Neural Network described in: https://www.ri.cmu.edu/app/uploads/2017/08/root.pdf
The original powerline network with Mithril's loss function; produced decent results
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import albumentations as A
import cv2





# CNN for powerline segmentation
class LitYOLOX(LightningModule):    # Name, Object of Interest, Special Format, Purpose
    # Power line model originally coded in TensorFlow by Mithril
    def __init__(self):
        super().__init__()
        # Set our init args as class attributes

        # Do not change
        self.layers_to_freeze = 3

        # Load and freeze first 2 conv layers of VGG16 (conv2d, relu, conv2d, relu)
        # Verified that weights are frozen in the debugger: for param in self.feature_extractor[0].parameters(): print(param.requires_grad)
        backbone = vgg16(weights=VGG16_Weights.DEFAULT)
        # feature_extractor = list(vgg16(weights=VGG16_Weights.DEFAULT).features)[:self.layers_to_freeze]
        self.feature_extractor = backbone.features[:self.layers_to_freeze]

        for i in range(self.layers_to_freeze):
            for param in self.feature_extractor[i].parameters():
                param.requires_grad = False
        # print(self.feature_extractor)

        # Define model architecture (verified that weights are unfrozen)
        self.conv1 = nn.Conv2d(64, 32, (3, 3), padding='valid')
        self.conv2 = nn.Conv2d(32 ,32, (3, 3), padding='valid', dilation=(2,2))
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding='valid', dilation=(4,4))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), padding='valid', dilation=(8,8))
        self.conv5 = nn.Conv2d(32, 3, (32, 32), padding='valid', stride=(16,16))

        # Save all hyperparameters to the checkpoint and a yaml file within the logging dir
        self.save_hyperparameters()

        self.train_iou = BinaryJaccardIndex(num_classes=2, threshold=0.5)
        self.train_f1 = BinaryF1Score(threshold=0.5, multidim_average="global")
        self.val_iou = BinaryJaccardIndex(num_classes=2, threshold=0.5)
        self.val_f1 = BinaryF1Score(threshold=0.5, multidim_average="global")
        self.test_iou = BinaryJaccardIndex(num_classes=2, threshold=0.5)
        self.test_f1 = BinaryF1Score(threshold=0.5, multidim_average="global")

    def forward(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        # print(x.shape)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        # print(x.shape)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        # print(x.shape)
        x = F.leaky_relu(x)
        x = self.conv5(x)
        # print(x.shape)
        # Convert logits to probabilities in [0,1] but only for 1st channel
        # x[:,0,:,:] = torch.sigmoid(x[:,0,:,:])
        return x


    def criterion(self, y_hat: torch.Tensor, y: torch.Tensor):
        """
        Compute the loss between the prediction and the ground truth label.
        Expects two tensors of Bx3xHxW dim for loss.
        This loss function was originally written by Mithril in tensorflow.
        """
        return


    # NOTE: We are using the PyTorch Lightning CLI. This is overriden by the optimizer and scheduler specific in the config file.
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #     # scheduler = ReduceLROnPlateau(optimizer, factor=self.gamma, patience=5, threshold)
    #     scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
    #     return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        # Get the image and the label
        x, y, y_hesse = train_batch

        # Perform forward pass through the network 
        y_hat = self(x)

        # Compute loss
        loss = self.criterion(y_hat, y_hesse)

        self.train_iou(y_hat[:,0], y_hesse[:,0].to(torch.int))
        self.train_f1(y_hat[:,0], y_hesse[:,0].to(torch.int))

        # If you want to calculate epoch-level metrics and log them, use log().
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        self.log("train_iou", self.train_iou, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=True)
        return loss


    def validation_step(self, val_batch, batch_idx):
        
        # Get the image and the label
        x, y, y_hesse = val_batch

        # Perform forward pass through the network 
        y_hat = self(x)

        # Compute loss
        loss = self.criterion(y_hat, y_hesse)

        self.val_iou(y_hat[:,0], y_hesse[:,0].to(torch.int))
        self.val_f1(y_hat[:,0], y_hesse[:,0].to(torch.int))

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", self.val_iou, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        return loss


    def test_step(self, test_batch, batch_idx):
        # Get the image and the label
        x, y, y_hesse = test_batch

        # Perform forward pass through the network 
        y_hat = self(x)

        # Compute loss
        loss = self.criterion(y_hat, y_hesse)

        self.test_iou.update(y_hat[:,0], y_hesse[:,0].to(torch.int))
        self.test_f1.update(y_hat[:,0], y_hesse[:,0].to(torch.int))

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_iou", self.test_iou, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)
        return loss


    # Alternative way of organizing the train and validation steps esp. if they share common code
    # https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb
    def predict(self, image: np.ndarray, threshold: float=0.6, transforms: A.Compose=None) -> np.ndarray:
        
        
        return img # HxWxC normalized b/t [0,1] with rectangles drawn

    