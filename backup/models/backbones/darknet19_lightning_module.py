"""
@author Bassam Bikdash
March 2023

Backbone models of YoloV2 and YoloV3, DarkNet19 and DarkNet53, respectively.
DarkNet19 and Darknet53 are classifiers consisting of 19 and 53 convolutional layers, respectively.
They were pretrained on ImageNet to allow the networks to learn a robust set of convolutional filters.
As such, once pretrained, the classifiers make very useful feature extractors for Yolo when
we exclude the classification layers as Redmon did in his YoloV2 and YoloV3 papers.

See: https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf
See: https://pjreddie.com/media/files/papers/YOLOv3.pdf
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import lightning.pytorch as pl
from torchmetrics.classification import Accuracy, AveragePrecision, Recall, F1Score, Specificity, AUROC


class DarkNet19(pl.LightningModule):    
    # Original Implementation by: https://github.com/visionNoob/pytorch-darknet19/blob/master/model/darknet.py
    # Modified by Bassam Bikdash, of course.
    def __init__(self, num_classes, num_channels=3, pretrained=False, weights_path=None):
        """
        Construct the DarkNet19 backbone architecture.
        As the author states, 
        Attributes:
            num_classes -- the number of classes that DarkNet19 outputs
            num_channels -- the number of channels the input image contains (default=3, i.e. an RGB image)
        """
        

        # super(DarkNet19, self).__init__()
        super().__init__()

        # Initialize feature extractor layers
        self.feature_extractor = nn.Sequential(
            # Block 1
            YoloConvBlock(in_channels=num_channels, out_channels=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        
            # Block 2
            YoloConvBlock(in_channels=32, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            YoloConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            YoloConvBlock(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            YoloConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            YoloConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            YoloConvBlock(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
            YoloConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            YoloConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            YoloConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            YoloConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            YoloConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            YoloConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 6
            YoloConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            YoloConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            YoloConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            YoloConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            YoloConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
        )

        # Initialize classifier portion
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            GlobalAvgPool2d(),
            nn.Softmax(dim=1)
        )

        # Save all hyperparameters to the checkpoint and a yaml file within the logging dir
        self.save_hyperparameters()

        # Create torchmetrics logging objects
        self.train_acc          = Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global", top_k=1)
        self.train_ap           = AveragePrecision(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.train_recall       = Recall(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global", top_k=1)
        self.train_f1           = F1Score(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global", top_k=1)
        self.train_specificity  = Specificity(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global", top_k=1)

        self.val_acc            = Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global", top_k=1)
        self.val_ap             = AveragePrecision(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.val_recall         = Recall(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global", top_k=1)
        self.val_f1             = F1Score(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global", top_k=1)
        self.val_specificity    = Specificity(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global", top_k=1)

        self.test_acc           = Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global", top_k=1)
        self.test_ap            = AveragePrecision(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.test_recall        = Recall(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global", top_k=1)
        self.test_f1            = F1Score(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global", top_k=1)
        self.test_specificity   = Specificity(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global", top_k=1)

        # Use CrossEntopyLoss for classification objective function
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.MSELoss()

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out

    def training_step(self, train_batch, batch_idx):
        # Get the image and the label
        input, target = train_batch
        # print(f"Input Shape: {input.shape}, Target Shape: {target.shape}")
        # print(input)
        # print(target)

        # Perform forward pass through the network 
        output = self(input)
        # print(f"Output Shape: {output.shape}")
        # print(torch.argmax(output,1))

        # Compute loss
        loss = self.criterion(output, target)
        # print(self.train_acc(output, target))

        # Call metric objects with update to update the metrics (do not change, ex. taken from the docs)
        self.train_acc.update(output, target)
        self.train_ap.update(output, target)
        self.train_recall.update(output, target)
        self.train_f1.update(output, target)
        self.train_specificity.update(output, target)

        # If you want to calculate epoch-level metrics and log them, use log().
        # Logging to TensorBoard by default
        # Lightning will log the metric based on on_step and on_epoch flags present in self.log(...).
        # If on_epoch is True, the logger automatically logs the end of epoch metric value by calling .compute()
        # NOTE: Not recommended to change this logging code
        # During the training epoch, the average metric for each batch is logged. An epoch-level average is not computed 
        self.log("train_loss",          loss,                              prog_bar=False)
        self.log("train_acc",           self.train_acc.compute(),          prog_bar=True)
        self.log("train_ap",            self.train_ap.compute(),           prog_bar=False)
        self.log("train_recall",        self.train_recall.compute(),       prog_bar=False)
        self.log("train_f1",            self.train_f1.compute(),           prog_bar=True)
        self.log("train_specificity",   self.train_specificity.compute(),  prog_bar=False)
        return loss


    def validation_step(self, val_batch, batch_idx):
        # Get the image and the label
        input, target = val_batch
        # print(f"Input Shape: {input.shape}, Target Shape: {target.shape}")

        # Perform forward pass through the network 
        output = self(input)
        # print(f"Output Shape: {output.shape}")
        # print(output)
        # print(target)

        # Compute loss
        loss = self.criterion(output, target)

        # Calling self.log will surface up scalars for you in TensorBoard
        # Call metric objects with update to update the metrics (do not change, ex. taken from the docs)
        self.val_acc.update(output, target)
        self.val_ap.update(output, target)
        self.val_recall.update(output, target)
        self.val_f1.update(output, target)
        self.val_specificity.update(output, target)

        # NOTE: Not recommended to change this logging code
        # Logs epoch averages for the above metrics during validation. Note difference to logging during training which logs metrics per-batch
        self.log("val_loss",        loss,                                  prog_bar=True)
        self.log("val_acc",         self.val_acc,           on_epoch=True, prog_bar=True)
        self.log("val_ap",          self.val_ap,            on_epoch=True, prog_bar=False)
        self.log("val_recall",      self.val_recall,        on_epoch=True, prog_bar=False)
        self.log("val_f1",          self.val_f1,            on_epoch=True, prog_bar=True)
        self.log("val_specificity", self.val_specificity,   on_epoch=True, prog_bar=False)
        return loss


    def test_step(self, test_batch, batch_idx):
        # Get the image and the label
        input, target = test_batch

        # Perform forward pass through the network 
        output = self(input)
        # print(output)
        # Compute loss
        loss = self.criterion(output, target)
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.test_acc.update(output, target)
        self.test_ap.update(output, target)
        self.test_recall.update(output, target)
        self.test_f1.update(output, target)
        self.test_specificity.update(output, target)

        # If you want to calculate epoch-level metrics and log them, use log().
        # Logging to TensorBoard by default
        # Lightning will log the metric based on on_step and on_epoch flags present in self.log(...).
        # If on_epoch is True, the logger automatically logs the end of epoch metric value by calling .compute()
        self.log("test_loss",           loss,                   prog_bar=True)
        self.log("test_acc",            self.test_acc,          prog_bar=True)
        self.log("test_ap",             self.test_ap,           prog_bar=True)
        self.log("test_recall",         self.test_recall,       prog_bar=True)
        self.log("test_f1",             self.test_f1,           prog_bar=True)
        self.log("test_specificity",    self.test_specificity,  prog_bar=True)
        return loss


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


class DarkNet19_Verbose(pl.LightningModule):    
    # Implementation by: https://github.com/visionNoob/pytorch-darknet19/blob/master/model/darknet.py
    # See: https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf
    def __init__(self, num_classes, num_anchors=5, num_channels=3, pretrained=False):
        """
        Construct the DarkNet19 backbone architecture.

        Layers accessible with: list(model.children())[i])
                
        Attributes:
            num_classes -- the number of classes that DarkNet19 outputs
            num_anchors -- the number of anchor boxes to output for each class (original paper uses 5)
            num_channels -- the number of channels the input image contains (default=3, i.e. an RGB image)
        """
        super(DarkNet19_Verbose, self).__init__()

        model_paths = {
           'darknet19': 'https://s3.ap-northeast-2.amazonaws.com/deepbaksuvision/darknet19-deepBakSu-e1b3ec1e.pth'
        }

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm8 = nn.BatchNorm2d(256)
        
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm13 = nn.BatchNorm2d(512)
        
        self.conv14 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm14 = nn.BatchNorm2d(1024)
        self.conv15 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm16 = nn.BatchNorm2d(1024)
        self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm17 = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm18 = nn.BatchNorm2d(1024)

        self.conv19 = nn.Conv2d(1024, (5 + num_classes) * num_anchors, kernel_size=(1, 1), stride=(1, 1)), # we predict 5 boxes with 5 coordinates each and 20
        self.global_avg_pool = GlobalAvgPool2d()
        self.softmax = nn.Softmax(dim=1)

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_paths['darknet19'],  progress=True))
            print('Model is loaded')

    def forward(self, x):
        out = F.max_pool2d(F.leaky_relu(self.batchnorm1(self.conv1(x)), negative_slope=0.1), 2, stride=2)
        out = F.max_pool2d(F.leaky_relu(self.batchnorm2(self.conv2(out)), negative_slope=0.1), 2, stride=2)
        
        out = F.leaky_relu(self.batchnorm3(self.conv3(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm4(self.conv4(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm5(self.conv5(out)), negative_slope=0.1)
        out = F.max_pool2d(out, 2, stride=2)
        
        out = F.leaky_relu(self.batchnorm6(self.conv6(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm7(self.conv7(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm8(self.conv8(out)), negative_slope=0.1)
        out = F.max_pool2d(out, 2, stride=2)

        out = F.leaky_relu(self.batchnorm9(self.conv9(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm10(self.conv10(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm11(self.conv11(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm12(self.conv12(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm13(self.conv13(out)), negative_slope=0.1)
        out = F.max_pool2d(out, 2, stride=2)

        out = F.leaky_relu(self.batchnorm14(self.conv14(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm15(self.conv15(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm16(self.conv16(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm17(self.conv17(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm18(self.conv18(out)), negative_slope=0.1)

        out = self.conv19(out)
        out = self.global_avg_pool(out)
        out = self.softmax(out)

        return out



# Standard Yolo block with Convolution, Batch Normalization, and Leaky ReLU activation.
# These operations in sequence are used many times throughout the YOLOs, so I thought it best to group them together here.
# Partially inspired by Avery's implementation.
def YoloConvBlock(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))