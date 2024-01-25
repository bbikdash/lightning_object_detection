"""
@author Bassam Bikdash, 


"""
from typing import Any, Dict, List, Optional, Tuple, Type


import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import utils.visualization as u
from typing import Dict, List
from .. import backbones.yang_darknet19
from .. import backbones.darknet_lightning_module.YoloConvBlock

class YoloV2(pl.LightningModule):
    """
    PyTorch Lightning of YoloV2. Highly recommend readings this: https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf
    DarkNet-19 is the original backbone of YoloV2.
    According to the paper, batch normalization is applied during every convolution layer and eliminates the need for other forms of regularization
    like dropout. Using BN led to a 2% improvement in mAP. This is implemented in a custom nn.Module down below, ConvBN.

    Avery likely referred to this very helpful article for YoloV2: https://towardsdatascience.com/training-object-detection-yolov2-from-scratch-using-cyclic-learning-rates-b3364f7e4755    
    NOTE: YOLOV2 Architecture: See http://ethereon.github.io/netscope/#/gist/d08a41711e48cf111e330827b1279c31.


    """
    
    def __init__(self,
                 num_classes: int,
                 num_channels: int = 3, # TODO: Redundant????
                 num_anchors: int = 5,
                 confidence_threshold: float = 0.2,
                 nms_threshold: float = 0.45,
                 pretrained: bool = True,
                 freeze_backbone: bool = True):
        """
        Construct the YoloV2 architecture. Initializes DarkNet19 backbone architecture and loads into and the rest
        of the layers necessary to train YoloV2.

        Attributes:
            num_classes -- the number of classes that DarkNet19 outputs
            num_anchors -- the number of anchor boxes to output for each class (original paper uses 5)
            num_channels -- Deprecated input argument
                Images should always be a 3 channel image. The number of channels the input image contains (default=3, i.e. an RGB image)
                The DarkNet19 feature extractor would need to be retrained with a new architecture to change the number of input channels.
        """
        super(YoloV2, self).__init__()

        # TODO: Load backbone here
        if pretrained:
            # Load the feature extractor layers a version of DarkNet19 published
            # by Jianhua Yang (https://github.com/yjh0410/yolov2-yolov3_PyTorch) recently trained using PyTorch
            # Does not contain the classification head, only the feature extraction layers
            self.feature_extractor = yang_darknet19.build_darknet19()

        # TODO: Freeze backbone here
        # In the OG paper, DarkNet19 was pre-trained on ImageNet 1000 and it's weights were frozen.
        if freeze_backbone:
            print("Freezing backbone ...")
            for i in range(self.feature_extractor):
                for param in self.feature_extractor[i].parameters():
                    param.requires_grad = False
        
        # TODO: Define the head and neck layers here
        # Then the additional layers of the  network were trained for detection.
        # These layers/blocks are specific to YoloV2. They are applied after all of the DarkNet19 feature extraction layers/blocks
        self.yolo_layer_1 = nn.Sequential(
            YoloConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            YoloConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        )

        # Last 2 main layers of YoloV2. Conv21 -> BatchNorm21 -> LeakyReLU21 -> Conv22
        self.yolo_layer_2 = nn.Sequential(
            YoloConvBlock(in_channels=3072, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=(5 + num_classes) * num_anchors, kernel_size=(1, 1), stride=(1, 1), padding=0) # Conv 22
        )


        # TODO: Create torchmetrics logging objects here

        # TODO: Create any additional nn.Module loss functions here
        # confidence_loss_function = MSEWithLogitsLoss(reduction='mean')
        # class_loss_function = nn.CrossEntropyLoss(reduction='none')
        # txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
        # twth_loss_function = nn.MSELoss(reduction='none')
        # iou_loss_function = nn.SmoothL1Loss(reduction='none')


    def reorg_layer(self, x):
        stride = 2
        batch_size, channels, height, width = x.size()
        new_ht = height/stride
        new_wd = width/stride
        new_channels = channels * stride * stride
        
        passthrough = x.permute(0, 2, 3, 1)
        passthrough = passthrough.contiguous().view(-1, new_ht, stride, new_wd, stride, channels)
        passthrough = passthrough.permute(0, 1, 3, 2, 4, 5)
        passthrough = passthrough.contiguous().view(-1, new_ht, new_wd, new_channels)
        passthrough = passthrough.permute(0, 3, 1, 2)
        return passthrough
    

    def forward(self, x):
        out = self.backbone_section_1(x)        # Conv 1-13
        passthrough = self.reorg_layer(out)

        out = self.backbone_section_2(out)      # MaxPool5 and Conv 14-21
        
        out = self.yolo_layer_1(out)
        out = torch.cat([passthrough, out], 1)
        out = self.yolo_layer_2(out)      # Conv 22

        return out  # BxH'xW'x((5 + NUM_CLASSES) * NUM_ANCHORS)
    
    def criterion(self, output, target):
        # TODO: needs to be implemented
        """
        The YOLO loss function has 5 terms.

            1. 
            2. Best IOU with Ground Truth --> Train everything (x,y,w,h,obj,class)
            3. Not best IOU but IOU > 0.5 --> Train nothing
            4. IOU < 0.5 --> Train objectness
            5. 
        YOLOv2 --> object confidence and class predictions use squared errors
        """
        raise NotImplementedError

    def configure_optimizers(self) -> Tuple[List, List]:
        """Constructs the optimizer and learning rate scheduler."""
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)
        lr_scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_params)
        return [optimizer], [lr_scheduler]
    

    def training_step(self, train_batch, batch_idx):
        """
         The input from the data loader is expected to be a list of images. Each image is a tensor with
        shape ``[channels, height, width]``. The images from a single batch will be stacked into a
        single tensor, so the sizes have to match. Different batches can have different image sizes, as
        long as the size is divisible by the ratio in which the network downsamples the input.
        During training, the model expects both the input tensors and a list of targets. *Each target is
        a dictionary containing*:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in `(x1, y1, x2, y2)` format
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        """
        # Get the image and the label
        input, target = train_batch

        # Perform forward pass through the network 
        output = self(input)
        detections = self._split_detections(output)

        # Compute loss
        loss = self.criterion(output, target)


        # TODO: Write the metric computations here

        # self.train_iou.update(y_hat[:,0], y_hesse[:,0].to(torch.int))
        # self.train_f1.update(y_hat[:,0], y_hesse[:,0].to(torch.int))

        # TODO: Log the metrics here
        # # If you want to calculate epoch-level metrics and log them, use log().
        # # Logging to TensorBoard by default

        # Call metric objects with update to update the metrics (do not change, ex. taken from the docs)
        self.train_acc.update(output, target)
        self.train_ap.update(output, target)

        # Lightning will log the metric based on on_step and on_epoch flags present in self.log(...).
        # If on_epoch is True, the logger automatically logs the end of epoch metric value by calling .compute()
        # NOTE: Not recommended to change this logging code
        # During the training epoch, the average metric for each batch is logged. An epoch-level average is not computed 
        self.log("train_loss",          loss,                              prog_bar=False)
        self.log("train_acc",           self.train_acc.compute(),          prog_bar=True)
        return loss



    def validation_step(self, val_batch, batch_idx):
        # Get the image and the label
        input, target = val_batch

        # Perform forward pass through the network 
        output = self(input)
        detections = self._split_detections(output)

        # Compute loss
        loss = self.criterion(y_hat, y_hesse)

        # TODO: Write the metric computations here

        # TODO: Log the metrics here
        # Calling self.log will surface up scalars for you in TensorBoard
        # self.log("val_loss", loss, prog_bar=True)
        # self.log("val_iou", self.val_iou, on_epoch=True, prog_bar=True)
        # self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        return loss


    def test_step(self, test_batch, batch_idx):
        # Get the image and the label
        x, y = test_batch

        # Perform forward pass through the network 
        y_hat = self(x)

        # Compute loss
        loss = self.criterion(y_hat, y_hesse)

        self.test_iou.update(y_hat[:,0], y_hesse[:,0].to(torch.int))
        self.test_f1.update(y_hat[:,0], y_hesse[:,0].to(torch.int))

        # TODO: Write the metric computations here

        # TODO: Log the metrics here
        # Calling self.log will surface up scalars for you in TensorBoard
        # self.log("test_loss", loss, prog_bar=True)
        # self.log("test_iou", self.test_iou, prog_bar=True)
        # self.log("test_f1", self.test_f1, prog_bar=True)
        return loss


    def infer(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Feeds an image to the network and returns the detected bounding boxes, confidence scores, and class
        labels.
        Args:
            image: An input image, a tensor of uint8 values sized ``[channels, height, width]``.
        Returns:
            boxes (:class:`~torch.Tensor`), confidences (:class:`~torch.Tensor`), labels (:class:`~torch.Tensor`):
            A matrix of detected bounding box `(x1, y1, x2, y2)` coordinates, a vector of
            confidences for the bounding box detections, and a vector of predicted class labels.
        """
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)

        self.eval()
        detections = self(image.unsqueeze(0))
        detections = self._split_detections(detections)
        detections = self._filter_detections(detections)
        boxes = detections["boxes"][0]
        scores = detections["scores"][0]
        labels = detections["labels"][0]
        return boxes, scores, labels

    def _split_detections(self, detections:torch.Tensor) -> Dict[str, torch.Tensor]:
        """Splits the detection tensor returned by a forward pass into a dictionary.
        The fields of the dictionary are as follows:
            - boxes (``Tensor[batch_size, N, 4]``): detected bounding box `(x1, y1, x2, y2)` coordinates
            - scores (``Tensor[batch_size, N]``): detection confidences
            - classprobs (``Tensor[batch_size, N]``): probabilities of the best classes
            - labels (``Int64Tensor[batch_size, N]``): the predicted labels for each image
        Args:
            detections: A tensor of detected bounding boxes and their attributes.
        Returns:
            A dictionary of detection results.
        """
        boxes = detections[..., :4]
        scores = detections[..., 4]
        classprobs = detections[..., 5:]
        classprobs, labels = torch.max(classprobs, -1)
        return {"boxes": boxes, "scores": scores, "classprobs": classprobs, "labels": labels}


    def _filter_detections(self, detections: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """Filters detections based on confidence threshold. Then for every class performs non-maximum suppression
        (NMS). NMS iterates the bounding boxes that predict this class in descending order of confidence score, and
        removes lower scoring boxes that have an IoU greater than the NMS threshold with a higher scoring box.
        Finally the detections are sorted by descending confidence and possible truncated to the maximum number of
        predictions.
        Args:
            detections: All detections. A dictionary of tensors, each containing the predictions
                from all images.
        Returns:
            Filtered detections. A dictionary of lists, each containing a tensor per image.
        """
        boxes = detections["boxes"]
        scores = detections["scores"]
        classprobs = detections["classprobs"]
        labels = detections["labels"]

        out_boxes = []
        out_scores = []
        out_classprobs = []
        out_labels = []

        for img_boxes, img_scores, img_classprobs, img_labels in zip(boxes, scores, classprobs, labels):
            # Select detections with high confidence score.
            selected = img_scores > self.confidence_threshold
            img_boxes = img_boxes[selected]
            img_scores = img_scores[selected]
            img_classprobs = img_classprobs[selected]
            img_labels = img_labels[selected]

            img_out_boxes = boxes.new_zeros((0, 4))
            img_out_scores = scores.new_zeros(0)
            img_out_classprobs = classprobs.new_zeros(0)
            img_out_labels = labels.new_zeros(0)

            # Iterate through the unique object classes detected in the image and perform non-maximum
            # suppression for the objects of the class in question.
            for cls_label in labels.unique():
                selected = img_labels == cls_label
                cls_boxes = img_boxes[selected]
                cls_scores = img_scores[selected]
                cls_classprobs = img_classprobs[selected]
                cls_labels = img_labels[selected]

                # NMS will crash if there are too many boxes.
                cls_boxes = cls_boxes[:100000]
                cls_scores = cls_scores[:100000]
                selected = nms(cls_boxes, cls_scores, self.nms_threshold)

                img_out_boxes = torch.cat((img_out_boxes, cls_boxes[selected]))
                img_out_scores = torch.cat((img_out_scores, cls_scores[selected]))
                img_out_classprobs = torch.cat((img_out_classprobs, cls_classprobs[selected]))
                img_out_labels = torch.cat((img_out_labels, cls_labels[selected]))

            # Sort by descending confidence and limit the maximum number of predictions.
            indices = torch.argsort(img_out_scores, descending=True)
            if self.max_predictions_per_image >= 0:
                indices = indices[: self.max_predictions_per_image]
            out_boxes.append(img_out_boxes[indices])
            out_scores.append(img_out_scores[indices])
            out_classprobs.append(img_out_classprobs[indices])
            out_labels.append(img_out_labels[indices])

        return {"boxes": out_boxes, "scores": out_scores, "classprobs": out_classprobs, "labels": out_labels}