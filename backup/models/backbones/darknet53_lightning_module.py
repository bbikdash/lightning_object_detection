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


class Darknet53(pl.LightningModule):
    # Original Implementation by: https://github.com/developer0hye/PyTorch-Darknet53/blob/master/model.py
    # Modifications made by Bassam Bikdash
    def __init__(self, num_classes, num_channels=3):
        """
        Construct the DarkNet53 backbone architecture.
        As the author states, 
        Attributes:
            num_classes -- the number of classes that DarkNet19 outputs
            num_channels -- the number of channels the input image contains (default=3, i.e. an RGB image)
        """
        super(Darknet53, self).__init__()

        block = DarkResidualBlock(num_channels)

        self.num_classes = num_classes

        self.conv1 = YoloConvBlock(in_channels=num_channels, out_channels=32)
        self.conv2 = YoloConvBlock(in_channels=32, out_channels=64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = YoloConvBlock(in_channels=64, out_channels=128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = YoloConvBlock(in_channels=128, out_channels=256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = YoloConvBlock(in_channels=256, out_channels=512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = YoloConvBlock(in_channels=512, out_channels=1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)


        # Create torchmetrics logging objects
        self.train_acc          = Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.train_ap           = AveragePrecision(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.train_recall       = Recall(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.train_f1           = F1Score(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.train_specificity  = Specificity(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")

        self.val_acc            = Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.val_ap             = AveragePrecision(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.val_recall         = Recall(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.val_f1             = F1Score(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.val_specificity    = Specificity(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")

        self.test_acc           = Accuracy(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.test_ap            = AveragePrecision(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.test_recall        = Recall(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.test_f1            = F1Score(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")
        self.test_specificity   = Specificity(task="multiclass", threshold=0.5, num_classes=num_classes, multidim_average="global")

        # Use CrossEntopyLoss for classification objective function
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)
    

    def training_step(self, train_batch, batch_idx):
        # Get the image and the label
        input, target = train_batch
        # print(f"Input Shape: {input.shape}, Target Shape: {target.shape}")

        # Perform forward pass through the network 
        output = self(input)
        # print(f"Output Shape: {output.shape}")

        # Compute loss
        loss = self.criterion(output, target)

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
        self.log("train_loss",          loss)
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
        self.log("val_loss",        loss)
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
        self.log("test_loss",           loss)
        self.log("test_acc",            self.test_acc,          prog_bar=True)
        self.log("test_ap",             self.test_ap,           prog_bar=True)
        self.log("test_recall",         self.test_recall,       prog_bar=True)
        self.log("test_f1",             self.test_f1,           prog_bar=True)
        self.log("test_specificity",    self.test_specificity,  prog_bar=True)
        return loss

# Residual block
class DarkResidualBlock(pl.LightningModule):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = YoloConvBlock(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = YoloConvBlock(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

# Standard Yolo block with Convolution, Batch Normalization, and Leaky ReLU activation.
# These operations in sequence are used many times throughout the YOLOs, so I thought it best to group them together here.
# Partially inspired by Avery's implementation.
def YoloConvBlock(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))
    