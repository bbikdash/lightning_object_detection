# Train KEF YOLOX on Custom Datasets

## 0. Prerequisites
Clone repo, install dependencies, download pretrained weights.

## 1. Prepare your custom dataset with images and labels


## 2. Create and save class IDS dictionary
Define the dictionary of class IDs and class labels in a `yaml` file and place it within `config/`. See [`config/coco_classes.yaml`](../config/coco_classes.yaml) for reference.

## 3. Write a `Dataset` class to load images and labels


1. Create a PyTorch `Dataset` class and place it in `data/`. The following convention is recommended: using underscores for the name of the containing `.py` file and camel case naming for the class names within that file. For example, `InriaBuildingDetection` is the `Dataset` class defined in `building_detection.py`.


2. The `Dataset` class should have the following functions:
* `load_image()`: TODO
* `load_annotation()`: TODO
* `pull_item()`: TODO
* `_apply_augmentations()`: TODO
* `__getitem__()`: calls `pullitem()`  TODO

### 4. IMPORTANT: `__getitem__()` function return convention

This is the function header and `return` with type hinting:
```python
import numpy as np
import torch
from typing import Tuple
def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int], np.ndarray]:
    ...
    return img, target, img_info, img_id
```

* `img` is an `1 x 3 x H x W` torch float tensor array representing the image. It is __unnormalized__.
* `target` is an `N x 5` numpy array where each row represents a single bounding box and class: `[class_id, x_center, y_center, width, height]`. The bounding boxes are in __unnormalized YOLO format__.
* `img_info` is an `1 x 2` numpy array: `[original height, original width]` of the image (before any preprocessing)
* `img_id` is an `1 x 1` numpy array containing the image ID of the current sample. This is a remnant of the COCO dataset where every image had a unique ID. It is unused for custom datasets (i.e. set to 0)

This convention is consistent among all of the original `Dataset` classes written by Bassam (see [UAV Vehicle Dataset](../data/uav_vehicle.py)) and is inherited from the original YOLOX COCO dataset class.




## 4. Edit `__init__.py` to expose your Dataset to the LightningDataModule
Make sure the `Dataset` class is exposed within `data/` by editing [`data/__init__.py`](../data/__init__.py).

## 5. Change data augmentations
Change the training, validation, and testing data augmentations within the constructor of [`detection_data_module.py`](../data/detection_data_module.py). These augmentations will be saved at training time inside the appropriate log directory.

## 6. Configure hyperparameters
Configure all necessary hyperparameters for training from `config.yaml`.

## 7. Run training

```shell
python3 main.py fit --config config.yaml
```
or
```shell
./run_train.sh
```

## 