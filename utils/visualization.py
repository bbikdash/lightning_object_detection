# Copyright (c) 2022 KEF Robotics in Pittsburgh, PA

"""
@author Bassam Bikdash, Mithril Hugunin, Albumentations Team

Contains various functions for visualizing data augmentations, bounding boxes, and saving videos.
"""

import os
import math
import copy
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio
from loguru import logger
from typing import Tuple

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names: Tuple[str] = None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


# Useful utility from: https://github.com/formigone/tf-imagenet/blob/master/data_exploration.ipynb
def parse_synset_mapping(path):
    """Parse the synset mapping file into a dictionary mapping <synset_id>:[<synonyms in English>]
    This assumes an input file formatted as:
        <synset_id> <category>, <synonym...>
    Example:
        n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
    """
    synset_map = {}
    with open(path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            parts = line.split(' ')
            synset_map[parts[0]] = [label.strip() for label in ' '.join(parts[1:]).split(',')]
        return synset_map

# Useful utility from: https://github.com/formigone/tf-imagenet/blob/master/data_exploration.ipynb
def generate_synset_to_int_mapping(synset_mapping):
    synset_to_int_map = {}
    for index, (key, val) in enumerate(synset_mapping.items()):
        synset_to_int_map[key] = index
    return synset_to_int_map
    
# Useful utility from: https://github.com/formigone/tf-imagenet/blob/master/data_exploration.ipynb
def generate_int_to_synset_mapping(synset_mapping):
    int_to_synset_map = {}
    for index, (key, val) in enumerate(synset_mapping.items()):
        int_to_synset_map[index] = key
    return int_to_synset_map

def preprocess_mask(mask):
    """
    Function written by Albumentations Team: https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/pytorch_semantic_segmentation.ipynb
    """
    mask = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask


def display_image_grid(images_filenames, images_directory, masks_directory, predicted_masks=None):
    """
    Function written by Albumentations Team: https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/pytorch_semantic_segmentation.ipynb
    """
    cols = 3 if predicted_masks else 2
    rows = len(images_filenames)
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
    for i, image_filename in enumerate(images_filenames):
        image = cv2.imread(os.path.join(images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(masks_directory, image_filename.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED,)
        mask = preprocess_mask(mask)
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")

        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("Ground truth mask")

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

        if predicted_masks:
            predicted_mask = predicted_masks[i]
            ax[i, 2].imshow(predicted_mask, interpolation="nearest")
            ax[i, 2].set_title("Predicted mask")
            ax[i, 2].set_axis_off()
    plt.tight_layout()
    plt.show()
    

def visualize_augmentations(dataset, idx=0, samples=5):
    """
    Function written by Albumentations Team: https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/pytorch_semantic_segmentation.ipynb
    """
    dataset = copy.deepcopy(dataset)
    dataset.transforms = A.Compose([t for t in dataset.transforms if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, figsize=(10,10))
    for i in range(samples):
        image, label = dataset[idx]
        ax[i].imshow(image)
        ax[i].set_title(label)
        ax[i].set_axis_off()
    plt.tight_layout()
    plt.show()

def visualize_bbox_augmentations(dataset, idx=0, samples=5):
    if samples >= 10:
        logger.warning(f"{samples} figures will be displayed before training")
    # dataset = copy.deepcopy(dataset)

    for i in range(samples):
        img, target, _ = dataset[idx]
        img = img.permute(1,2,0).detach().cpu().numpy()
        height, width, channel = img.shape
        class_id, bboxes = target[:,0], target[:,1:5]

        background = img.copy()
        for j in range(len(bboxes)):
            box = bboxes[j]
            if (box[0] == 0 and box[1] == 0 and
                box[2] == 0 and box[3] == 0):
               continue
            cls_id = class_id[j]
            # Assuming yolo format for visualization

            # Assuming unnormalized yolo format for visualization
            xc, yc, w, h = box
            col_min = int(xc - w/2)
            row_min = int(yc - h/2)
            col_max = int(xc + w/2)
            row_max = int(yc + h/2)
            background = cv2.rectangle(background, (col_min, row_min), (col_max, row_max), (255,0,0), 2)
            
            text = f"{dataset.target_class_mapping[cls_id]}"
            txt_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(background, text, (col_min + 2, row_min + txt_size[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), thickness=1)

        background = (background - np.min(background)) / (np.max(background) - np.min(background))
        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), frameon=True, layout="tight", dpi=300)
        ax.imshow(background) ; ax.set_title(f"Bounding Boxes, Index: {idx}, Sample: {i}") ; ax.set_axis_off()
        plt.show()
    return


def write2mp4(path, frames, fps=10):
    writer = imageio.get_writer(path, fps=fps)

    for f in frames:
        writer.append_data(f)
    writer.close()
    