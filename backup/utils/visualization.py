# Copyright (c) 2022 KEF Robotics in Pittsburgh, PA

"""
@author Bassam Bikdash, Mithril Hugunin, Albumentations Team
November 2022

Trains a powerline detection dilated Convolutional Neural Network described in: https://www.ri.cmu.edu/app/uploads/2017/08/root.pdf
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


def write2mp4(path, frames, fps=10):
    writer = imageio.get_writer(path, fps=fps)

    for f in frames:
        writer.append_data(f)
    writer.close()
    