
import sys
import os
import shutil
from loguru import logger
import pdb
import numpy as np
import json


def _get_image_paths(directory):
    # Initialize lists to store file paths and file names
    file_paths = []
    file_names = []
    for root, _, files in os.walk(directory):
        for filename in files:
            # Check if the file has a .png or .jpg extension
            if filename.lower().endswith(('.png', '.jpg')):
                # Get the absolute file path
                file_path = os.path.join(root, filename)

                # Get the file name with extension (without parent directories)
                file_name = os.path.relpath(file_path, directory)

                # Append the file path and file name to the respective lists
                file_paths.append(file_path)
                file_names.append(file_name)
    return file_paths, file_names

json_path = "/mnt/data/Datasets/Building_Detection/eo/ETG/etg.json"

try:
    with open(json_path, 'r') as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    logger.error(f"Error parsing JSON: {e}")
    raise

output_dir = "/mnt/data/Datasets/Building_Detection/eo/ETG/processed/"

annotations = {}
for image_filepath, image_data in data.items():
    # Extract the relevant fields
    objects = image_data.get("objects", [])
    # You can also extract other fields as needed

    # Make output directory
    os.makedirs(os.path.join(output_dir, os.path.dirname(image_filepath)), exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(json_path), image_filepath), os.path.join(output_dir, os.path.dirname(image_filepath))) # Copy relevant file to processed dir

    # File name of the image is the key    
    image_filename = os.path.basename(image_filepath)

    # Iterate through every bbox dictionary in the list
    for bbox_obj in objects:

        # Retrieve unnormalized yolo style bbox coords
        bbox = np.array([bbox_obj.get('xcent'),
                         bbox_obj.get('ycent'),
                         bbox_obj.get('width'),
                         bbox_obj.get('height')])
        
        # Check if key has been added to the annotations dict
        current_bboxes = annotations.get(image_filename)
        if current_bboxes is None:
            # Image sample has not been encountered before. Add it to the dictionary
            annotations[image_filename] = bbox
        else:
            # Initial bbox was already added. Add new one to the list
            annotations[image_filename] = np.vstack((current_bboxes, bbox))

print(annotations)



file_paths, file_names = _get_image_paths(os.path.dirname(json_path))
# print(file_paths, file_names) 