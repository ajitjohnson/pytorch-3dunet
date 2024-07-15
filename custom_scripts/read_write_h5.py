#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:21:23 2024
@author: aj
"""

import h5py
import numpy as np


import os
import re
import numpy as np
import tifffile

def read_image_label_pairs(folder_path, extension=".tif"):
    """
    Reads image and label pairs from a given folder with the specified extension.
    Assumes file naming convention of {name}_img.tif for images and {name}_mask.tif for labels.
    
    Parameters:
    - folder_path: Path to the folder containing the TIFF files.
    - extension: The file extension of the image files (default is ".tif").
    
    Returns:
    - raw_data: Numpy array of all the images.
    - label_data: Numpy array of all the labels.
    """
    
    # Initialize lists to hold file paths
    image_files = []
    label_files = []
    
    # Compile regex patterns for matching file names
    img_pattern = re.compile(r"(.+)_img" + re.escape(extension) + "$")
    mask_pattern = re.compile(r"(.+)_mask" + re.escape(extension) + "$")
    
    # Scan directory for files matching the given extension
    for file in os.listdir(folder_path):
        if img_pattern.match(file):
            image_files.append(file)
        elif mask_pattern.match(file):
            label_files.append(file)
    
    # Sort files to ensure matching pairs align
    image_files.sort()
    label_files.sort()
    
    # Initialize lists to hold image and label arrays
    images = []
    labels = []
    
    # Read each image-label pair into numpy arrays
    for img_file, label_file in zip(image_files, label_files):
        img_path = os.path.join(folder_path, img_file)
        label_path = os.path.join(folder_path, label_file)
        
        img = tifffile.imread(img_path)
        label = tifffile.imread(label_path)
        
        images.append(img)
        labels.append(label)
    
    # Convert lists of arrays into single 3D numpy arrays (images, height, width)
    raw_data = np.stack(images)
    label_data = np.stack(labels)
    
    return raw_data, label_data

# Example usage
folder_path = '/Users/aj/Dropbox (Partners HealthCare)/nirmal lab/resources/exemplarData/cspotExampleData/CSPOT/TrainingData/CD3D/training'
folder_path = '/Users/aj/Dropbox (Partners HealthCare)/nirmal lab/resources/exemplarData/cspotExampleData/CSPOT/TrainingData/CD3D/test'
folder_path = '/Users/aj/Dropbox (Partners HealthCare)/nirmal lab/resources/exemplarData/cspotExampleData/CSPOT/TrainingData/CD3D/validation'

raw_data, label_data = read_image_label_pairs(folder_path)



# Define your file path
file_path = '/Users/aj/Dropbox (Partners HealthCare)/nirmal lab/resources/exemplarData/cspotExampleData/CSPOT/TrainingData/CD3D/validation.h5'

with h5py.File(file_path, 'w') as file:
    # Create datasets for your images and labels
    file.create_dataset('raw', data=raw_data)
    file.create_dataset('label', data=label_data)




# load and check image
import matplotlib.pyplot as plt


path = '/Users/aj/Dropbox (Partners HealthCare)/nirmal lab/resources/exemplarData/cspotExampleData/CSPOT/TrainingData/CD3D/train.h5'
path ='/Users/aj/Downloads/Movie1_t00003_crop_gt.h5'
with h5py.File(path, 'r') as f:
    raw = f['raw'][...]
    label = f['label'][...]
    

raw.shape
label.shape


mid_raw = np.expand_dims(raw[raw.shape[0] // 2], 0)
mid_label = np.expand_dims(label[raw.shape[0] // 2], 0)


plt.imshow(mid_raw[0])
plt.imshow(mid_label[0])


# train3dunet --config <CONFIG>


###############################################################################
# Prediction functions


def read_image_predict (folder_path, extension=".tif"):    
    # Initialize lists to hold file paths
    image_files = []
    
    # Compile regex patterns for matching file names
    img_pattern = re.compile(r'.*' + re.escape(extension) + r'$')
    
    # Scan directory for files matching the given extension
    for file in os.listdir(folder_path):
        if img_pattern.match(file):
            image_files.append(file)
    
    # Sort files to ensure matching pairs align
    image_files.sort()
    
    # Initialize lists to hold image and label arrays
    images = []
    
    # Read each image-label pair into numpy arrays
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        
        img = tifffile.imread(img_path)

        images.append(img)
    
    # Convert lists of arrays into single 3D numpy arrays (images, height, width)
    raw_data = np.stack(images)
    
    return raw_data



folder_path = '/Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/dev/pytorch-3dunet/prediction/images'
raw_data = read_image_predict(folder_path)


# write 
file_path = '/Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/dev/pytorch-3dunet/prediction/image_to_predict/predict.h5'

with h5py.File(file_path, 'w') as file:
    # Create datasets for your images and labels
    file.create_dataset('raw', data=raw_data)


# predict3dunet --config test_config.yml	


# load and check
path = '/Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/dev/pytorch-3dunet/prediction/final_prediction/predict_predictions.h5'

with h5py.File(path, 'r') as f:
    predictions = f['predictions'][...]
    

predictions.shape

array_channel_1 = predictions[:, 0, :, :].squeeze()
array_channel_2 = predictions[:, 1, :, :].squeeze()


plt.imshow(array_channel_1)
plt.imshow(array_channel_2)
