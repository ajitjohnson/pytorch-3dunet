#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:22:53 2024

@author: aj
"""

import tifffile
import h5py
import os
import matplotlib.pyplot as plt



def convert_ome_tiff_to_h5(ome_tiff_path, h5_directory, h5_name, channels=None):
    """
    Convert an OME-TIFF file to an HDF5 file and save only specified channels.
    
    Parameters:
    ome_tiff_path (str): Path to the input OME-TIFF file.
    h5_directory (str): Directory to save the output HDF5 file.
    h5_name (str): Name of the output HDF5 file.
    channels (list, optional): List of channel indices to save. If None, all channels are saved.
    """
    # Ensure the output directory exists
    os.makedirs(h5_directory, exist_ok=True)

    # Read the OME-TIFF file
    ome_tiff_data = tifffile.imread(ome_tiff_path)

    # Define the output HDF5 file path
    h5_file_path = os.path.join(h5_directory, h5_name + '.h5')

    # If channels is None, save all channels
    if channels is None:
        selected_data = ome_tiff_data
    else:
        # Validate the channels
        if ome_tiff_data.ndim == 3:
            max_channels = ome_tiff_data.shape[0]
            for channel in channels:
                if channel < 0 or channel >= max_channels:
                    raise ValueError(f"Invalid channel index {channel}. It should be between 0 and {max_channels - 1}.")
            selected_data = ome_tiff_data[channels, :, :]
        elif ome_tiff_data.ndim == 2:
            if len(channels) > 1 or channels[0] != 0:
                raise ValueError("Invalid channel selection for a single channel image.")
            selected_data = ome_tiff_data
        else:
            raise ValueError(f"Unexpected data shape: {ome_tiff_data.shape}")

    # Write the selected data to an HDF5 file
    with h5py.File(h5_file_path, 'w') as h5_file:
        h5_file.create_dataset('raw', data=selected_data)

    print(f"Selected channels successfully saved to {h5_file_path}")

# Example usage
convert_ome_tiff_to_h5('path/to/your/file.ome.tif', 'path/to/output/directory', 'output_file_name', channels=[0, 1])




# Example usage
ome_tiff_path = '/Users/aj/Partners HealthCare Dropbox/Ajit Nirmal/nirmal lab/resources/exemplarData/mcmicroExemplar001/registration/exemplar-001.ome.tif'
h5_directory = '/Users/aj/Desktop/cspotExampleData/h5_trial/predict'
h5_name = 'predict'
channels = None

convert_ome_tiff_to_h5(ome_tiff_path, h5_directory, h5_name, channels)

# running this on O2
ome_tiff_path = '/n/scratch/users/a/ajn16/cspot_new/testing/registration/image.tif'
h5_directory = '/n/scratch/users/a/ajn16/cspot_new/testing'
h5_name = 'predict'
channels = None


def read_h5_and_plot_channel(h5_file_path, dataset='raw', channel_index=0, sample_index=0):
    """
    Read an HDF5 file and plot a specified channel from a specified sample.

    Parameters:
    h5_file_path (str): Path to the HDF5 file.
    dataset (str): Name of the dataset to read from the HDF5 file. Default is 'raw'.
    channel_index (int): Index of the channel to plot. Default is 0.
    sample_index (int): Index of the sample to plot. Default is 0.
    """
    # Read the HDF5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Load the specified dataset
        data = h5_file[dataset][:]
        
        # Check if the dataset has more than one sample
        if data.ndim == 4:  # Assuming data has shape (samples, channels, height, width)
            if sample_index >= data.shape[0]:
                raise ValueError(f"Invalid sample_index {sample_index}. It should be less than {data.shape[0]}.")
            if channel_index >= data.shape[1]:
                raise ValueError(f"Invalid channel_index {channel_index}. It should be less than {data.shape[1]}.")
            
            channel_data = data[sample_index, channel_index, :, :]
        elif data.ndim == 3:  # Assuming data has shape (channels, height, width)
            if channel_index >= data.shape[0]:
                raise ValueError(f"Invalid channel_index {channel_index}. It should be less than {data.shape[0]}.")
            
            channel_data = data[channel_index, :, :]
        elif data.ndim == 2:  # Assuming data has shape (height, width)
            if channel_index != 0:
                raise ValueError(f"Invalid channel_index {channel_index}. Data has only one channel.")
            
            channel_data = data
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

    # Plot the specified channel
    plt.imshow(channel_data, cmap='gray')
    plt.title(f'Sample {sample_index}, Channel {channel_index}')
    plt.colorbar()
    plt.show()


# Example usage
h5_file_path = '/Users/aj/Desktop/cspotExampleData/h5_trial/predict/predict_single_channel.h5'
read_h5_and_plot_channel(h5_file_path, channel_index = 0, dataset='raw')
h5_file_path =  '/Users/aj/Desktop/cspotExampleData/h5_trial/final_predictions/predict_multi_channel_predictions.h5'
read_h5_and_plot_channel(h5_file_path, channel_index = 2, dataset='predictions')




def load_and_print_h5_datasets(h5_file_path):
    """
    Load an HDF5 file and print the names and contents of the datasets within it.
    
    Parameters:
    h5_file_path (str): Path to the HDF5 file.
    """
    with h5py.File(h5_file_path, 'r') as h5_file:
        # List all groups
        print("Keys in the HDF5 file:")
        for key in h5_file.keys():
            print(key)
        
        # Print the data in each dataset
        for key in h5_file.keys():
            print(f"\nData in dataset '{key}':")
            data = h5_file[key][:]
            print(data)

# Example usage
h5_file_path = '/Users/aj/Partners HealthCare Dropbox/Ajit Nirmal/nirmal lab/softwares/dev/pytorch-3dunet/prediction/final_prediction/predict_predictions.h5'
load_and_print_h5_datasets(h5_file_path)




def print_h5_dataset_shapes(h5_file_path):
    """
    Open an HDF5 file and print the shapes of all datasets within it.
    
    Parameters:
    h5_file_path (str): Path to the HDF5 file.
    """
    with h5py.File(h5_file_path, 'r') as h5_file:
        for key in h5_file.keys():
            dataset = h5_file[key]
            print(f"Dataset '{key}': shape = {dataset.shape}")

# Example usage in the context of generateTrainTestSplit
h5_file_path = '/Users/aj/Desktop/cspotExampleData/h5_trial/final_predictions/predict_multi_channel_predictions.h5'
print_h5_dataset_shapes(h5_file_path)




























