#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Thu Aug 18 16:37:29 2022
#@author: Ajit Johnson Nirmal
#Function to generate Masks for UNET model


"""
!!! abstract "Short Description"
    The function generates a mask for the deep learning model training, using 
    automated approaches. Splitting the data into training, validation and 
    test sets is also included in the function, making it easier to feed the 
    data directly into the deep learning algorithm. Note that manually drawing 
    the mask on thumbnails is the ideal approach, however for scalability 
    purposes, automation is used.


## Function
"""

# Libs
import pathlib
import h5py
import random
import numpy as np
import tifffile
import argparse
import os
import matplotlib.pyplot as plt
from skimage import filters


# Function
def generateTrainTestSplit (thumbnailFolder, 
                            num_classes=3,
                            mode='deploy',
                            testSamples=10,
                            outputDir=None,
                            seed=0,
                            updateH5=None,
                            file_extension=None,
                            verbose=True,
                            TruePos='TruePos', NegToPos='NegToPos',
                            TrueNeg='TrueNeg', PosToNeg='PosToNeg'):

    
    """
Parameters:
    thumbnailFolder (list):  
        List of folders that contain the human-sorted thumbnails to be used for generating
        training data and splitting them into train, test, and validation cohorts.

    num_classes (int, optional):  
        Number of classes for the OTSU thresholding task.

    mode (str, optional):  
        Mode of operation. Two options are available: 'test' and 'deploy'. It is recommended to first run in 'test' mode to evaluate the precision of OTSU thresholding. If needed, adjust the `num_classes` before running in 'deploy' mode. 

    testSamples (int, optional):  
        This is the number of random image plots that will be generated to assess the accuracy of OTSU thresholding. The results will be saved in a new folder under `outputDir`.
        
    outputDir (str, optional):   
        Path to the directory where the `.h5` file would be saved. If None, current working directory is used.

    seed (int, optional):  
        Random seed for reproducibility. The default is 0.

    updateH5 (str, optional):  
        Option to update an existing H5 file. Pass in the name of the `h5` file. It needs to be in the specified `outputDir`.

    file_extension (str, optional):  
        File extension to filter files in thumbnailFolder. The default is None.

    verbose (bool, optional):  
        If True, print detailed information about the process to the console. The default is True.

    TruePos (str, optional):  
        Name of the folder that holds the thumbnails classified as True Positive. The default is 'TruePos'.

    NegToPos (str, optional):  
        Name of the folder that holds the thumbnails classified as Negative to Positive. The default is 'NegToPos'.

    TrueNeg (str, optional):  
        Name of the folder that holds the thumbnails classified as True Negative. The default is 'TrueNeg'.

    PosToNeg (str, optional):  
        Name of the folder that holds the thumbnails classified as Positive to Negative. The default is 'PosToNeg'.

Returns:
    H5 file (object):  
        LNew or updated H5 file.

Example:

    ```python
    # Folder where the raw thumbnails are stored
    thumbnailFolder = ['/Thumbnails/CD3D',
                       '/Thumbnails/ECAD']
    
    # The function accepts the four pre-defined folders. If you had renamed them, please change it using the parameters below.
    # If you had deleted any of the folders and are not using them, replace the folder name with `None` in the parameters.
    cs.generateTrainTestSplit(thumbnailFolder, 
                              outputDir='/Users/aj/Documents/cspotExampleData',
                              file_extension=None,
                              TruePos='TruePos', NegToPos='NegToPos',
                              TrueNeg='TrueNeg', PosToNeg='PosToNeg')
    
    # Same function if the user wants to run it via Command Line Interface
    python generateTrainTestSplit.py \
        --thumbnailFolder /Users/aj/Desktop/cspotExampleData/CSPOT/Thumbnails/CD3D /Users/aj/Desktop/cspotExampleData/CSPOT/Thumbnails/ECAD \
        --outputDir /Users/aj/Desktop/cspotExampleData/
    ```
    """

    # Function takes in path to two folders, processes the images in those folders,
    # and saves them into a different folder that contains Train, Validation and Test samples
    #TruePos='TruePos'; NegToPos='NegToPos'; TrueNeg='TrueNeg'; PosToNeg='PosToNeg'; verbose=True
    #thumbnailFolder = ['/Users/aj/Partners HealthCare Dropbox/Ajit Nirmal/nirmal lab/resources/exemplarData/cspotExampleData/CSPOT/Thumbnails/CD3D',
    #                   '/Users/aj/Partners HealthCare Dropbox/Ajit Nirmal/nirmal lab/resources/exemplarData/cspotExampleData/CSPOT/Thumbnails/ECAD']


    # convert the folder into a list
    if isinstance (thumbnailFolder, str):
        thumbnailFolder = [thumbnailFolder]
    # convert all path names to pathlib
    thumbnailFolder = [pathlib.Path(p) for p in thumbnailFolder]
    
    # resolve outputDir
    if outputDir is None:
        outputDir = pathlib.Path(os.getcwd())
    else:
        outputDir = pathlib.Path(outputDir)
        outputDir.mkdir(parents=True, exist_ok=True)
    
    # resolve the final H5 file to be saved
    if updateH5 is None:
        file_path = outputDir / 'thumbnails.h5'
    else:
        file_path = outputDir / str(updateH5)


    # standard format
    if file_extension is None:
        file_extension = '*'
    else:
        file_extension = '*' + str(file_extension)
    
    
    # create a master list of all thumbnails that need to be processed now. 
    def process_folders(thumbnailFolder, TruePos, NegToPos, TrueNeg, PosToNeg, file_extension):
        master_positive_cells, master_negative_cells = [], []
        master_positive_cells_names, master_negative_cells_names = [], []
    
        for folder in thumbnailFolder:
            pos, negtopos, neg, postoneg = [], [], [], []
    
            if TruePos is not None:
                pos = list(pathlib.Path(folder / TruePos).glob(file_extension))
            if NegToPos is not None:
                negtopos = list(pathlib.Path(folder / NegToPos).glob(file_extension))
            if TrueNeg is not None:
                neg = list(pathlib.Path(folder / TrueNeg).glob(file_extension))
            if PosToNeg is not None:
                postoneg = list(pathlib.Path(folder / PosToNeg).glob(file_extension))
    
            master_positive_cells.extend(pos + negtopos)
            master_negative_cells.extend(neg + postoneg)
    
            master_positive_cells_names.extend(["pos_" + file.name for file in pos + negtopos])
            master_negative_cells_names.extend(["neg_" + file.name for file in neg + postoneg])
    
        return master_positive_cells, master_negative_cells, master_positive_cells_names, master_negative_cells_names

    
    # run the function
    master_positive_cells, master_negative_cells, master_positive_cells_names, master_negative_cells_names = process_folders(thumbnailFolder=thumbnailFolder, 
                                                  TruePos=TruePos, 
                                                  NegToPos=NegToPos, 
                                                  TrueNeg=TrueNeg, 
                                                  PosToNeg=PosToNeg, 
                                                  file_extension=file_extension)
    
    if mode == 'test':
        # subsample the data
        np.random.seed(seed)
        # Create a random permutation of indices
        num_total_images = len(master_positive_cells)
        sampled_indices = random.sample(range(num_total_images), testSamples)
        # Sample the data
        master_positive_cells = [master_positive_cells[i] for i in sampled_indices]
        master_positive_cells_names = [master_positive_cells_names[i] for i in sampled_indices]
        master_negative_cells = []; master_negative_cells_names=[] # just empy out negs as we do not want to test those



    # First process the positive_cells
    # Load all images into a 3D array
    if len(master_positive_cells) > 0:
        posimages = np.array([tifffile.imread(img_path) for img_path in master_positive_cells])
        num_images, height, width = posimages.shape
        
        # Apply Gaussian blur to all images
        blurred_images = np.array([filters.gaussian(img, sigma=1, truncate=1) for img in posimages])
        
        # Apply multi-level Otsu thresholding to all images
        thresholds = np.array([filters.threshold_multiotsu(img, classes=num_classes) for img in blurred_images])
        
        # Create binary label images for all images
        pos_label_images = np.zeros_like(blurred_images)
        for i in range(num_images):
            regions = np.digitize(blurred_images[i], bins=thresholds[i])
            pos_label_images[i][regions == num_classes - 1] = 1
    else:
        posimages = []; blurred_images = []; pos_label_images= []
    
    # Now process the negative_cells
    if len (master_negative_cells) > 0:
        negimages = np.array([tifffile.imread(img_path) for img_path in master_negative_cells])
        num_images, height, width = negimages.shape
        neg_label_images = np.zeros_like(negimages)
    else:
        negimages = []; neg_label_images=[]
    
    
    # Combine arrays
    def concatenate_images(arr1, arr2):
        # Filter out empty arrays
        arrays_to_concatenate = [arr for arr in [arr1, arr2] if arr.size > 0]
        # Concatenate the non-empty arrays or return an empty array if all are empty
        return np.concatenate(arrays_to_concatenate, axis=0) if arrays_to_concatenate else np.array([])
    
    # Combine positive and negative images
    all_images = concatenate_images(np.array(posimages), np.array(negimages))
    # Combine positive and negative label images
    all_label_images = concatenate_images(np.array(pos_label_images), np.array(neg_label_images))
    # Combine positive and negative filenames
    all_filenames = master_positive_cells_names + master_negative_cells_names

    
# =============================================================================
#     # shuffle the data
#     # Set a seed for reproducibility
#     np.random.seed(seed)
#     # Create a random permutation of indices
#     num_total_images = all_images.shape[0]
#     indices = np.random.permutation(num_total_images)
#     # Apply the permutation to shuffle the data
#     shuffled_images = all_images[indices]
#     shuffled_label_images = all_label_images[indices]
#     shuffled_filenames = [all_filenames[i] for i in indices]
# =============================================================================
    
    
   # Save the processed data as a H5 file
    if mode == 'deploy':
        def save_to_hdf5(file_path, images, label_images, filenames):
            if os.path.exists(file_path):
                # If the file exists, append the new data to the existing datasets
                with h5py.File(file_path, 'a') as file:
                    if 'raw' in file and 'label' in file and 'name' in file:
                        # Get the current size of the datasets
                        raw_size = file['raw'].shape[0]
                        label_size = file['label'].shape[0]
                        name_size = file['name'].shape[0]
        
                        # Resize the datasets to accommodate the new data
                        file['raw'].resize((raw_size + images.shape[0]), axis=0)
                        file['label'].resize((label_size + label_images.shape[0]), axis=0)
                        file['name'].resize((name_size + len(filenames)), axis=0)
        
                        # Append the new data to the datasets
                        file['raw'][-images.shape[0]:] = images
                        file['label'][-label_images.shape[0]:] = label_images
                        file['name'][-len(filenames):] = filenames
                    else:
                        # Create new datasets if they don't already exist
                        file.create_dataset('raw', data=images, maxshape=(None, *images.shape[1:]))
                        file.create_dataset('label', data=label_images, maxshape=(None, *label_images.shape[1:]))
                        file.create_dataset('name', data=filenames, maxshape=(None,))
            else:
                # If the file does not exist, create a new file and datasets
                with h5py.File(file_path, 'w') as file:
                    file.create_dataset('raw', data=images, maxshape=(None, *images.shape[1:]))
                    file.create_dataset('label', data=label_images, maxshape=(None, *label_images.shape[1:]))
                    file.create_dataset('name', data=filenames, maxshape=(None,))
    
        # usage
        save_to_hdf5(file_path, all_images, all_label_images, all_filenames)
        #save_to_hdf5(file_path, shuffled_images, shuffled_label_images, shuffled_filenames)
        print(f"Data saved to {file_path}")
    
    
    def sample_and_plot(posimages, blurred_images, pos_label_images, master_positive_cells_names, nSamples, outputDir):
        # Ensure the output directory exists
        examplemasks_dir = os.path.join(outputDir, 'examplemasks')
        os.makedirs(examplemasks_dir, exist_ok=True)
    
        # Randomly sample indices
        indices = random.sample(range(len(posimages)), nSamples)
    
        # Sample the data
        sampled_posimages = [posimages[i] for i in indices]
        sampled_blurred_images = [blurred_images[i] for i in indices]
        sampled_pos_label_images = [pos_label_images[i] for i in indices]
        sampled_names = [master_positive_cells_names[i] for i in indices]
    
        # Plot and save the sampled images
        for i in range(nSamples):
            a = sampled_posimages[i]
            b = sampled_blurred_images[i]
            c = sampled_pos_label_images[i]
            name = sampled_names[i]
    
            fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    
            # Original image
            ax[0].imshow(a, cmap='gray')
            ax[0].set_title(name)
            ax[0].axis('off')
    
            # Blurred image
            ax[1].imshow(b, cmap='gray')
            ax[1].set_title('Blurred Image')
            ax[1].axis('off')
    
            # Segmented regions
            ax[2].imshow(c, cmap='gray')
            ax[2].set_title('Segmented Regions')
            ax[2].axis('off')
    
            plt.tight_layout()
    
            # Save the figure
            plot_path = os.path.join(examplemasks_dir, f'sample_{i}.png')
            plt.savefig(plot_path)
            plt.close()
    
    # Example usage:
    if mode == 'test':
        sample_and_plot(posimages=posimages, 
                        blurred_images=blurred_images,
                        pos_label_images=pos_label_images, 
                        master_positive_cells_names=master_positive_cells_names, 
                        nSamples=testSamples, 
                        outputDir=outputDir)
    if mode == 'deploy':
        sample_and_plot(posimages=posimages, 
                        blurred_images=blurred_images, 
                        pos_label_images=pos_label_images, 
                        master_positive_cells_names=master_positive_cells_names, 
                        nSamples=testSamples, 
                        outputDir=outputDir)



# Make the Function CLI compatable
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate train, test, and validation cohorts from human sorted thumbnails.')
    parser.add_argument('--thumbnailFolder', type=str, nargs='+', help='List of folders that contain the human-sorted thumbnails to be used for generating training data')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes for the OTSU thresholding task.')
    parser.add_argument('--mode', type=str, default='deploy', help='Mode of operation: "test" or "deploy".')
    parser.add_argument('--testSamples', type=int, default=10, help='Number of random image plots to be generated to assess the accuracy of OTSU thresholding.')
    parser.add_argument('--outputDir', type=str, default=None, help='Path to the directory where the `.h5` file would be saved.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--updateH5', type=str, default=None, help='Option to update an existing H5 file.')
    parser.add_argument('--file_extension', type=str, default=None, help='File extension to filter files in thumbnailFolder.')
    parser.add_argument('--verbose', type=bool, default=True, help='If True, print detailed information about the process to the console.')
    parser.add_argument('--TruePos', type=str, default='TruePos', help='Name of the folder that holds the thumbnails classified as True Positive.')
    parser.add_argument('--NegToPos', type=str, default='NegToPos', help='Name of the folder that holds the thumbnails classified as Negative to Positive.')
    parser.add_argument('--TrueNeg', type=str, default='TrueNeg', help='Name of the folder that holds the thumbnails classified as True Negative.')
    parser.add_argument('--PosToNeg', type=str, default='PosToNeg', help='Name of the folder that holds the thumbnails classified as Positive to Negative.')
    args = parser.parse_args()

    generateTrainTestSplit(thumbnailFolder=args.thumbnailFolder,
                           num_classes=args.num_classes,
                           mode=args.mode,
                           testSamples=args.testSamples,
                           outputDir=args.outputDir,
                           seed=args.seed,
                           updateH5=args.updateH5,
                           file_extension=args.file_extension,
                           verbose=args.verbose,
                           TruePos=args.TruePos,
                           NegToPos=args.NegToPos, 
                           TrueNeg=args.TrueNeg,
                           PosToNeg=args.PosToNeg)
