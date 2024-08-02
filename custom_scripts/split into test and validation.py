#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:46:52 2024

@author: aj
"""

import h5py
import numpy as np
import os

def split_h5_dataset(path, train_percentage, val_percentage, output_dir, seed=42):
    np.random.seed(seed)

    # Load the H5 file
    with h5py.File(path, 'r') as f:
        raw = f['raw'][...]
        label = f['label'][...]
        names = f['name'][...].astype('S')  # Convert to fixed-length byte strings

    # Separate pos_ and neg_ samples
    pos_indices = [i for i, name in enumerate(names) if name.startswith(b'pos_')]
    neg_indices = [i for i, name in enumerate(names) if name.startswith(b'neg_')]

    # Ensure equal numbers of pos_ and neg_ samples
    num_samples = min(len(pos_indices), len(neg_indices))

    pos_indices = np.random.choice(pos_indices, num_samples, replace=False)
    neg_indices = np.random.choice(neg_indices, num_samples, replace=False)

    balanced_indices = np.concatenate((pos_indices, neg_indices))
    np.random.shuffle(balanced_indices)

    # Split into training, validation, and test datasets
    train_idx = int(len(balanced_indices) * train_percentage)
    val_idx = int(len(balanced_indices) * (train_percentage + val_percentage))

    train_indices = balanced_indices[:train_idx]
    val_indices = balanced_indices[train_idx:val_idx]
    test_indices = balanced_indices[val_idx:]

    # Create training dataset
    train_raw = raw[train_indices]
    train_label = label[train_indices]
    train_names = names[train_indices]

    # Create validation dataset
    val_raw = raw[val_indices]
    val_label = label[val_indices]
    val_names = names[val_indices]

    # Create test dataset
    test_raw = raw[test_indices]
    test_label = label[test_indices]
    test_names = names[test_indices]

    # Count pos_ and neg_ examples in each category
    train_pos_count = sum(1 for name in train_names if name.startswith(b'pos_'))
    train_neg_count = sum(1 for name in train_names if name.startswith(b'neg_'))
    val_pos_count = sum(1 for name in val_names if name.startswith(b'pos_'))
    val_neg_count = sum(1 for name in val_names if name.startswith(b'neg_'))
    test_pos_count = sum(1 for name in test_names if name.startswith(b'pos_'))
    test_neg_count = sum(1 for name in test_names if name.startswith(b'neg_'))

    print(f"Training set: {train_pos_count} pos, {train_neg_count} neg")
    print(f"Validation set: {val_pos_count} pos, {val_neg_count} neg")
    print(f"Test set: {test_pos_count} pos, {test_neg_count} neg")

    # Create subdirectories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    # Save training dataset
    train_path = os.path.join(output_dir, 'train', 'train.h5')
    with h5py.File(train_path, 'w') as f:
        f.create_dataset('raw', data=train_raw)
        f.create_dataset('label', data=train_label)
        f.create_dataset('name', data=train_names)

    # Save validation dataset
    val_path = os.path.join(output_dir, 'validation', 'validation.h5')
    with h5py.File(val_path, 'w') as f:
        f.create_dataset('raw', data=val_raw)
        f.create_dataset('label', data=val_label)
        f.create_dataset('name', data=val_names)

    # Save test dataset
    test_path = os.path.join(output_dir, 'test', 'test.h5')
    with h5py.File(test_path, 'w') as f:
        f.create_dataset('raw', data=test_raw)
        f.create_dataset('label', data=test_label)
        f.create_dataset('name', data=test_names)

    return train_path, val_path, test_path

# Example usage
path = '/Users/aj/Desktop/cspotExampleData/h5_trial/thumbnails.h5'
output_dir = '/Users/aj/Desktop/cspotExampleData/h5_trial'
train_percentage = 0.7
val_percentage = 0.2
split_h5_dataset(path, train_percentage, val_percentage, output_dir)


