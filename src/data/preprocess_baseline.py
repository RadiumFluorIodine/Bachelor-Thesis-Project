"""
Preprocessing for Non-Spatio-Temporal Baselines.

This script prepares data for models that do not handle temporal sequences or spatial context directly 
(e.g., Linear Regression, Random Forest, XGBoost).

Strategies:
1. Temporal Aggregation: Reduce (N, T, C, H, W) -> (N, C, H, W) using mean/median.
2. Flattening: Reduce (N, C, H, W) -> (N*H*W, C) for pixel-wise regression.

"""

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

try:
    from src.data.dataset import BiomassDataset
except ImportError: 
    from dataset import BiomassDataset

def prepare_baseline_data(data_dir, mode='train', aggregation='median',
                          max_patches=5000, pixel_subsample=0.01, file_list=None):
    """
    Loads and preprocesses data for baseline models.
    
    Args:
        data_dir (str): Path to processed data.
        mode (str): 'train', 'val', or 'test'.
        aggregation (str): Method to aggregate temporal dimension ('mean', 'median').
        max_patches (int): Maximum number of patches to load (to prevent RAM crash).
        pixel_subsample (float): Fraction of pixels to keep per patch.
        file_list (list): Specific list of filenames to load (to prevent data leakage).
        
    Returns:
        X (np.ndarray): Feature matrix (num_pixels, num_channels).
        y (np.ndarray): Target vector (num_pixels,).
    """
    print(f"Preparing {mode} data for baselines (Aggregation: {aggregation})...")
    
    # Load Dataset
    dataset = BiomassDataset(root_dir=data_dir, mode=mode, file_list=file_list)

    total_len = len(dataset)
    if max_patches and total_len > max_patches:
        # Put random sampling or sorted sampling
        indices = np.random.choice(total_len, max_patches, replace=False)
        dataset = Subset(dataset, indices)
        print(f"Subsampling dataset: Uses {max_patches}/{total_len} patches.")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 
    
    X_list = []
    y_list = []
    
    for batch in tqdm(loader):
        # image shape: (1, T, C, H, W)
        img = batch['image'].numpy()[0]
        # label shape: (1, H, W)
        label = batch['label'].numpy()[0]
        
        # Temporal Aggregation
        if aggregation == 'mean':
            img_agg = np.mean(img, axis=0) # (C, H, W)
        elif aggregation == 'median':
            img_agg = np.median(img, axis=0) # (C, H, W)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
            
        # Pixel-wise Flattening
        # Transpose (C, H, W) -> (C, H*W) -> (H*W, C)
        c, h, w = img_agg.shape
        pixels = img_agg.transpose(1, 2, 0).reshape(-1, c)
        targets = label.flatten()

        # Pixel Subsampling
        if pixel_subsample < 1.0:
            n_pixels = len(pixels)
            n_keep = int(n_pixels * pixel_subsample)
            # Random indices
            idx = np.random.choice(n_pixels, n_keep, replace=False)
            pixels = pixels[idx]
            targets = targets[idx]
        
        X_list.append(pixels)
        y_list.append(targets)
        
    # Concatenate all patches
    print("Concatenating arrays...")
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    print(f"Final Data Shape: X={X.shape}, y={y.shape}")
    return X, y

if __name__ == "__main__":
    # Example usage for testing
    X, y = prepare_baseline_data(
        'data/processed/lampung/version_2', 
        mode='val', 
        max_patches=1000,      
        pixel_subsample=0.05,
        file_list=None 
    )