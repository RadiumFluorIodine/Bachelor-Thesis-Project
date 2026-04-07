import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import os

# Adjust path to find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src.data.dataset import BiomassDataset
except ImportError:
    from data.dataset import BiomassDataset

def visualize_utae_input(dataset, sample_idx=0):
    """
    Visualizes the 12-month time series input for U-TAE.
    Displays RGB for each month + Label + Mask.
    """
    print(f"Dataset Size: {len(dataset)}")
    
    # Load sample
    sample = dataset[sample_idx]
    
    # Unpack data
    # Image: (T, C, H, W) -> (12, 10, 128, 128)
    images = sample['image']
    label = sample['label']
    mask = sample['valid_mask']
    
    if isinstance(images, torch.Tensor):
        images = images.numpy()
        label = label.numpy()
        mask = mask.numpy()

    # Setup Plot: 3 rows x 4 columns for 12 months, plus extra for label/mask
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    plt.suptitle(f"U-TAE Input Time-Series (Sample {sample_idx})", fontsize=16)
    
    # Flatten axes for easier iteration
    ax_flat = axes.flatten()

    # Visualize 12 Months of RGB
    for t in range(12):
        ax = ax_flat[t]
        
        # Shape: (C, H, W) -> (3, 128, 128)
        rgb = images[t, [2, 1, 0], :, :]
        
        # Transpose to (H, W, 3)
        rgb = np.transpose(rgb, (1, 2, 0))
        
        # Normalize for display 
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb_norm = np.clip((rgb - p2) / (p98 - p2), 0, 1)
        
        ax.imshow(rgb_norm)
        ax.set_title(f"Month {t+1}")
        ax.axis('off')

    # Visualize Label (AGB)
    ax_label = ax_flat[12]
    im = ax_label.imshow(label, cmap='viridis')
    ax_label.set_title(f"Target AGB\nMax: {label.max():.2f}")
    ax_label.axis('off')
    plt.colorbar(im, ax=ax_label, fraction=0.046, pad=0.04)

    # Visualize Valid Mask
    ax_mask = ax_flat[13]
    ax_mask.imshow(mask * 255, cmap='gray')
    ax_mask.set_title("Valid Mask")
    ax_mask.axis('off')

    # Hide unused subplots
    ax_flat[14].axis('off')
    ax_flat[15].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_ROOT = 'data/processed/lampung/version_2' 
    
    print("Loading U-TAE Dataset...")
    try:
        dataset = BiomassDataset(root_dir=DATA_ROOT, mode='train')
        
        # Visualize a random sample
        rand_idx = np.random.randint(0, len(dataset))
        visualize_utae_input(dataset, sample_idx=rand_idx)
        
        print("\n✅ Visualization Complete.")
        print("Check the 12-month sequence. You should see seasonal changes or cloud movements.")
        
    except FileNotFoundError:
        print(f"❌ Error: Folder '{DATA_ROOT}' not found.")
    except Exception as e:
        print(f"❌ Error: {e}")