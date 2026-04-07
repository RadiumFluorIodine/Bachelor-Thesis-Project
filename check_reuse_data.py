import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import os

# Setup Path agar bisa import modul
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

try:
    from src.data.dataset_reuse import ReUseDataset
except ImportError:
    from data.dataset_reuse import ReUseDataset

def visualize_reuse_input(dataset, num_samples=3):
    """
    Memvisualisasikan input model ReUse (Median Composite)
    dibandingkan dengan Ground Truth Biomassa.
    """
    print(f"Dataset Size: {len(dataset)}")
    
    # Ambil beberapa sampel acak
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    plt.suptitle("ReUse Model Input Visualization (Median Composite)", fontsize=16)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # Data Image: (C, H, W) -> (10, 128, 128)
        image = sample['image']
        label = sample['label']
        
        # Konversi ke NumPy jika Tensor
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        # RGB Visualization (Bands 4, 3, 2)
        rgb = image[[2, 1, 0], :, :] 
        rgb = np.transpose(rgb, (1, 2, 0)) # (H, W, 3)
        
        # Normalisasi untuk visualisasi
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb_norm = np.clip((rgb - p2) / (p98 - p2), 0, 1)

        # NIR Visualization (False Color - Vegetation)
        nir_idx = 3 
        red_idx = 2
        green_idx = 1
        
        fcc = image[[nir_idx, red_idx, green_idx], :, :]
        fcc = np.transpose(fcc, (1, 2, 0))
        p2, p98 = np.percentile(fcc, (2, 98))
        fcc_norm = np.clip((fcc - p2) / (p98 - p2), 0, 1)

        # Plotting
        ax1 = axes[i, 0] if num_samples > 1 else axes[0]
        ax1.imshow(rgb_norm)
        ax1.set_title(f"Input Median (RGB)\nSample {idx}")
        ax1.axis('off')

        # False Color (NIR) Median
        ax2 = axes[i, 1] if num_samples > 1 else axes[1]
        ax2.imshow(fcc_norm)
        ax2.set_title(f"Input Median (False Color NIR)\nVegetation Highlight")
        ax2.axis('off')

        # Ground Truth (Label AGB)
        ax3 = axes[i, 2] if num_samples > 1 else axes[2]
        im = ax3.imshow(label, cmap='viridis')
        ax3.set_title(f"Target AGB (Label)\nMax: {label.max():.2f} tons/ha")
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    DATA_ROOT = 'data/processed/lampung/version_2' 
    
    print("Loading Dataset...")
    try:
        # Inisialisasi Dataset ReUse
        dataset = ReUseDataset(root_dir=DATA_ROOT, mode='train')
        
        # Jalankan Visualisasi
        visualize_reuse_input(dataset, num_samples=3)
        
        print("\n✅ Visualisasi selesai.")
        print("Pastikan gambar 'Input Median' terlihat bersih dari awan (karena efek median).")
        
    except FileNotFoundError:
        print(f"❌ Error: Folder '{DATA_ROOT}' tidak ditemukan.")
    except Exception as e:
        print(f"❌ Error: {e}")