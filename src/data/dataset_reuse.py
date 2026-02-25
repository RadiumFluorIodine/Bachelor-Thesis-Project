"""
Dataset Adapter for ReUse Model (2D Spatial).

Strategies:
- Converts Spatio-Temporal data (T, C, H, W) -> Spatial Data (C, H, W).
- Method: Median Composite (Robust against clouds/outliers in time series).

Reference:
Adapted for ReUse architecture inputs.
"""

import torch
import numpy as np

try:
    from src.data.dataset import BiomassDataset
except ImportError:
    from dataset import BiomassDataset

class ReUseDataset(BiomassDataset):
    """
    Wrapper dataset for ReUse model.
    Takes the median of the time dimension to create static spatial input.
    """
    def __init__(self, root_dir, mode='train', augment=False, file_list=None):
        """
        Args:
            root_dir: Path to processed data directory
            mode: 'train', 'val', or 'test'
            augment: Apply data augmentation (only works if mode='train')
            file_list: List of specific filenames to load
        """
        super().__init__(root_dir, mode=mode, augment=augment, file_list=file_list)
        print(f"ReUseDataset (Median Composite) initialized: {len(self)} patches")
        
    def __getitem__(self, idx):
        """
        Get single spatial sample.
        
        Returns:
            dict with:
            - 'image': (C, H, W) - median composite
            - 'label': (H, W) - AGB label
            - 'valid_mask': (H, W) - spatial mask
        """

        # Get original temporal sample
        sample = super().__getitem__(idx)
        sample = super().__getitem__(idx)
        
        image_sequence = sample['image']  # (T, C, H, W)
        temporal_mask = sample['temporal_mask']  # (T,)
        label = sample['label']  # (H, W)
        valid_mask = sample['valid_mask']  # (H, W)
        
        valid_months = temporal_mask > 0  # (T,) boolean


        # Temporal Reduction (Median Composite)
        if valid_months.sum() == 0:
            image_valid = image_sequence
        else:
            # Use only valid months
            image_valid = image_sequence[valid_months]  # (T_valid, C, H, W)
        
        # Temporal reduction (median composite)
        if isinstance(image_valid, torch.Tensor):
            image_spatial = torch.median(image_valid, dim=0).values
        else:
            image_spatial = np.median(image_valid, axis=0)
        
        # Return spatial data
        return {
            'image': image_spatial,  # (C, H, W)
            'label': label,          # (H, W)
            'valid_mask': valid_mask # (H, W)
        }



def collate_fn_reuse(batch):
    """
    Custom collate function for ReUse Dataset.
    
    Stacks samples and adds channel dimension to labels.
    
    Args:
        batch: List of samples from ReUseDataset
    
    Returns:
        dict with batched tensors:
        - 'image': (B, C, H, W)
        - 'label': (B, 1, H, W)  ← Channel dim for ReUse
        - 'valid_mask': (B, H, W)
    """
    # Stack images (B, C, H, W)
    images = torch.stack([item['image'] for item in batch])
    
    # Stack labels and add channel dimension
    labels = torch.stack([item['label'] for item in batch])
    labels = labels.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
    
    # Stack valid masks (B, H, W)
    valid_masks = torch.stack([item['valid_mask'] for item in batch])
    
    return {
        'image': images,            # (B, C, H, W)
        'label': labels,            # (B, 1, H, W)
        'valid_mask': valid_masks   # (B, H, W)
    }


# Code Testing
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from dataset import collate_fn_biomass

    print("-" * 80)
    print("TESTING REUSE DATASET")
    print("-" * 80)

    try:
        # Create dataset
        dataset = ReUseDataset(root_dir='data/processed/lampung/version_2', mode='train')
        print(f"Loaded ReUseDataset dengan {len(dataset)} patches")
        
        # Test Single Sample
        print("\n--- Test 1: Single Sample ---")
        sample = dataset[0]
        
        print(f"✓ Image: {sample['image'].shape} \t→ Expected: (10, 128, 128)")
        assert sample['image'].shape == (10, 128, 128), "❌ Wrong image shape!"
        
        print(f"✓ Label: {sample['label'].shape} \t→ Expected: (128, 128)")
        assert sample['label'].shape == (128, 128), "❌ Wrong label shape!"
        
        print(f"✓ Valid mask: {sample['valid_mask'].shape} \t→ Expected: (128, 128)")
        assert sample['valid_mask'].shape == (128, 128), "❌ Wrong mask shape!"
        
        print("\n Single sample shapes correct!")
        
        # Test DataLoader dengan Batch
        print("\n--- Test 2: DataLoader with Custom Collate ---")
        train_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_reuse  # ← Use custom collate
        )
        
        batch = next(iter(train_loader))
        
        print(f"✓ Batch image: {batch['image'].shape} \t→ Expected: (4, 10, 128, 128)")
        assert batch['image'].shape == (4, 10, 128, 128), "❌ Wrong batch image!"
        
        print(f"✓ Batch label: {batch['label'].shape} \t→ Expected: (4, 1, 128, 128)")
        assert batch['label'].shape == (4, 1, 128, 128), "❌ Wrong batch label!"
        
        print(f"✓ Batch mask: {batch['valid_mask'].shape} \t→ Expected: (4, 128, 128)")
        assert batch['valid_mask'].shape == (4, 128, 128), "❌ Wrong batch mask!"
        
        print("\n All batch shapes correct!")
        

        # Data Statistics
        print("\n--- Test 3: Data Statistics ---")
        print(f"Image range: [{batch['image'].min():.4f}, {batch['image'].max():.4f}]")
        print(f"Label range: [{batch['label'].min():.4f}, {batch['label'].max():.4f}]")
        print(f"Valid mask coverage: {batch['valid_mask'].mean():.2%}")
        
        # Compatibility with ReUse Model
        print("\n--- Test 4: ReUse Model Compatibility ---")
        
        # Simulate ReUse forward pass
        try:
            from src.models.reuse_unet import ReUseUNet
            
            model = ReUseUNet(input_channels=10)
            model.eval()
            
            with torch.no_grad():
                output = model(batch['image'])  # (B, 1, H, W)
            
            print(f"✓ Model output: {output.shape} \t→ Expected: (4, 1, 128, 128)")
            assert output.shape == (4, 1, 128, 128), "❌ Wrong model output!"
            
            # Test loss computation
            criterion = torch.nn.L1Loss()
            loss = criterion(output, batch['label'])
            print(f"✓ Loss computed: {loss.item():.4f}")
            
            print("\n ReUse model compatibility confirmed!")
            
        except ImportError:
            print(" ReUse model not found, skipping model test")


        print("\n" + "-" * 80)
        print("✅ ALL TESTS PASSED!")
        print("-" * 80)
        print(f"Dataset ready for ReUse training!")
        print(f"- Input:  (B, C=10, H=128, W=128)")
        print(f"- Output: (B, 1, H=128, W=128)")
        print(f"- Loss:   MAE between output and label")
        print("-" * 80)
        
    except FileNotFoundError as e:
        print(f"\n❌ FILE NOT FOUND: {e}")
        print("   Make sure 'data/processed/lampung' exists with .npz files")
    except AssertionError as e:
        print(f"\n❌ ASSERTION ERROR: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()