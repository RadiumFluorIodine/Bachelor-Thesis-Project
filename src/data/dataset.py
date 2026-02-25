"""
PyTorch Dataset for U-TAE AGB Estimation
"""
import torch
from torch.utils.data import Dataset 
import numpy as np
import os
import json
from typing import Dict, Optional
from sklearn.model_selection import train_test_split



def get_or_create_global_split(data_dir: str, split_dir: str, test_size: float = 0.2, random_state: int = 42):
    """
    Check and load the split JSON file. If it does not exist, this function will 
    create it so that all models (Baseline, ReUse, U-TAE) have the same reference.
    """
    os.makedirs(split_dir, exist_ok=True)
    train_json_path = os.path.join(split_dir, 'global_train_files.json')
    val_json_path = os.path.join(split_dir, 'global_val_files.json')

    # If the JSON file already exists, read it directly (to prevent inconsistent splits).
    if os.path.exists(train_json_path) and os.path.exists(val_json_path):
        print(f"Loading existing global splits from {split_dir}")
        with open(train_json_path, 'r') as f:
            train_files = json.load(f)
        with open(val_json_path, 'r') as f:
            val_files = json.load(f)
        return train_files, val_files

    # If there isn't one yet, create a new split globally.
    print(f"Global split files not found. Creating new global split (Train {100-test_size*100}% / Val {test_size*100}%)...")
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    
    if not all_files:
        raise ValueError(f"No .npz files found in {data_dir}")

    # Do a split
    train_files, val_files = train_test_split(
        all_files, test_size=test_size, random_state=random_state
    )

    # Save to JSON to make it a permanent reference
    with open(train_json_path, 'w') as f:
        json.dump(train_files, f, indent=4)
    with open(val_json_path, 'w') as f:
        json.dump(val_files, f, indent=4)
        
    print(f"Saved global splits to {split_dir}")
    return train_files, val_files

class BiomassDataset(Dataset):
    """
    PyTorch Dataset untuk U-TAE AGB Estimation
    
     Expected file structure:
    - root_dir/00000.npz, 00001.npz, ...
    - root_dir/normalization.json (optional, untuk reference)
    
    Setiap .npz berisi:
    - 'image': (T=12, C=10, H=128, W=128) normalized
    - 'label': (H=128, W=128) AGB values normalized
    - 'valid_mask': (H=128, W=128) boolean spatial mask
    """

    def __init__(self, root_dir: str, mode: str = 'train', augment: bool = False, file_list: Optional[list] = None):
        """
        Args:
            root_dir: Directory containing .npz files
            mode: 'train' or 'val' or 'test'
            augment: Apply data augmentation (only for training)
        """
        super().__init__()
        
        self.root_dir = root_dir
        self.mode = mode
        self.augment = augment and (mode == 'train')
        
        # List all .npz files
        if file_list is not None:
            self.files = sorted([os.path.join(root_dir, f) for f in file_list])
        else:
            self.files = sorted([
                os.path.join(root_dir, f) for f in os.listdir(root_dir)
                if f.endswith('.npz')
            ])
        
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {root_dir}")
            
        print(f"Loaded {len(self.files)} patches for {mode} mode.")

        
        # Load normalization stats
        norm_file = os.path.join(root_dir, 'normalization.json')
        if os.path.exists(norm_file):
            with open(norm_file, 'r') as f:
                self.norm_stats = json.load(f)
            print(f"Loaded normalization stats")
        else:
            self.norm_stats = None
            print(f"normalization.json not found")


        # Temporal positions (day of year for each month)
        self.days = torch.tensor(
            [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345], 
            dtype=torch.float32
        )
        
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return single sample
        
        Returns:
            Dict containing:
            - 'image': (T=12, C=10, H=128, W=128)
            - 'label': (H=128, W=128)
            - 'valid_mask': (H=128, W=128)
            - 'temporal_mask': (T=12)
            - 'batch_positions': (T=12)
        """
        try:
            # Load .npz file
            data = np.load(self.files[idx])
            
            # Load Image (T, C, H, W)
            if 'image' not in data:
                raise KeyError(f"Missing 'image' in {self.files[idx]}")
            img = data['image'].astype(np.float32)

            # Load Label (H, W)
            if 'label' not in data:
                raise KeyError(f"Missing 'label' in {self.files[idx]}")
            lbl = data['label'].astype(np.float32)

            # Load Spatial Mask (H, W)
            if 'valid_mask' in data:
                spatial_mask = data['valid_mask'].astype(np.float32)
            else:
                spatial_mask = np.ones_like(lbl, dtype=np.float32)
            
            temporal_mask = (img.sum(axis=(1, 2, 3)) > 0).astype(np.float32)

            # Convert to torch tensors
            img_tensor = torch.from_numpy(img)                      # (12, 10, 128, 128)
            lbl_tensor = torch.from_numpy(lbl)                      # (128, 128)
            spatial_mask_tensor = torch.from_numpy(spatial_mask)    # (128, 128)
            temporal_mask_tensor = torch.from_numpy(temporal_mask)  # (12,)

            if self.augment:
                img_tensor, lbl_tensor, spatial_mask_tensor = self._augment(
                    img_tensor, lbl_tensor, spatial_mask_tensor
                )


            return {
                'image': img_tensor,                    # (12, 10, 128, 128)
                'label': lbl_tensor,                    # (128, 128)
                'valid_mask': spatial_mask_tensor,      # (128, 128)
                'temporal_mask': temporal_mask_tensor,  # (12,) 
                'batch_positions': self.days            # (12,)
            }
            
        except Exception as e:
            print(f"ERROR loading {self.files[idx]}: {str(e)}")
            raise e


    def _augment(self, img, lbl, mask):
        """
        Apply random spatial augmentation
        (Preserves temporal dimension)
        """
        # Random horizontal flip (50%)
        if torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[-1])
            lbl = torch.flip(lbl, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        
        # Random vertical flip (50%)
        if torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[-2])
            lbl = torch.flip(lbl, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        
        # Random 90° rotation (0°, 90°, 180°, 270°)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            img = torch.rot90(img, k=k, dims=[-2, -1])
            lbl = torch.rot90(lbl, k=k, dims=[-2, -1])
            mask = torch.rot90(mask, k=k, dims=[-2, -1])
        
        return img, lbl, mask



def collate_fn_biomass(batch):
    """
    Custom collate function untuk BiomassDataset
    
    Input: List of sample dicts
    Output: Batched dict
    """
    
    # Stack images
    images = torch.stack([item['image'] for item in batch])
    # (B, 12, 10, 128, 128)
    
    # Stack labels
    labels = torch.stack([item['label'] for item in batch])
    # (B, 128, 128) 
    
    # Stack masks 
    masks = torch.stack([item['valid_mask'] for item in batch])
    # (B, 128, 128) 
    
    # Stack temporal masks
    temporal_masks = torch.stack([item['temporal_mask'] for item in batch])
    # (B, 12) 
    
    # Positions - take from first sample (all identical)
    positions = batch[0]['batch_positions']
    # (12,) 
    
    return {
        'image': images,                    # (B, 12, 10, 128, 128)
        'label': labels,                    # (B, 128, 128) 
        'valid_mask': masks,                # (B, 128, 128) 
        'temporal_mask': temporal_masks,    # (B, 12) 
        'batch_positions': positions        # (12,)
    }


# Testing
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("-" * 80)
    print("TESTING BIOMASS DATASET")
    print("-" * 80)
    
    try:
        # Create dataset
        dataset = BiomassDataset(
            root_dir='data/processed/lampung/version_2',
            mode='train',
            augment=True
        )
        print(f"Loaded {len(dataset)} patches")
        
        # Test single sample
        print("\n--- Testing Single Sample ---")
        sample = dataset[0]
        
        print(f"✓ Image: {sample['image'].shape} \t→ Expected: (12, 10, 128, 128)")
        assert sample['image'].shape == (12, 10, 128, 128)
        
        print(f"✓ Label: {sample['label'].shape} \t→ Expected: (128, 128)")
        assert sample['label'].shape == (128, 128)
        
        print(f"✓ Valid mask: {sample['valid_mask'].shape} \t→ Expected: (128, 128)")
        assert sample['valid_mask'].shape == (128, 128)
        
        print(f"✓ Temporal mask: {sample['temporal_mask'].shape} \t→ Expected: (12,)")
        assert sample['temporal_mask'].shape == (12,)
        
        print(f"✓ batch_positions: {sample['batch_positions'].shape} \t→ Expected: (12,)")
        assert sample['batch_positions'].shape == (12,)
        
        # Check temporal mask values
        n_valid_months = sample['temporal_mask'].sum().item()
        print(f"✓ Valid months in sample: {int(n_valid_months)}/12")
        
        print("\nAll single sample shapes correct!")
        
        # Test DataLoader
        print("\n--- Testing DataLoader with Batch ---")
        train_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_biomass
        )
        
        batch = next(iter(train_loader))
        
        print(f"✓ Batch image: {batch['image'].shape} \t→ Expected: (4, 12, 10, 128, 128)")
        assert batch['image'].shape == (4, 12, 10, 128, 128)
        
        print(f"✓ Batch label: {batch['label'].shape} \t→ Expected: (4, 128, 128)")
        assert batch['label'].shape == (4, 128, 128)
        
        print(f"✓ Batch mask: {batch['valid_mask'].shape} \t→ Expected: (4, 128, 128)")
        assert batch['valid_mask'].shape == (4, 128, 128)
        
        print(f"✓ Temporal masks: {batch['temporal_mask'].shape} \t→ Expected: (4, 12)")
        assert batch['temporal_mask'].shape == (4, 12)
        
        print(f"✓ Positions: {batch['batch_positions'].shape} \t→ Expected: (12,)")
        assert batch['batch_positions'].shape == (12,)
        
        print("\nAll batch shapes correct!")
        
        # Statistics
        print("\n--- Data Statistics ---")
        print(f"Image range: [{batch['image'].min():.4f}, {batch['image'].max():.4f}]")
        print(f"Label range: [{batch['label'].min():.4f}, {batch['label'].max():.4f}]")
        print(f"Spatial mask coverage: {batch['valid_mask'].mean():.2%}")
        print(f"Temporal mask (valid months): {batch['temporal_mask'].mean(dim=1)}")
        
        print("\n" + "-" * 80)
        print("ALL TESTS PASSED!")
        print("Dataset ready untuk U-TAE training!")
        print("-" * 80)
        
    except Exception as e:
        print(f"TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
