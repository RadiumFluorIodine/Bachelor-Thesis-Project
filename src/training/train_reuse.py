"""
Training Script for ReUse (Regressive U-Net).

Hyperparameters based on Pascarella et al. (2023):
- Loss: MAE 
- Optimizer: Adam 
- Scheduler: ReduceLROnPlateau (patience=25, factor=0.2) 
- Early Stopping: Patience 35 
- Max Epochs: 500 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import logging
from tqdm import tqdm
from pathlib import Path
import numpy as np
import sys
import os


# Setup Path
current_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


# Import module
from src.models.reuse_unet import ReUseUNet
from src.data.dataset_reuse import ReUseDataset
from src.data.dataset import get_or_create_global_split
from src.training.training_utils import (
    RegressionMetrics, 
    CheckpointManager, 
    EarlyStopping,
    setup_logging
)

class ReUseTrainer:
    def __init__(self, model, train_loader, val_loader, device, config, save_dirs):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Setup Logging & Tensorboard ke folder results
        self.logger = setup_logging(
            log_dir=save_dirs['logs'], 
            experiment_name=config['experiment_name']
        )
        self.writer = SummaryWriter(log_dir=save_dirs['runs'])


        # Optimizer: Adam 
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate']
        )
        
        # Loss: MAE 
        self.criterion = nn.L1Loss()
        
        # Scheduler: ReduceLROnPlateau 
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.2, 
            patience=25
        )
        
        # Early Stopping 
        self.early_stopping = EarlyStopping(
            patience=35, 
            mode='min', 
            min_delta=1e-4
        )
        
        # Checkpoint Manager diarahkan ke results/checkpoints
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=save_dirs['checkpoints'],
            keep_best_k=3
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
    
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            images = batch['image'].float().to(self.device)
            labels = batch['label'].float().to(self.device)
        
            # Get valid mask
            valid_mask = batch.get('valid_mask', None)
            if valid_mask is not None:
                valid_mask = valid_mask.float().to(self.device)
                if valid_mask.dim() == 3:
                    valid_mask = valid_mask.unsqueeze(1)
        
            # Ensure consistent shapes
            if labels.dim() == 3:
                labels = labels.unsqueeze(1)
        
            self.optimizer.zero_grad()
            outputs = self.model(images)
        
            # Masked loss
            if valid_mask is not None:
                masked_outputs = outputs * valid_mask
                masked_labels = labels * valid_mask
                loss = self.criterion(masked_outputs, masked_labels) * (
                    valid_mask.numel() / (valid_mask.sum() + 1e-6)
            )
            else:
                loss = self.criterion(outputs, labels)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=1.0
            )

            self.optimizer.step()
        
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

            if valid_mask is not None:
                mask_np = valid_mask.bool().cpu().numpy()
                all_preds.append(outputs.detach().cpu().numpy()[mask_np])
                all_targets.append(labels.cpu().numpy()[mask_np])
            else:
                all_preds.append(outputs.detach().cpu().numpy().flatten())
                all_targets.append(labels.cpu().numpy().flatten())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
    
        train_metrics = RegressionMetrics.compute_all(all_preds, all_targets)
    
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, train_metrics
    

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []
    
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].float().to(self.device)
                labels = batch['label'].float().to(self.device)
            
                # Ensure labels are (B, 1, H, W)
                if labels.dim() == 3:  # (B, H, W)
                    labels = labels.unsqueeze(1)
            
                outputs = self.model(images)  # (B, 1, H, W)
            
                # Apply valid mask if available
                valid_mask = batch.get('valid_mask', None)
                if valid_mask is not None:
                    valid_mask = valid_mask.to(self.device)
                    if valid_mask.dim() == 3:
                        valid_mask = valid_mask.unsqueeze(1)
                
                    # Mask out invalid pixels
                    mask_np = valid_mask.bool().cpu().numpy()
                    preds = outputs.cpu().numpy()[mask_np]
                    targets = labels.cpu().numpy()[mask_np]
                else:
                    preds = outputs.cpu().numpy().flatten()
                    targets = labels.cpu().numpy().flatten()
            
                all_preds.append(preds)
                all_targets.append(targets)
    
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
    
        metrics = RegressionMetrics.compute_all(all_preds, all_targets)
        return metrics

    def fit(self):
        best_mae = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            # Update Scheduler
            self.scheduler.step(val_metrics['mae'])
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1}: "
                f"Train Loss={train_loss:.4f}, MAE={train_metrics['mae']:.4f} | "
                f"Val MAE={val_metrics['mae']:.4f}, RMSE={val_metrics['rmse']:.4f}, "
                f"R²={val_metrics['r2']:.4f}, CCC={val_metrics['ccc']:.4f}"
            )
        
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('MAE/train', train_metrics['mae'], epoch)
            self.writer.add_scalar('MAE/val', val_metrics['mae'], epoch)
            self.writer.add_scalar('RMSE/val', val_metrics['rmse'], epoch)
            self.writer.add_scalar('R2/val', val_metrics['r2'], epoch)
            self.writer.add_scalar('CCC/val', val_metrics['ccc'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Checkpoint
            is_best = val_metrics['mae'] < best_mae
            if is_best:
                best_mae = val_metrics['mae']
                
            self.checkpoint_manager.save(
                self.model, self.optimizer, epoch, 
                val_metrics['mae'], is_best
            )
            
            # Early Stopping
            if self.early_stopping(val_metrics['mae']):
                self.logger.info("Early stopping triggered.")
                break

def get_default_config():
    return {
        'data_root': 'data/processed/lampung/version_2',
        'input_channels': 10,
        'batch_size': 128,     
        'learning_rate': 1e-3, # Default Adam LR
        'num_epochs': 100,
        'num_workers': 9,    
        'pin_memory': True,        
        'experiment_name': 'ReUse'
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    
    config = get_default_config()

    # Setup Directories (Standardized)
    results_dir = os.path.join(root_dir, 'src', 'results')
    
    save_dirs = {
        'logs': os.path.join(results_dir, 'logs', config['experiment_name']),
        'checkpoints': os.path.join(results_dir, 'checkpoints', config['experiment_name']),
        'runs': os.path.join(results_dir, 'runs', config['experiment_name']),
        'splits': os.path.join(root_dir, 'data', 'processed', 'lampung', 'splits') # <-- Tambahkan direktori JSON
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device menggunakan: {device}")
    
    print("Loading dataset and applying Global Split...")
    
    # Get the list of files from JSON
    test_size_ratio = 1.0 - config.get('train_val_split', 0.8) 
    train_files, val_files = get_or_create_global_split(
        data_dir=config['data_root'], 
        split_dir=save_dirs['splits'], 
        test_size=test_size_ratio, 
        random_state=42 
    )
    
    # Create a dataset 
    train_set = ReUseDataset(
        root_dir=config['data_root'], 
        mode='train', 
        augment=False, 
        file_list=train_files
    )
    
    val_set = ReUseDataset(
        root_dir=config['data_root'], 
        mode='val', 
        augment=False, 
        file_list=val_files
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'],    # Load data parallel
        pin_memory=config['pin_memory'],      # Cepat masuk GPU
        persistent_workers=True               # Agar CPU tidak restart worker tiap epoch
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True
    )
    
    # Initialize Model
    model = ReUseUNet(input_channels=config['input_channels'])
    
    # Start Training
    trainer = ReUseTrainer(model, train_loader, val_loader, device, config, save_dirs)
    trainer.fit()


# Unit Testing
def test_pipeline_dry_run():
    print("\n🧪 STARTING DRY RUN TEST...")
    
    # 1. Create Dummy Data
    B, C, H, W = 2, 10, 128, 128
    dummy_input = torch.randn(B, C, H, W)
    dummy_label = torch.randn(B, H, W) # Label is 3D (B, H, W)
    
    # 2. Init Model
    model = ReUseUNet(input_channels=C)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    
    # 3. Forward Pass Test
    print("   Testing Forward Pass...")
    try:
        output = model(dummy_input)
        print(f"   Output Shape: {output.shape}")
        # ReUse output should be (B, 1, H, W)
        assert output.shape == (B, 1, H, W)
    except Exception as e:
        print(f"❌ Forward Failed: {e}")
        return

    # 4. Backward Pass Test
    print("   Testing Backward Pass...")
    try:
        # Label needs unsqueeze to match output (B, 1, H, W)
        loss = criterion(output, dummy_label.unsqueeze(1))
        loss.backward()
        optimizer.step()
        print(f"   Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Backward Failed: {e}")
        return
        
    print("✅ REUSE TRAINING PIPELINE TEST PASSED!")

if __name__ == "__main__":
    #test_pipeline_dry_run()
    main()