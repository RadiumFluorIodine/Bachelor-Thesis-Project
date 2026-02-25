"""
U-TAE Main Training Loop

Main Features:

1. Complete training pipeline
2. Validation during training
3. Logging & progress tracking
4. Checkpoint saving
5. Early stopping
6. GPU/CPU compatibility
7. Mixed precision training (optional)


Expected Usage:
    python train_utae.py --config config.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from typing import Dict, Tuple, Optional
import logging
import sys
import os

# Setup Path
current_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


from src.data.dataset import BiomassDataset, collate_fn_biomass, get_or_create_global_split
from src.models.utae import UTAE
from src.training.training_utils import (
    BiomassRegressionLoss,
    RegressionMetrics,
    WarmupScheduler,
    EarlyStopping,
    CheckpointManager,
    TrainingState,
    setup_logging
)

class UTAETrainer:
    """
    Complete trainer untuk U-TAE model.
    
    Menangani:
    - Training loop dengan batches
    - Validation loop
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    - Logging
    
    Args:
        model: U-TAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: torch.device
        config: Configuration dictionary
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict,
        save_dirs: Dict
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.save_dirs = save_dirs
        
        # Setup logging
        self.logger = setup_logging(
            log_dir=save_dirs['logs'],
            experiment_name=config.get('experiment_name', 'U-TAE')
        )

        self.writer = SummaryWriter(log_dir=save_dirs['runs'])

        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Setup loss function
        self.loss_fn = BiomassRegressionLoss(
            smoothness_weight=config.get('smoothness_weight', 0.1)
        )
        
        # Setup learning rate scheduler
        num_epochs = config['num_epochs']
        num_batches = len(train_loader)
        total_steps = num_epochs * num_batches
        num_warmup_steps = int(0.1 * total_steps)  # 10% warmup
        
        self.scheduler = WarmupScheduler(
            optimizer=self.optimizer,
            peak_lr=config['learning_rate'],
            num_warmup_steps=num_warmup_steps,
            total_steps=total_steps
        )
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            min_delta=1e-4,
            mode='min'
        )
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=save_dirs['checkpoints'],
            keep_best_k=3
        )
        
        # Setup training state
        self.train_state = TrainingState(save_dir=save_dirs['logs'])
        
        # Log configuration
        self.logger.info("-" * 70)
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info("-" * 70)
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("-" * 70)

        # Enable mixed precision
        self.use_amp = config.get('use_amp', True)
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
            self.logger.info("Mixed Precision Training (AMP) Enabled")
        else:
            self.scaler = None
            if self.device.type == 'cpu':
                self.logger.info("AMP disabled")
            else:
                self.logger.info("AMP disabled by config")
    
    def train_epoch(self, epoch_idx) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Returns:
            Dictionary dengan training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch_idx+1} Train",
            unit="batch",
            leave=False
        )
        
        for batch in pbar:
            # Prepare batch
            images = batch['image'].to(self.device)  # (B, T, C, H, W)
            labels = batch['label'].to(self.device)  # (B, H, W) or (B, 1, H, W)
            batch_positions = batch['batch_positions'].to(self.device)

            if batch_positions.dim() == 1:
                batch_positions = batch_positions.unsqueeze(0).expand(images.size(0), -1)

            valid_mask = batch.get('valid_mask', None)
            
            if valid_mask is not None:
                valid_mask = valid_mask.to(self.device)  # (B, H, W)
                if valid_mask.dim() == 3: 
                    valid_mask = valid_mask.unsqueeze(1)  # (B, 1, H, W)
            
            # Reshape labels for loss computation
            if labels.dim() == 3: 
                labels = labels.unsqueeze(1)  # (B, 1, H, W)
            
            # Forward pass
            self.optimizer.zero_grad()
  
            with torch.autocast(device_type='cuda', enabled=(self.use_amp and self.device.type == 'cuda')):
                output = self.model(images, batch_positions=batch_positions)

                if isinstance(output, dict): # (B, 1, H, W)
                    pred = output['agb']
                else:
                    pred = output  
            
                # Compute loss
                loss = self.loss_fn(pred, labels, valid_mask)
            
            # Backward pass
            if self.use_amp and self.scaler is not None:
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping 
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('grad_clip_norm', 1.0)
                )
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                
                # Update scaler for next iteration
                self.scaler.update()
            else:
                # Standard backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('grad_clip_norm', 1.0)
                )
                
                # Standard optimizer step
                self.optimizer.step()
            
            
            # Update weights
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions for metrics
            if valid_mask is not None:
                mask_np = valid_mask.bool().cpu().numpy()
                pred_np = pred.detach().cpu().numpy()[mask_np]
                target_np = labels.cpu().numpy()[mask_np]
            else:
                pred_np = pred.detach().cpu().numpy().flatten()
                target_np = labels.cpu().numpy().flatten()
            
            all_preds.append(pred_np)
            all_targets.append(target_np)

            pbar.set_postfix({'samples_processed': sum(len(p) for p in all_preds)})
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_lr():.6f}"
            })
        
        # Compute epoch metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        avg_loss = total_loss / num_batches

        metrics = RegressionMetrics.compute_all(all_preds, all_targets)

        # Update training state
        self.train_state.update_train(avg_loss, **metrics)

        metrics['loss'] = avg_loss
        

        # TensorBoard Logging (Train)
        self.writer.add_scalar('Loss/train', avg_loss, epoch_idx)
        self.writer.add_scalar('MAE/train', metrics['mae'], epoch_idx)
        self.writer.add_scalar('LR', self.scheduler.get_lr(), epoch_idx)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch_idx) -> Dict[str, float]:
        """
        Run validation loop.
        
        Returns:
            Dictionary dengan validation metrics
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        max_samples = 10000
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch_idx+1} Val",
            unit="batch",
            leave=False
        )
        
        for batch in pbar:
            # Prepare batch
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            batch_positions = batch['batch_positions'].to(self.device)
            
            if batch_positions.dim() == 1:
                batch_positions = batch_positions.unsqueeze(0).expand(images.size(0), -1)


            valid_mask = batch.get('valid_mask', None)
            if valid_mask is not None:
                valid_mask = valid_mask.to(self.device)
                if valid_mask.dim() == 3: 
                    valid_mask = valid_mask.unsqueeze(1)
            
            if labels.dim() == 3: 
                labels = labels.unsqueeze(1)
            
            # Forward pass
            output = self.model(images, batch_positions=batch_positions)

            if isinstance(output, dict):
                pred = output['agb']
            else:
                pred = output
            
            # Store predictions for metrics
            if valid_mask is not None:
                mask_np = valid_mask.bool().cpu().numpy()
                pred_np = pred.cpu().numpy()[mask_np]
                target_np = labels.cpu().numpy()[mask_np]
            else:
                pred_np = pred.cpu().numpy().flatten()
                target_np = labels.cpu().numpy().flatten()
            
            all_preds.append(pred_np)
            all_targets.append(target_np)

            # Prevent memory overflow
            if sum(len(p) for p in all_preds) > max_samples:
                # Compute partial metrics and clear
                pass
        
        # Compute validation metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        metrics = RegressionMetrics.compute_all(all_preds, all_targets)
        
        # Update training state
        self.train_state.update_val(**metrics)

        # TensorBoard Logging (Val)
        self.writer.add_scalar('MAE/val', metrics['mae'], epoch_idx)
        self.writer.add_scalar('RMSE/val', metrics['rmse'], epoch_idx)
        self.writer.add_scalar('R2/val', metrics['r2'], epoch_idx)
        
        return metrics
    
    def fit(self):
        """
        Complete training loop.
        
        Melatih model untuk num_epochs dengan early stopping.
        """
        
        num_epochs = self.config['num_epochs']
        best_val_mae = float('inf')
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training epoch
            train_metrics = self.train_epoch(epoch)
            self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae']:.4f} | R2: {train_metrics['r2']:.4f}")
            
            # Validation epoch
            val_metrics = self.validate(epoch)
            self.logger.info(f"Val   - MAE: {val_metrics['mae']:.4f} | RMSE: {val_metrics['rmse']:.4f} | R2: {val_metrics['r2']:.4f}")
            
            # Save checkpoint if best model
            is_best = val_metrics['mae'] < best_val_mae
            if is_best:
                best_val_mae = val_metrics['mae']
                self.logger.info(f"New best model found! (MAE: {best_val_mae:.4f})")
            
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metric_value=val_metrics['mae'],
                is_best=is_best,
                metadata=self.config
            )
            
            # Check early stopping
            if self.early_stopping(val_metrics['mae']):
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}"
                )
                break
        
        # Save training history
        history_path = self.train_state.save_history()
        self.writer.close()
        self.logger.info(f"History saved to {history_path}")
        self.logger.info("TRAINING COMPLETED")


def get_default_config() -> Dict:
    """Get default training configuration"""
    return {
        # Model Parameters
        'input_dim': 10,
        'output_dim': 1,
        'encoder_widths': [64, 64, 128, 128],
        'decoder_widths': [32, 32, 64, 128],
        'd_model': 256,
        'n_head': 4,
        'd_k': 4,
        'encoder_norm': 'group',
        
        # Training
        'num_epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip_norm': 1.0,
        
        # Loss & Metrics
        'smoothness_weight': 0.1,
        
        # Scheduling
        'early_stopping_patience': 10,

        # Mixed Precision Training
        'use_amp': True, 
        
        # Data
        'data_root': 'data/processed/lampung/version_2',
        'train_val_split': 0.8,
        'num_workers': 10,
        
        # Other
        'experiment_name': 'U-TAE',
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }


def validate_config(config: Dict) -> Dict:
    """Validate and set defaults for config"""
    
    # Required parameters
    required = ['input_dim', 'output_dim', 'data_root']
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config parameter: {key}")
    
    # Validate ranges
    assert config['batch_size'] > 0, "batch_size must be positive"
    assert config['learning_rate'] > 0, "learning_rate must be positive"
    assert 0 < config['train_val_split'] < 1, "train_val_split must be in (0,1)"
    
    # Validate encoder/decoder widths
    encoder_widths = config.get('encoder_widths', [64, 64, 128, 128])
    decoder_widths = config.get('decoder_widths', [32, 32, 64, 128])
    assert len(encoder_widths) == len(decoder_widths), \
        "encoder_widths and decoder_widths must have same length"
    assert encoder_widths[-1] == decoder_widths[-1], \
        "Last encoder and decoder widths must match"
    
    return config


def main():
    """Main training script"""
    
    parser = argparse.ArgumentParser(description='Train U-TAE model')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config JSON file'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate'
    )

    parser.add_argument(
        '--resume',
        type=str, 
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_default_config()
    config = validate_config(config)

    if args.config:
        with open(args.config) as f:
            user_config = json.load(f)
        config.update(user_config)
    
    # Override with command line arguments
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'])
    print(f"Using device: {device}")

    # Setup Directories 
    results_dir = os.path.join(root_dir, 'src', 'results') 
    
    save_dirs = {
        'logs': os.path.join(results_dir, 'logs', config['experiment_name']),
        'checkpoints': os.path.join(results_dir, 'checkpoints', config['experiment_name']),
        'runs': os.path.join(results_dir, 'runs', config['experiment_name']),
        'splits': os.path.join(root_dir, 'data', 'processed', 'lampung', 'splits') 
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)

    print("Loading dataset and applying Global Split...")
    
    # Dapatkan file_list untuk train dan val dari JSON
    test_size_ratio = 1.0 - config['train_val_split']
    train_files, val_files = get_or_create_global_split(
        data_dir=config['data_root'], 
        split_dir=save_dirs['splits'], 
        test_size=test_size_ratio, 
        random_state=config['seed']
    )
    
    # Create a training dataset 
    train_dataset = BiomassDataset(
        root_dir=config['data_root'],
        mode='train',
        augment=True,
        file_list=train_files
    )
    
    # Create a val dataset
    val_dataset = BiomassDataset(
        root_dir=config['data_root'],
        mode='val',
        augment=False,
        file_list=val_files
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_biomass,
        num_workers=config['num_workers'],
        pin_memory=True, # Ubah ke True jika menggunakan GPU
        persistent_workers=True 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_biomass,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True # Ubah ke True jika menggunakan GPU
    )
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Create model
    print("Creating model...")
    model = UTAE(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        encoder_widths=config['encoder_widths'],
        decoder_widths=config['decoder_widths'],
        d_model=config['d_model'],
        n_head=config['n_head'],
        d_k=config['d_k']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer and fit
    print("Starting training...")
    trainer = UTAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        save_dirs=save_dirs
    )
    
    trainer.fit()
    
    # Load best model and final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON VALIDATION SET")
    
    try:
        trainer.checkpoint_manager.load_best(model)
        print("✅ Best model loaded")
        
        val_metrics = trainer.validate(epoch_idx=-1)
        print(f"Final Validation MAE: {val_metrics['mae']:.4f}")
        print(f"Final Validation RMSE: {val_metrics['rmse']:.4f}")
        print(f"Final Validation R²: {val_metrics['r2']:.4f}")
        print(f"Final Validation MAPE: {val_metrics['mape']:.2f}%")
        print(f"Final Validation CCC: {val_metrics['ccc']:.4f}")
    except Exception as e:
        print(f"Error loading best model: {e}")
    
    print("✅ TRAINING COMPLETE")
    print("=" * 70)

def test_mixed_precision():
    """Test mixed precision training setup"""
    print("\n" + "="*70)
    print("TESTING MIXED PRECISION TRAINING")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cpu':
        print("⚠️  Skipping test (CPU detected, AMP requires CUDA)")
        return
    
    # Create dummy data
    # Create dummy data
    B, T, C, H, W = 2, 12, 10, 128, 128
    images = torch.randn(B, T, C, H, W).to(device)
    labels = torch.randn(B, 1, H, W).to(device)
    positions = torch.arange(T).float().unsqueeze(0).expand(B, -1).to(device)
    
    # Create model
    model = UTAE(input_dim=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = BiomassRegressionLoss()
    
    # Test with AMP
    print("\n1️⃣  Testing WITH AMP:")
    scaler = torch.cuda.amp.GradScaler()
    
    try:
        with torch.cuda.amp.autocast(enabled=True):
            output = model(images, batch_positions=positions)
            pred = output['agb']
            loss = criterion(pred, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        print(f"   ✅ AMP Forward/Backward successful")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Scaler scale: {scaler.get_scale():.1f}")
    except Exception as e:
        print(f"   ❌ AMP Test failed: {e}")
        return
    
    # Test without AMP
    print("\n2️⃣  Testing WITHOUT AMP:")
    optimizer.zero_grad()
    
    try:
        with torch.cuda.amp.autocast(enabled=False):
            output = model(images, batch_positions=positions)
            pred = output['agb']
            loss = criterion(pred, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        print(f"   ✅ Standard training successful")
        print(f"   Loss: {loss.item():.6f}")
    except Exception as e:
        print(f"   ❌ Standard test failed: {e}")
        return
    
    print("\n" + "="*70)
    print("✅ ALL MIXED PRECISION TESTS PASSED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    #test_mixed_precision()
    main()