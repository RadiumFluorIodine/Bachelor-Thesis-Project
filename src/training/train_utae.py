"""
Main Training Loop

Main Features:
1. Complete training pipeline
2. Validation during training
3. Logging & progress tracking
4. Checkpoint saving & Resume Capability
5. Early stopping
6. GPU/CPU compatibility
7. Automatic Plotting (Learning Curve Only)
8. Comprehensive Metrics Tracking (Loss, MAE, RMSE, R2 for Train & Val)
9. Fixed batch_positions dimension bug
10. ASCII/UTF-8 Safe (No Emojis)

Expected Usage:
    python train_utae.py --config config.yaml
    python train_utae.py --resume  # Untuk melanjutkan training yang terhenti
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from typing import Dict, Tuple, Optional
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
            experiment_name=config.get('experiment_name', 'utae_training_no_cliping')
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
        
        # Training Tracking (History for Plotting & Logging)
        self.history = {
            'train_loss': [], 'train_mae': [], 'train_rmse': [], 'train_r2': [],
            'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []
        }
        self.start_epoch = 0
        self.best_val_mae = float('inf')
        self.best_preds_raw = None
        self.best_labels_raw = None

        # Resume Mechanism
        self.resume_checkpoint_path = os.path.join(save_dirs['checkpoints'], "resume_checkpoint.pt")
        if config.get('resume', False) and os.path.exists(self.resume_checkpoint_path):
            self._load_resume_checkpoint()

        # Log configuration
        self.logger.info("=" * 70)
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info("=" * 70)
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 70)

    def _load_resume_checkpoint(self):
        """Memuat checkpoint untuk melanjutkan training."""
        self.logger.info("[RESUME] Menemukan checkpoint! Melanjutkan proses training yang tertunda...")
        checkpoint = torch.load(self.resume_checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        
        # Update history if exists, ensuring backward compatibility
        loaded_history = checkpoint.get('history', {})
        for k in self.history.keys():
            self.history[k] = loaded_history.get(k, [])
        
        if hasattr(self.early_stopping, 'best'):
            self.early_stopping.best = checkpoint.get('best_val_loss_es', float('inf'))
        
        self.logger.info(f"[RESUME] Melanjutkan dari Epoch {self.start_epoch+1}. Best Val MAE sebelumnya: {self.best_val_mae:.4f}")

    def plot_training_results(self):
        """Membuat grafik Learning Curve (Train Loss & Val Loss)."""
        if not self.history['train_loss']:
            return

        self.logger.info("Membuat grafik Learning Curve...")
        plt.figure(figsize=(8, 6))
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        
        # Plot Learning Curve
        plt.plot(epochs_range, self.history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        plt.plot(epochs_range, self.history['val_loss'], 'r-', linewidth=2, label='Val Loss')
        plt.title('Learning Curve', fontweight='bold', fontsize=14)
        plt.xlabel('Epoch', fontweight='bold', fontsize=12)
        plt.ylabel('Loss', fontweight='bold', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plot_path = Path(self.save_dirs['logs']) / 'learning_curve.png'
        plt.savefig(plot_path, dpi=300)
        plt.close() # Menutup figure agar tidak memenuhi memory
        self.logger.info(f"Grafik tersimpan di: {plot_path}")

    def train_epoch(self, epoch_idx) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_preds, all_targets = [], []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch_idx+1:03d}] Train", leave=False, ncols=100, file=sys.stdout)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            batch_positions = batch['batch_positions'].to(self.device)
            
            # --- Perbaikan Bug Dimensi ---
            if batch_positions.dim() == 1:
                batch_positions = batch_positions.unsqueeze(0).expand(images.size(0), -1)
            # -----------------------------

            valid_mask = batch.get('valid_mask', None)
            
            if valid_mask is not None:
                valid_mask = valid_mask.to(self.device)
                if valid_mask.dim() == 3: valid_mask = valid_mask.unsqueeze(1)
            
            if labels.dim() == 3: labels = labels.unsqueeze(1)
            
            self.optimizer.zero_grad()
            output = self.model(images, batch_positions=batch_positions)

            pred = output['agb'] if isinstance(output, dict) else output
            
            loss = self.loss_fn(pred, labels, valid_mask)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get('grad_clip_norm', 1.0)
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if valid_mask is not None:
                mask_np = valid_mask.bool().cpu().numpy()
                all_preds.append(pred.detach().cpu().numpy()[mask_np])
                all_targets.append(labels.cpu().numpy()[mask_np])
            else:
                all_preds.append(pred.detach().cpu().numpy().flatten())
                all_targets.append(labels.cpu().numpy().flatten())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{self.scheduler.get_lr():.6f}"})
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        avg_loss = total_loss / max(num_batches, 1)
        
        metrics = RegressionMetrics.compute_all(all_preds, all_targets)
        self.train_state.update_train(avg_loss, **metrics)
        metrics['loss'] = avg_loss

        self.writer.add_scalar('Loss/train', avg_loss, epoch_idx)
        self.writer.add_scalar('MAE/train', metrics['mae'], epoch_idx)
        self.writer.add_scalar('RMSE/train', metrics['rmse'], epoch_idx)
        self.writer.add_scalar('R2/train', metrics['r2'], epoch_idx)
        self.writer.add_scalar('LR', self.scheduler.get_lr(), epoch_idx)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch_idx) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds, all_targets = [], []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch [{epoch_idx+1:03d}] Val", leave=False, ncols=100, file=sys.stdout)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            batch_positions = batch['batch_positions'].to(self.device)
            
            # --- Perbaikan Bug Dimensi ---
            if batch_positions.dim() == 1:
                batch_positions = batch_positions.unsqueeze(0).expand(images.size(0), -1)
            # -----------------------------
            
            valid_mask = batch.get('valid_mask', None)
            if valid_mask is not None:
                valid_mask = valid_mask.to(self.device)
                if valid_mask.dim() == 3: valid_mask = valid_mask.unsqueeze(1)
            
            if labels.dim() == 3: labels = labels.unsqueeze(1)
            
            output = self.model(images, batch_positions=batch_positions)
            pred = output['agb'] if isinstance(output, dict) else output

            # Compute internal validation loss for history
            loss = self.loss_fn(pred, labels, valid_mask)
            total_loss += loss.item()
            num_batches += 1
            
            if valid_mask is not None:
                mask_np = valid_mask.bool().cpu().numpy()
                all_preds.append(pred.cpu().numpy()[mask_np])
                all_targets.append(labels.cpu().numpy()[mask_np])
            else:
                all_preds.append(pred.cpu().numpy().flatten())
                all_targets.append(labels.cpu().numpy().flatten())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        avg_loss = total_loss / max(num_batches, 1)
        
        metrics = RegressionMetrics.compute_all(all_preds, all_targets)
        metrics['loss'] = avg_loss
        self.train_state.update_val(**metrics)

        self.writer.add_scalar('Loss/val', avg_loss, epoch_idx)
        self.writer.add_scalar('MAE/val', metrics['mae'], epoch_idx)
        self.writer.add_scalar('RMSE/val', metrics['rmse'], epoch_idx)
        self.writer.add_scalar('R2/val', metrics['r2'], epoch_idx)
        
        return metrics, all_preds, all_targets
    
    def fit(self):
        num_epochs = self.config['num_epochs']
        
        for epoch in range(self.start_epoch, num_epochs):
            # Training epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validation epoch
            val_metrics, val_preds, val_targets = self.validate(epoch)
            
            # Update history with all metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['train_r2'].append(train_metrics['r2'])
            
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_r2'].append(val_metrics['r2'])

            # Print ke terminal per epoch (Aman dari Emoji dan karakter khusus)
            self.logger.info(
                f"Epoch [{epoch+1:03d}/{num_epochs}] -> "
                f"Train [Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae']:.4f} | RMSE: {train_metrics['rmse']:.4f} | R2: {train_metrics['r2']:.4f}] || "
                f"Val [Loss: {val_metrics['loss']:.4f} | MAE: {val_metrics['mae']:.4f} | RMSE: {val_metrics['rmse']:.4f} | R2: {val_metrics['r2']:.4f}]"
            )
            
            # Save checkpoint if best model
            is_best = val_metrics['mae'] < self.best_val_mae
            if is_best:
                self.best_val_mae = val_metrics['mae']
                self.best_preds_raw = val_preds
                self.best_labels_raw = val_targets
                self.logger.info(f"New best model found! (MAE: {self.best_val_mae:.4f})")
            
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metric_value=val_metrics['mae'],
                is_best=is_best,
                metadata=self.config
            )

            # Simpan Resume Checkpoint setiap epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_mae': self.best_val_mae,
                'best_val_loss_es': self.early_stopping.best if hasattr(self.early_stopping, 'best') else float('inf'),
                'history': self.history
            }, self.resume_checkpoint_path)
            
            # Check early stopping
            if self.early_stopping(val_metrics['mae']):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Save training history and Plot Results
        history_path = self.train_state.save_history()
        self.plot_training_results()
        
        self.writer.close()
        self.logger.info(f"History saved to {history_path}")
        self.logger.info("TRAINING COMPLETED")

def get_default_config() -> Dict:
    return {
        'input_dim': 10,
        'output_dim': 1,
        'encoder_widths': [64, 64, 64, 128],
        'decoder_widths': [32, 32, 64, 128],
        'd_model': 256,
        'n_head': 16,
        'd_k': 4,
        'encoder_norm': 'group',
        'num_epochs': 300,
        'batch_size': 8,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip_norm': 1.0,
        'smoothness_weight': 0.1,
        'early_stopping_patience': 10,
        'data_dirc': 'data/processed/lampung/no_cliping',
        'split_dirc': 'data/processed/lampung/splits',
        'num_workers': 10,
        'experiment_name': 'U-TAE_No-Cliping',
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'resume': False  # Setting default resume
    }

def main():
    parser = argparse.ArgumentParser(description='Train U-TAE model No Cliping')
    parser.add_argument('--config', type=str, default=None, help='Path to config JSON file')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()
    
    config = get_default_config()
    
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    
    if args.num_epochs: config['num_epochs'] = args.num_epochs
    if args.batch_size: config['batch_size'] = args.batch_size
    if args.lr: config['learning_rate'] = args.lr
    if args.resume: config['resume'] = True
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    results_dir = os.path.join(root_dir, 'src', 'results') 
    save_dirs = {
        'logs': os.path.join(results_dir, 'logs', config['experiment_name']),
        'checkpoints': os.path.join(results_dir, 'checkpoints', config['experiment_name']),
        'runs': os.path.join(results_dir, 'runs', config['experiment_name'])
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)
    
    print("Loading dataset and splitting...")
    train_files, val_files = get_or_create_global_split(
        data_dir=config['data_dirc'], split_dir=config['split_dirc'], test_size=0.2, random_state=config['seed']
    )
    
    train_dataset = BiomassDataset(root_dir=config['data_dirc'], mode='train', file_list=train_files)
    val_dataset = BiomassDataset(root_dir=config['data_dirc'], mode='val', file_list=val_files)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        collate_fn=collate_fn_biomass, num_workers=config['num_workers'],
        pin_memory=True, persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        collate_fn=collate_fn_biomass, num_workers=config['num_workers'],
        pin_memory=True, persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    print("Creating model...")
    model = UTAE(
        input_dim=config['input_dim'], output_dim=config['output_dim'],
        encoder_widths=config['encoder_widths'], decoder_widths=config['decoder_widths'],
        d_model=config['d_model'], n_head=config['n_head'], d_k=config['d_k']
    )
    
    print("Starting training process...")
    trainer = UTAETrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, config=config, save_dirs=save_dirs
    )
    
    trainer.fit()
    
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON VALIDATION SET (BEST MODEL)")
    try:
        trainer.checkpoint_manager.load_best(model)
        print("Best model successfully loaded")
        val_metrics, _, _ = trainer.validate(epoch_idx=-1)
        print(f"Final Validation MAE: {val_metrics['mae']:.4f}")
        print(f"Final Validation RMSE: {val_metrics['rmse']:.4f}")
        print(f"Final Validation R2: {val_metrics['r2']:.4f}")
    except Exception as e:
        print(f"Error loading best model: {e}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
