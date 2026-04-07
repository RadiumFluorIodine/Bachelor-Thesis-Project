"""
Training Utilities for U-TAE AGB Estimation

Main Features:
1. Robust Loss function (Masked MSE + Smoothness)
2. Metrics computation (R², RMSE, MAE, MAPE)
3. Warmup Cosine Scheduler
4. Early Stopping & Checkpoint Manager

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, List
import json
from pathlib import Path
from datetime import datetime
import logging
import os
import shutil

def setup_logging(log_dir: str = 'logs', experiment_name: str = 'utae_training'):
    """
    Setup logging for training monitoring.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Experiment Name
    
    Returns:
        logger: Configured logger object
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Configure logging
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(experiment_name)
    logger.info(f"Logging initialized. File: {log_file}")
    
    return logger


class BiomassRegressionLoss(nn.Module):
    """
    Custom loss function for AGB regression.
    
    Combines MSE + regularization untuk better training:
    - MSE: Main regression loss
    - Smoothness: Penalizes spatially inconsistent predictions
    - Valid mask: Only computes on valid pixels
    
    Args:
        smoothness_weight: Weight untuk spatial smoothness term
        use_log_scale: Apply log-scale for better distribution
    """
    
    def __init__(self, smoothness_weight: float = 0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # 1. Masked MSE Calculation

        # Square error calculation
        diff = (pred - target) ** 2

        # Apply valid for if provided
        if valid_mask is not None:
            if valid_mask.dtype == torch.bool:
                valid_mask = valid_mask.float()

            diff = diff * valid_mask

            mse_loss = diff.sum() / (valid_mask.sum() + 1e-6)
        else:
            mse_loss = diff.mean()
        
        
        # 2. Smoothness Regularization
        # Penalizes large gradients (encourages smooth predictions)
        if self.smoothness_weight > 0:
            grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
            grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
            
            if valid_mask is not None:
                mask_x = valid_mask[:, :, :, :-1] * valid_mask[:, :, :, 1:]
                mask_y = valid_mask[:, :, :-1, :] * valid_mask[:, :, 1:, :]

                smoothness_x = (grad_x * mask_x).sum() / (mask_x.sum() + 1e-6)
                smoothness_y = (grad_y * mask_y).sum() / (mask_y.sum() + 1e-6)
            else:
                smoothness_x = grad_x.mean()
                smoothness_y = grad_y.mean()
            
            smoothness = (smoothness_x + smoothness_y) / 2
            total_loss = mse_loss + self.smoothness_weight * smoothness
        else:
            total_loss = mse_loss
        
        return total_loss

        

class RegressionMetrics:
    """
    Compute regression metrics for model evaluation.
    
    Metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R²: Coefficient of determination
    - MAPE: Mean Absolute Percentage Error
    - Correlation: Pearson correlation coefficient
    """
    @staticmethod
    def calculate_ccc(p: np.ndarray, t: np.ndarray) -> float:
        """
        Lin's Concordance Correlation Coefficient (CCC).
        Measures agreement on the 1:1 line.
        """
        if len(p) < 2: return 0.0
        
        mean_p = np.mean(p)
        mean_t = np.mean(t)
        
        var_p = np.var(p)
        var_t = np.var(t)
        
        # Pearson Correlation (Covariance / (std_p * std_t))
        # Note: np.cov returns matrix, [0,1] is the covariance
        covariance = np.mean((p - mean_p) * (t - mean_t))
        
        # CCC Formula: (2 * Covariance) / (Var_p + Var_t + (Mean_p - Mean_t)^2)
        numerator = 2 * covariance
        denominator = var_p + var_t + (mean_p - mean_t)**2
        
        if denominator == 0: return 0.0
        
        return numerator / denominator
    

    @staticmethod
    def compute_all(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        p = pred.flatten()
        t = target.flatten()

        if len(p) == 0:
            return {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'mape': 0.0, 'correlation': 0.0}
    
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(p - t))

        # Mean Square Error (MSE)
        mse = np.mean((p - t) ** 2)

        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean((p - t) ** 2))

        # Determination Coeficient (R-Squared Score)
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

        # Mean Absolute Percent Error (MAPE)
        non_zero_mask = t != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((t[non_zero_mask] - p[non_zero_mask]) / t[non_zero_mask])) * 100
        else:
            mape = 0.0

        # Pearson Correlation
        if len(p) > 1:
            pearson_corr = np.corrcoef(p, t)[0, 1]
            if np.isnan(pearson_corr): pearson_corr = 0.0
        else:
            pearson_corr = 0.0

        # Spearman Correlation 
        if len(p) > 1:
            spearman_corr, _ = stats.spearmanr(p, t)
            if np.isnan(spearman_corr): spearman_corr = 0.0
        else:
            spearman_corr = 0.0

        # Concordance Correlation Coefficient (CCC)
        ccc = RegressionMetrics.calculate_ccc(p, t)

        return {
            'mae' : float(mae),
            'rmse' : float(rmse),
            'mse' : float(mse),
            'r2' : float(r2),
            'mape': float(mape),
            'pearson': float(pearson_corr),
            'spearman': float(spearman_corr),
            'ccc': float(ccc)
        }
        

class WarmupScheduler:
    """
    Learning rate scheduler dengan warmup dan cosine annealing.
    
    Training phase:
    1. Warmup: Linear increase dari 0 ke peak_lr (num_warmup_steps)
    2. Cosine annealing: Gradual decrease sesuai cosine curve
    
    Args:
        optimizer: PyTorch optimizer
        peak_lr: Peak learning rate
        num_warmup_steps: Number of warmup steps
        total_steps: Total training steps
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        num_warmup_steps: int,
        total_steps: int
    ):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.num_warmup_steps = num_warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
    
    def step(self):
        """Update learning rate for current step"""
        self.current_step += 1
        
        if self.current_step <= self.num_warmup_steps:
            # Warmup: linear increase
            lr = self.peak_lr * (self.current_step / self.num_warmup_steps)
        else:
            # Cosine annealing
            assert self.total_steps > self.num_warmup_steps, \
            "total_steps must be greater than num_warmup_steps"

            denominator = self.total_steps - self.num_warmup_steps
            progress = (self.current_step - self.num_warmup_steps) / denominator
            
            lr = self.peak_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def reset(self):
        """Reset scheduler (for new training run)"""
        self.current_step = 0

    def get_schedule_info(self) -> Dict:
        """Get schedule info for logging"""
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'num_warmup_steps': self.num_warmup_steps,
            'current_lr': self.get_lr(),
            'progress': f"{100*self.current_step/self.total_steps:.1f}%"
        }


class EarlyStopping:
    """
    Early stopping untuk prevent overfitting.
    
    Monitor validation metric dan stop training jika tidak ada improvement.
    
    Args:
        patience: Jumlah epoch tanpa improvement sebelum stop
        min_delta: Minimum improvement untuk dianggap sebagai improvement
        mode: 'min' untuk minimize metric, 'max' untuk maximize
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, current_value: float) -> bool:
        """
        Check apakah harus stop training.
        
        Args:
            current_value: Validation metric value
        
        Returns:
            True jika harus stop, False otherwise
        """
        
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        # Check improvement
        if self.mode == 'min':
            is_improvement = current_value < (self.best_value - self.min_delta)
        else:  # 'max'
            is_improvement = current_value > (self.best_value + self.min_delta)
        
        if is_improvement:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        
        return False
    
    def get_summary(self) -> str:
        """Summary for logging"""
        if not self.history:
            return "No history"
        
        return (
            f"Best: {self.best_value:.6f}, "
            f"Current: {self.history[-1]:.6f}, "
            f"Patience: {self.counter}/{self.patience}"
        )



class CheckpointManager:
    """
    Manage model checkpoints during training.
    
    Saves:
    - Model weights
    - Optimizer state
    - Training metadata
    - Best model based on validation metric
    
    Args:
        checkpoint_dir: Directory untuk save checkpoints
        keep_best_k: Keep top-k best models
    """
    
    def __init__(self, checkpoint_dir: str = 'checkpoints', keep_best_k: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.keep_best_k = keep_best_k
        self.best_models = []  # List of (metric_value, filepath)
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer, 
        epoch: int,
        metric_value: float,
        is_best: bool = False,
        metadata: Optional[Dict] = None
    ):
        """
        Save checkpoint.
        """
        
        if hasattr(model, 'get_model_summary'):
            model_config = model.get_model_summary()
        else:
            model_config = {"model_class": model.__class__.__name__}
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric_value': metric_value,
            'metadata': metadata or {},
            'model_config': model_config, # Memanggil variabel yang sudah aman
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
            # Maintain top-k best models
            self.best_models.append((metric_value, checkpoint_path))
            self.best_models.sort()  # Sort by metric
            
            # Remove old models if exceeding keep_best_k
            if len(self.best_models) > self.keep_best_k:
                old_metric, old_path = self.best_models.pop()
                if old_path.exists() and old_path != best_path:
                    try:
                        old_path.unlink()
                    except OSError:
                        pass
    
    def load(self, checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer):
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load weights into
            optimizer: Optimizer to load state into
        
        Returns:
            epoch: Epoch dari checkpoint
            metric_value: Metric value dari checkpoint
        """
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            raise RuntimeError(
                f"Model architecture mismatch! "
                f"Checkpoint may be from different model config.\n"
                f"Original error: {e}"
            )
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metric_value']
    
    def load_best(self, model: nn.Module):
        """Load best model weights"""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint['metric_value']
        else:
            raise FileNotFoundError(f"Best model not found at {best_path}")


class TrainingState:
    """
    Track training state dan metrics selama training.
    
    Maintains:
    - Training metrics (loss, learning rate)
    - Validation metrics (mae, rmse, r2)
    - Best metrics seen
    - Training history
    """
    
    def __init__(self, save_dir: str = 'training_logs'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.train_losses = []
        self.train_metrics = {}  # Dict of metric_name -> list
        self.val_metrics = {}    # Dict of metric_name -> list
        self.current_epoch = 0
    
    def update_train(self, loss: float, **metrics):
        """Update training metrics"""
        self.train_losses.append(loss)
        
        for key, value in metrics.items():
            if key not in self.train_metrics:
                self.train_metrics[key] = []
            self.train_metrics[key].append(value)
    
    def update_val(self, **metrics):
        """Update validation metrics"""
        for key, value in metrics.items():
            if key not in self.val_metrics:
                self.val_metrics[key] = []
            self.val_metrics[key].append(value)


    def save_history(self):
        """Save training history to JSON"""
        history = {
            'train_losses': self.train_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
        }
        
        save_path = self.save_dir / 'history.json'
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return save_path
    
    def get_summary(self) -> str:
        """Get simple training summary"""
        summary = "-" * 70 + "\n"
        summary += "TRAINING HISTORY SUMMARY\n"
        summary += "-" * 70 + "\n"
        summary += f"Total Epochs: {len(self.train_losses)}\n"
        summary += f"Last Train Loss: {self.train_losses[-1]:.6f}\n"
        
        if 'mae' in self.val_metrics:
            # Mengambil min/max dari history yang terekam
            best_mae = min(self.val_metrics['mae'])
            summary += f"Best Validation MAE recorded: {best_mae:.6f}\n"
            
        summary += "=" * 70
        return summary
    
    def next_epoch(self): 
        """Increment epoch counter"""
        self.current_epoch += 1
    


class MixedPrecisionTrainer:
    """
    Helper for mixed precision training (faster + less memory).
    
    Usage:
        >>> mp_trainer = MixedPrecisionTrainer()
        >>> with mp_trainer.autocast():
        ...     output = model(images)
        ...     loss = criterion(output, target)
        >>> mp_trainer.backward(loss, optimizer)
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def autocast(self):
        """Context manager for autocasting"""
        return torch.cuda.amp.autocast(enabled=self.enabled)
    
    def backward(self, loss, optimizer):
        """Backward pass with scaling"""
        if self.enabled:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()



# Test Code
if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    print("=" * 80)
    print("TRAINING UTILITIES TEST")
    print("=" * 80)

    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    print(f"📂 Project Root: {root_dir}")

    # Import Dataset
    try:
        from src.data.dataset_reuse import ReUseDataset, collate_fn_reuse # PENTING: Gunakan collate_fn_reuse
        print("✅ Berhasil import ReUseDataset.")
    except ImportError as e:
        print("\n❌ IMPORT ERROR!")
        print(f"   Detail: {e}")
        exit()

    # Load Data
    data_path = os.path.join(root_dir, 'data', 'processed', 'lampung', 'version_2')
    print(f"\n⏳ Loading 1 Batch from: {data_path}")

    try:
        dataset = ReUseDataset(root_dir=data_path, mode='train')
        if len(dataset) == 0:
            print("❌ Dataset kosong!")
            exit()
            
        # PENTING: Gunakan collate_fn_reuse di sini, BUKAN collate_fn_biomass
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_reuse)
        batch = next(iter(loader))
        
        # Ambil Real Target & Valid Mask
        target = batch['label']          # (B, 1, H, W)
        valid_mask = batch['valid_mask'] # (B, H, W) -> Akan di-unsqueeze nanti jika perlu
        
        # Buat "Prediksi Tiruan" (Fake Prediction) untuk simulasi
        noise = torch.randn_like(target) * 5.0 
        pred = target + noise
        
        # Pastikan prediksi tidak negatif 
        pred = torch.clamp(pred, min=0.0)
        pred.requires_grad = True 
        
        print(f"✅ Data Loaded: {target.shape}")
        print(f"   Target Range: [{target.min():.2f}, {target.max():.2f}]")
        print(f"   Fake Pred Range: [{pred.min():.2f}, {pred.max():.2f}]")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # Test 1: Loss Function
    print("\n✅ Test 1: Biomass Regression Loss (Real Data)")
    try:
        criterion = BiomassRegressionLoss(smoothness_weight=0.1)
        
        # Sesuaikan dimensi valid_mask agar sama dengan pred dan target (B, 1, H, W)
        if valid_mask is not None and valid_mask.dim() == 3:
            mask_for_loss = valid_mask.unsqueeze(1)
        else:
            mask_for_loss = valid_mask
            
        loss = criterion(pred, target, mask_for_loss)
        
        print(f"   Calculated Loss: {loss.item():.6f}")
        
        # Test Backward Pass
        loss.backward()
        print("   Backward pass successful (Gradients computed).")
        
    except Exception as e:
        print(f"   ❌ Loss Calculation Error: {e}")
        import traceback
        traceback.print_exc()

    
    # Test 2: Metrics
    print("\n✅ Test 2: Metrics Calculation (Real Data)")
    try:
        # Convert to numpy for metrics
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Filter mask jika perlu (metrics biasanya dihitung pada valid pixel saja)
        if valid_mask is not None:
            # valid_mask dari ReUseDataset ukurannya (B, H, W), perlu di-expand ke (B, 1, H, W) 
            # untuk filtering yang benar pada array numpy
            mask_np = valid_mask.unsqueeze(1).numpy().astype(bool)
            pred_np = pred_np[mask_np]
            target_np = target_np[mask_np]
            
        metrics = RegressionMetrics.compute_all(pred_np, target_np)
        
        print(f"   MAE  : {metrics['mae']:.4f}")
        print(f"   RMSE : {metrics['rmse']:.4f}")
        print(f"   R²   : {metrics['r2']:.4f}")
        print(f"   MAPE : {metrics['mape']:.2f}%")
        print(f"   Corr : {metrics['pearson']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Metrics Calculation Error: {e}")


    # Test 3: Simulated Training Utils
    print("\n✅ Test 3: Simulation (Scheduler, Stopping, Checkpoint)")
    try:
        # Setup Dummy Model & Optimizer
        model = torch.nn.Conv2d(10, 1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Init Utils
        scheduler = WarmupScheduler(optimizer, peak_lr=1e-3, num_warmup_steps=2, total_steps=5)
        early_stopping = EarlyStopping(patience=2, min_delta=0.01)
        
        temp_ckpt_dir = os.path.join(root_dir, 'results', 'temp_test_ckpt')
        ckpt_manager = CheckpointManager(checkpoint_dir=temp_ckpt_dir)
        train_state = TrainingState(save_dir=temp_ckpt_dir)
        
        print("   Simulating 5 Epochs...")
        # Simulated Losses (menurun lalu naik untuk trigger early stopping logic)
        sim_val_losses = [10.0, 8.0, 6.0, 6.1, 6.2]
        
        for epoch, val_loss in enumerate(sim_val_losses):
            # Step Scheduler
            scheduler.step()
            curr_lr = scheduler.get_lr()
            
            # Check Early Stopping
            stop = early_stopping(val_loss)
            is_best = (val_loss == min(sim_val_losses[:epoch+1]))
            
            # Save Checkpoint
            ckpt_manager.save(model, optimizer, epoch, val_loss, is_best=is_best)
            
            # Update Log
            train_state.update_train(loss=val_loss*0.9, lr=curr_lr)
            train_state.update_val(mae=val_loss)
            
            status = "BEST" if is_best else ""
            stop_msg = "-> STOP!" if stop else ""
            print(f"   Epoch {epoch}: Val Loss={val_loss:.2f}, LR={curr_lr:.6f} {status} {stop_msg}")
        
        # Save History
        log_path = train_state.save_history()
        print(f"   History saved to: {log_path}")
        
        # Cleanup
        if os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir)
            print("   Cleanup temporary files successful.")
            
    except Exception as e:
        print(f"   ❌ Simulation Error: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)