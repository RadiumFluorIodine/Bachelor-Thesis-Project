"""
Standalone Evaluation & Plotting Script for Fine-Tuned Models.
Membandingkan performa model U-TAE Zero-Shot (Lampung) vs Fine-Tuned (Kalsel)
pada dataset validasi Kalimantan Selatan.
Format visualisasi dioptimalkan untuk kertas A4.
"""
import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import rcParams

# ==========================================
# SETUP & TYPOGRAPHY STANDAR JURNAL
# ==========================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.linewidth'] = 1.0

# Setup Path
current_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.models.utae import UTAE
from src.data.dataset import BiomassDataset, collate_fn_biomass, get_or_create_global_split
from src.training.training_utils import RegressionMetrics

class FinetuneEvaluator:
    def __init__(self, zero_shot_path: str, finetuned_path: str, config_path: str, kalsel_norm_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        with open(config_path) as f:
            self.config = json.load(f)
            
        with open(kalsel_norm_path) as f:
            self.norm_stats = json.load(f)
            self.agb_log_mean = self.norm_stats.get('agb_log_mean', 4.145451)
            self.agb_log_std = self.norm_stats.get('agb_log_std', 1.157854)

        print("Memuat Model Zero-Shot (Base Lampung)...")
        self.model_zs = self._load_model(zero_shot_path)
        
        print("Memuat Model Fine-Tuned (Kalsel)...")
        self.model_ft = self._load_model(finetuned_path)

    def _load_model(self, model_path: str) -> UTAE:
        model = UTAE(
            input_dim=self.config['input_dim'],
            output_dim=self.config['output_dim'],
            encoder_widths=self.config['encoder_widths'],
            decoder_widths=self.config['decoder_widths'],
            d_model=self.config['d_model'],
            n_head=self.config['n_head'],
            d_k=self.config.get('d_k', 4)
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model.eval()
        return model

    def run_inference(self, model: UTAE, data_loader: DataLoader, desc: str) -> Tuple[np.ndarray, np.ndarray]:
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                pos = batch['batch_positions'].to(self.device)

                if pos.dim() == 1:
                    pos = pos.unsqueeze(0).expand(images.size(0), -1)
                
                output = model(images, batch_positions=pos)
                pred = output['agb']
                
                if 'valid_mask' in batch:
                    mask = batch['valid_mask'].bool().to(self.device).view(-1)
                    pred = pred.view(-1)[mask]
                    labels = labels.view(-1)[mask]
                else:
                    pred, labels = pred.flatten(), labels.flatten()
                
                # Denormalisasi ke Mg/ha
                pred_raw = torch.expm1((pred * self.agb_log_std) + self.agb_log_mean).cpu().numpy()
                labels_raw = torch.expm1((labels * self.agb_log_std) + self.agb_log_mean).cpu().numpy()
                
                all_preds.append(np.clip(pred_raw, 0, None))
                all_targets.append(np.clip(labels_raw, 0, None))
                
        return np.concatenate(all_preds).flatten(), np.concatenate(all_targets).flatten()

    def plot_bar_comparison(self, met_zs: Dict, met_ft: Dict, output_dir: str):
        """Grafik Batang Perbandingan Metrik"""
        labels = ['Zero-Shot\n(Tanpa Adaptasi)', 'Fine-Tuned\n(Adaptasi Kalsel)']
        mae_vals = [met_zs['mae'], met_ft['mae']]
        rmse_vals = [met_zs['rmse'], met_ft['rmse']]
        r2_vals = [met_zs['r2'], met_ft['r2']]
        
        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        # Plot 1: Error
        fig1, ax1 = plt.subplots(figsize=(5.5, 4.0))
        x = np.arange(len(labels))
        width = 0.35
        
        rects1 = ax1.bar(x - width/2, mae_vals, width, label='MAE', color='#d62728', edgecolor='black', alpha=0.8)
        rects2 = ax1.bar(x + width/2, rmse_vals, width, label='RMSE', color='#1f77b4', edgecolor='black', alpha=0.8)
        
        ax1.set_ylabel('Error (Mg/ha)', fontsize=10, fontweight='normal')
        ax1.set_title('Impact of Fine-Tuning on Error Rates', fontsize=12, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10, fontweight='normal')
        ax1.set_ylim(0, max(max(mae_vals), max(rmse_vals)) * 1.2)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        ax1.tick_params(axis='both', which='major', labelsize=9, direction='in')
        autolabel(rects1, ax1); autolabel(rects2, ax1)
        plt.tight_layout()
        fig1.savefig(Path(output_dir) / 'finetune_comparison_error.png', dpi=600, bbox_inches='tight')
        plt.close(fig1)

        # Plot 2: R2
        fig2, ax2 = plt.subplots(figsize=(5.5, 4.0))
        rects3 = ax2.bar(labels, r2_vals, width=0.45, color='#2ca02c', edgecolor='black', alpha=0.8)
        ax2.set_ylabel('R-Squared (R²)', fontsize=10, fontweight='normal')
        ax2.set_title('Impact of Fine-Tuning on Accuracy', fontsize=12, fontweight='bold', pad=15)
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', linestyle='--', alpha=0.4)
        ax2.tick_params(axis='both', which='major', labelsize=9, direction='in')
        autolabel(rects3, ax2)
        plt.tight_layout()
        fig2.savefig(Path(output_dir) / 'finetune_comparison_r2.png', dpi=600, bbox_inches='tight')
        plt.close(fig2)

    def plot_scatter_side_by_side(self, pred_zs: np.ndarray, pred_ft: np.ndarray, targets: np.ndarray, met_zs: Dict, met_ft: Dict, output_dir: str):
        """Scatter plot bersebelahan untuk membandingkan sebaran secara langsung"""
        # Subsampling
        n_s = min(8000, len(targets))
        idx = np.random.choice(len(targets), n_s, replace=False)
        
        # Lebar 8.0 agar muat 2 grafik kotak bersebelahan dalam halaman A4
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))
        max_val = max(np.max(targets), np.max(pred_zs), np.max(pred_ft)) * 1.05
        
        configs = [
            (axes[0], pred_zs, met_zs, 'Zero-Shot Inference', '#c44e52'),
            (axes[1], pred_ft, met_ft, 'Fine-Tuned Inference', '#4c72b0')
        ]
        
        for ax, preds, mets, title, color in configs:
            ax.scatter(targets[idx], preds[idx], alpha=0.2, s=4, c=color, edgecolors='none')
            ax.plot([0, max_val], [0, max_val], 'k--', lw=1.5, label='1:1 Line')
            
            ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
            ax.set_xlabel("Observed AGB (Mg/ha)", fontsize=10, fontweight='normal')
            ax.set_ylabel("Predicted AGB (Mg/ha)", fontsize=10, fontweight='normal')
            
            ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
            
            text = f"$R^2 = {mets['r2']:.3f}$\nMAE = {mets['mae']:.1f}"
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray', lw=0.5)
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
            
            ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='lower right', fontsize=8, frameon=False)
            
        plt.tight_layout()
        plot_path = Path(output_dir) / 'finetune_scatter_comparison.png'
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        plt.close()

    def plot_residual_kde(self, pred_zs: np.ndarray, pred_ft: np.ndarray, targets: np.ndarray, output_dir: str):
        """Plot Kepadatan Residual (Saran Tambahan Analitik)"""
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        
        # Hitung Error (Prediksi - Aktual)
        res_zs = pred_zs - targets
        res_ft = pred_ft - targets
        
        sns.kdeplot(res_zs, label='Zero-Shot', color='#d62728', linewidth=2.0, linestyle='--', ax=ax)
        sns.kdeplot(res_ft, label='Fine-Tuned', color='#1f77b4', linewidth=2.0, ax=ax)
        
        ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Batasi X-axis ke rentang yang masuk akal
        p_min, p_max = np.percentile(res_zs, [2, 98])
        ax.set_xlim(p_min, p_max)
        
        ax.set_xlabel('Residual Error (Predicted - Observed)', fontsize=10, fontweight='normal')
        ax.set_ylabel('Density', fontsize=10, fontweight='normal')
        ax.set_title('Residual Bias Distribution Shift', fontsize=12, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(axis='both', which='major', direction='in', labelsize=9)
        
        plt.tight_layout()
        plot_path = Path(output_dir) / 'finetune_residual_shift.png'
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        plt.close()

def main():
    # PATHS
    kalsel_data_dir = "data/processed/kalsel"
    kalsel_norm_path = "data/processed/kalsel/normalization.json"
    kalsel_split_dir = "data/processed/kalsel/splits"
    
    # Model Paths
    zero_shot_path = "src/results/checkpoints/U-TAE/best_model.pt"
    finetuned_path = "src/results/checkpoints/U-TAE_Finetuned_Kalsel/best_finetuned_model.pt"
    config_path = "src/results/checkpoints/U-TAE/config.json"
    output_dir = "src/results/eval_finetuning"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Init Evaluator
    evaluator = FinetuneEvaluator(zero_shot_path, finetuned_path, config_path, kalsel_norm_path)
    
    # Load Validation Data Kalsel
    _, val_files = get_or_create_global_split(data_dir=kalsel_data_dir, split_dir=kalsel_split_dir, test_size=0.2, random_state=42)
    val_dataset = BiomassDataset(root_dir=kalsel_data_dir, mode='val', augment=False, file_list=val_files)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_biomass, num_workers=4)
    
    # Run Inference
    pred_zs, targets_zs = evaluator.run_inference(evaluator.model_zs, val_loader, "Inferensi Zero-Shot")
    pred_ft, targets_ft = evaluator.run_inference(evaluator.model_ft, val_loader, "Inferensi Fine-Tuned")
    
    # Compute Metrics
    met_zs = RegressionMetrics.compute_all(pred_zs, targets_zs)
    met_ft = RegressionMetrics.compute_all(pred_ft, targets_ft)
    
    print("\n--- HASIL EVALUASI ---")
    print(f"Zero-Shot  -> MAE: {met_zs['mae']:.2f} | RMSE: {met_zs['rmse']:.2f} | R²: {met_zs['r2']:.4f}")
    print(f"Fine-Tuned -> MAE: {met_ft['mae']:.2f} | RMSE: {met_ft['rmse']:.2f} | R²: {met_ft['r2']:.4f}")
    
    improvement = ((met_zs['mae'] - met_ft['mae']) / met_zs['mae']) * 100
    print(f"🌟 Peningkatan (Penurunan Error MAE): {improvement:.2f}%\n")
    
    # Generate Plots
    print("Membuat grafik...")
    evaluator.plot_bar_comparison(met_zs, met_ft, output_dir)
    evaluator.plot_scatter_side_by_side(pred_zs, pred_ft, targets_zs, met_zs, met_ft, output_dir)
    evaluator.plot_residual_kde(pred_zs, pred_ft, targets_zs, output_dir)
    print(f"✅ Seluruh plot berhasil disimpan di folder: {output_dir}")

if __name__ == "__main__":
    main()