"""
Cross-region generalization testing.
Train on Lampung, test on Kalimantan Selatan without retraining.
Format visualisasi dioptimalkan untuk kertas A4.
"""
import os
import sys
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import logging
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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

# Import module
from src.models.utae import UTAE
from src.data.dataset import BiomassDataset, collate_fn_biomass, get_or_create_global_split
from src.training.training_utils import RegressionMetrics

class CrossRegionExperiment:
    def __init__(self, model_path: str, config_path: str, norm_stats_path: str, device: str = "cuda"):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.config_path = config_path
        
        with open(config_path) as f:
            self.config = json.load(f)
        with open(norm_stats_path) as f:
            self.norm_stats = json.load(f)

        self.model = self._load_model()
        
    def _load_model(self) -> UTAE:
        self.logger.info("-"*80)
        self.logger.info("LOADING TRAINED MODEL")
        self.logger.info("-"*80)
        
        model = UTAE(
            input_dim=self.config['input_dim'],
            output_dim=self.config['output_dim'],
            encoder_widths=self.config['encoder_widths'],
            decoder_widths=self.config['decoder_widths'],
            d_model=self.config['d_model'],
            n_head=self.config['n_head'],
            d_k=self.config.get('d_k', 4)
        ).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def load_region_data(self, region_dir: str, region_name: str, file_list: list = None) -> DataLoader:
        self.logger.info(f"Mencari dataset {region_name.upper()} di: {region_dir}")
        dataset = BiomassDataset(root_dir=str(region_dir), mode='val', augment=False, file_list=file_list)
        return DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_biomass, num_workers=4)

    def evaluate_on_region(self, data_loader: DataLoader, region_name: str) -> Tuple[Dict, np.ndarray, np.ndarray]:
        self.logger.info(f"\n{'-'*80}")
        self.logger.info(f"EVALUATING ON: {region_name.upper()}")
        self.logger.info("-"*80)
        
        all_preds_norm, all_targets_norm = [], []
        all_preds_phys, all_targets_phys = [], []
        
        agb_log_mean = self.norm_stats.get('agb_log_mean', 4.145451)
        agb_log_std = self.norm_stats.get('agb_log_std', 1.157854)
        
        with torch.no_grad():
            from tqdm import tqdm
            for batch in tqdm(data_loader, desc=f"Inference ({region_name})"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                batch_positions = batch['batch_positions'].to(self.device)

                if batch_positions.dim() == 1:
                    batch_positions = batch_positions.unsqueeze(0).expand(images.size(0), -1)
                
                output = self.model(images, batch_positions=batch_positions)
                pred = output['agb']
                
                if 'valid_mask' in batch:
                    mask = batch['valid_mask'].bool().to(self.device).view(-1)
                    pred = pred.view(-1)[mask]
                    labels = labels.view(-1)[mask]
                else:
                    pred, labels = pred.flatten(), labels.flatten()
                
                pred_np = pred.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                # Simpan versi Normalisasi (Z-Score Log-space) untuk R2 dan CCC
                all_preds_norm.append(pred_np)
                all_targets_norm.append(labels_np)
                
                # Simpan versi Fisik (Mg/ha) untuk MAE dan RMSE
                pred_raw = np.expm1((pred_np * agb_log_std) + agb_log_mean)
                labels_raw = np.expm1((labels_np * agb_log_std) + agb_log_mean)
                
                all_preds_phys.append(np.clip(pred_raw, 0, None))
                all_targets_phys.append(np.clip(labels_raw, 0, None))
                
        # Gabungkan semua data
        preds_norm = np.concatenate(all_preds_norm).flatten().astype(np.float64)
        targets_norm = np.concatenate(all_targets_norm).flatten().astype(np.float64)
        
        preds_phys = np.concatenate(all_preds_phys).flatten().astype(np.float64)
        targets_phys = np.concatenate(all_targets_phys).flatten().astype(np.float64)
        
        # Hitung Metrik secara terpisah (Persis seperti skrip benchmarking)
        m_norm = RegressionMetrics.compute_all(preds_norm, targets_norm)
        m_phys = RegressionMetrics.compute_all(preds_phys, targets_phys)
        
        # Hybrid Metrics Dictionary
        metrics = {
            'mae': m_phys['mae'],
            'rmse': m_phys['rmse'],
            'mse': m_phys['mse'],
            'mape': m_phys['mape'],
            'r2': m_norm['r2'],
            'ccc': m_norm['ccc'],
            'pearson': m_norm['pearson'],
            'spearman': m_norm['spearman']
        }
        
        self.logger.info(f"\n{region_name.upper()} Results: MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | R²: {metrics['r2']:.4f}")
        return metrics, preds_phys, targets_phys
    
    def plot_generalization_gap(self, metrics_lampung: Dict, metrics_kalsel: Dict, output_dir: str):
        """Membuat 2 grafik terpisah (Error dan Akurasi) dengan ukuran A4."""
        self.logger.info("\n🎨 Membuat grafik perbandingan Bar Chart...")
        
        labels = ['Lampung\n(Source Domain)', 'Kalsel\n(Target Domain)']
        mae_vals = [metrics_lampung['mae'], metrics_kalsel['mae']]
        rmse_vals = [metrics_lampung['rmse'], metrics_kalsel['rmse']]
        r2_vals = [metrics_lampung['r2'], metrics_kalsel['r2']]
        
        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        # PLOT 1: ERROR METRICS
        fig1, ax1 = plt.subplots(figsize=(5.5, 4.0))
        x = np.arange(len(labels))
        width = 0.35
        
        rects1 = ax1.bar(x - width/2, mae_vals, width, label='MAE', color='#4c72b0', edgecolor='black', linewidth=1)
        rects2 = ax1.bar(x + width/2, rmse_vals, width, label='RMSE', color='#dd8452', edgecolor='black', linewidth=1)
        
        ax1.set_ylabel('Error (Mg/ha)', fontsize=10, fontweight='normal')
        ax1.set_title('Cross-Region Error Comparison', fontsize=12, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10, fontweight='normal')
        
        max_err = max(max(mae_vals), max(rmse_vals))
        ax1.set_ylim(0, max_err * 1.2)
        
        ax1.legend(loc='upper left', fontsize=9, frameon=True)
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        ax1.tick_params(axis='both', which='major', labelsize=9, direction='in')
        
        autolabel(rects1, ax1)
        autolabel(rects2, ax1)
        
        plt.tight_layout()
        plot_path_error = Path(output_dir) / 'cross_region_error_bars.png'
        fig1.savefig(plot_path_error, dpi=600, bbox_inches='tight')
        plt.close(fig1)

        # PLOT 2: ACCURACY METRIC
        fig2, ax2 = plt.subplots(figsize=(5.5, 4.0))
        rects3 = ax2.bar(labels, r2_vals, width=0.45, color='#55a868', edgecolor='black', linewidth=1)
        
        ax2.set_ylabel('R-Squared (R²)', fontsize=10, fontweight='normal')
        ax2.set_title('Cross-Region R² Comparison', fontsize=12, fontweight='bold', pad=15)
        
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', linestyle='--', alpha=0.4)
        ax2.tick_params(axis='both', which='major', labelsize=9, direction='in')
        
        autolabel(rects3, ax2)
        
        plt.tight_layout()
        plot_path_r2 = Path(output_dir) / 'cross_region_r2_bars.png'
        fig2.savefig(plot_path_r2, dpi=600, bbox_inches='tight')
        plt.close(fig2)

        self.logger.info(f"✅ Bar Charts tersimpan di: {output_dir}")

    def plot_scatter_generalization(self, kalsel_preds: np.ndarray, kalsel_targets: np.ndarray, metrics_kalsel: Dict, output_dir: str):
        self.logger.info("\n🎨 Membuat grafik Scatter Plot Kalsel...")
        
        n_s = min(8000, len(kalsel_targets))
        idx = np.random.choice(len(kalsel_targets), n_s, replace=False)
        
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        max_val = max(np.max(kalsel_targets), np.max(kalsel_preds)) * 1.05
        
        ax.scatter(kalsel_targets[idx], kalsel_preds[idx], alpha=0.2, s=5, c='#c44e52', edgecolors='none')
        ax.plot([0, max_val], [0, max_val], 'k--', lw=1.5, label='1:1 Ideal Line')
        
        ax.set_title('Generalization Test', fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Observed AGB (Mg/ha)", fontsize=10, fontweight='normal')
        ax.set_ylabel("Predicted AGB (Mg/ha)", fontsize=10, fontweight='normal')
        
        ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
        
        metric_text = f"$R^2 = {metrics_kalsel['r2']:.3f}$\nMAE = {metrics_kalsel['mae']:.1f}"
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray', lw=0.5)
        ax.text(0.05, 0.95, metric_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        ax.tick_params(axis='both', which='major', direction='in', length=5, top=True, right=True, labelsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=9, frameon=False)
        
        plt.tight_layout()
        plot_path = Path(output_dir) / 'cross_region_kalsel_scatter.png'
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        plt.close()
        self.logger.info(f"✅ Scatter Plot Kalsel tersimpan di: {plot_path}")

    def run_cross_region_test(self, lampung_dir: str, kalsel_dir: str, output_dir: str, split_dir: str) -> Dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = {}
        
        self.logger.info("\n" + "="*80)
        self.logger.info("MENGAMBIL GLOBAL SPLIT UNTUK LAMPUNG (SOURCE DOMAIN)")
        self.logger.info("="*80)
        
        _, val_files_lampung = get_or_create_global_split(
            data_dir=lampung_dir, 
            split_dir=split_dir
        )
        
        self.logger.info(f"-> Memuat {len(val_files_lampung)} file validasi Lampung...")
        lampung_loader = self.load_region_data(lampung_dir, 'lampung', file_list=val_files_lampung)
        met_lampung, pred_lampung, tar_lampung = self.evaluate_on_region(lampung_loader, 'lampung')
        results['lampung_val'] = met_lampung
        
        self.logger.info(f"\n-> Memuat SELURUH file Kalsel (Zero-Shot)...")
        kalsel_loader = self.load_region_data(kalsel_dir, 'kalsel', file_list=None)
        met_kalsel, pred_kalsel, tar_kalsel = self.evaluate_on_region(kalsel_loader, 'kalsel')
        results['kalsel_all'] = met_kalsel
        
        results['generalization_gap'] = {
            'mae_gap': met_kalsel['mae'] - met_lampung['mae'],
            'r2_gap': met_lampung['r2'] - met_kalsel['r2']
        }
        
        with open(Path(output_dir) / 'cross_region_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        self.plot_generalization_gap(met_lampung, met_kalsel, output_dir)
        self.plot_scatter_generalization(pred_kalsel, tar_kalsel, met_kalsel, output_dir)
            
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='src/results/checkpoints/U-TAE/best_model.pt')
    parser.add_argument('--config-path', default='src/results/checkpoints/U-TAE/config.json')
    parser.add_argument('--norm-stats', default='data/processed/lampung/version_2/normalization.json')
    
    parser.add_argument('--lampung-dir', default='data/processed/lampung/version_2')
    parser.add_argument('--kalsel-dir', default='data/processed/kalsel') 
    
    parser.add_argument('--split-dir', default='data/splits/lampung', help='Directory containing global split JSONs')
    parser.add_argument('--output-dir', default='src/results/cross_region')
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    experiment = CrossRegionExperiment(args.model_path, args.config_path, args.norm_stats, args.device)
    experiment.run_cross_region_test(args.lampung_dir, args.kalsel_dir, args.output_dir, args.split_dir)

if __name__ == '__main__':
    main()