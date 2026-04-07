"""
Unified Benchmarking Script.

Compares:
1. U-TAE (Proposed - Spatio-Temporal)
2. ReUse (Baseline DL - Spatial Only)
3. Random Forest (Baseline ML)
4. XGBoost (Baseline ML)
5. Linear Regression (Baseline ML)

References:
- Pascarella et al. (2023): ReUse benchmarking protocol
- Garnot & Landrieu (2021): U-TAE evaluation

Output:
- Consolidated Metrics CSV
- Comparison Plots (6 Types of Charts)
- Statistical summary in Mg/ha
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
import sys

# Untuk menghindari error jika library scienceplots tidak ada
try:
    import scienceplots
except ImportError:
    pass

import warnings
# Menghilangkan UserWarning spesifik dari sklearn parallel
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

# Ensure that the root directory is included in sys.path
current_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import Model Architectures & Utils
from src.models.utae import UTAE
from src.models.reuse_unet import ReUseUNet 
from src.data.dataset import BiomassDataset, collate_fn_biomass, get_or_create_global_split
from src.data.dataset_reuse import ReUseDataset, collate_fn_reuse
from src.training.training_utils import RegressionMetrics

# Configuration
MODEL_PATHS = {
    'U-TAE': {
        'type': 'utae',
        'path': 'src/results/checkpoints/U-TAE/best_model.pt',
        'config': { 
            'input_dim': 10, 'output_dim': 1,
            'encoder_widths': [64,64,128,128],
            'decoder_widths': [32,32,64,128],
            'd_model': 256, 'n_head': 4, 'd_k': 4
        }
    },
    'ReUse': {
        'type': 'reuse',
        'path': 'src/results/checkpoints/ReUse/best_model.pt'
    },
    'Random Forest': {
        'type': 'ml',
        'subtype': 'random_forest',
        'path': 'src/results/checkpoints/baselines/random_forest_20260224_101331.joblib'
    },
    'XGBoost': {
        'type': 'ml',
        'subtype': 'xgboost',
        'path': 'src/results/checkpoints/baselines/xgboost_20260224_101331.joblib'
    },
    'Linear Regression': {
        'type': 'ml',
        'subtype': 'linear_regression',
        'path': 'src/results/checkpoints/baselines/linear_regression_20260224_101331.joblib'
    }
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'data/processed/lampung/version_2'
OUTPUT_DIR = 'src/results/benchmark_final'
SPLIT_DIR = 'data/processed/lampung/splits'

# Normalization Statistics
NORM_STATS = {
    'agb_log_mean': 4.145451545715332,
    'agb_log_std': 1.1578547954559326
}

def denormalize_agb(agb_normalized, norm_stats):
    """Converting AGB from log-space z-scores to pure Mg/ha"""
    agb_log = (agb_normalized * norm_stats['agb_log_std']) + norm_stats['agb_log_mean']
    agb_mgha = np.expm1(agb_log) 
    return np.clip(agb_mgha, a_min=0.0, a_max=None) 

class ModelBenchmarker:
    def __init__(self, data_dir, output_dir, split_dir, test_size_ratio=0.2, seed=42):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        _, val_files = get_or_create_global_split(
            data_dir=data_dir, split_dir=split_dir,
            test_size=test_size_ratio, random_state=seed
        )

        print(f"\n⏳ Loading Benchmark/Test Dataset from: {data_dir}")
        self.test_dataset = BiomassDataset(root_dir=data_dir, mode='val', augment=False, file_list=val_files)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False, 
                                     collate_fn=collate_fn_biomass, num_workers=10, pin_memory=True)

        self.reuse_dataset = ReUseDataset(root_dir=data_dir, mode='val', augment=False, file_list=val_files)
        self.reuse_loader = DataLoader(self.reuse_dataset, batch_size=128, shuffle=False, 
                                      collate_fn=collate_fn_reuse, num_workers=9, pin_memory=True)
        
        print("\n⏳ Extracting Ground Truth...")
        self.y_true_norm = []
        for batch in tqdm(self.test_loader, desc="GT Extraction"):
            labels = batch['label'].numpy()
            mask = batch.get('valid_mask', None)
            if mask is not None:
                m = mask.numpy().astype(bool)
                if m.ndim == 4: m = m.squeeze(1)
                labels = labels[m]
            else: labels = labels.flatten()
            self.y_true_norm.append(labels)
        
        self.y_true_norm = np.concatenate(self.y_true_norm).astype(np.float64)
        self.y_true_phys = denormalize_agb(self.y_true_norm, NORM_STATS)

    def _predict_utae(self, info):
        cfg = info['config']
        model = UTAE(input_dim=cfg['input_dim'], output_dim=cfg['output_dim'],
                     encoder_widths=cfg['encoder_widths'], decoder_widths=cfg['decoder_widths'],
                     d_model=cfg['d_model'], n_head=cfg['n_head'], d_k=cfg['d_k']).to(DEVICE)
        
        ckpt = torch.load(info['path'], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="U-TAE"):
                img = batch['image'].to(DEVICE)
                pos = batch['batch_positions'].to(DEVICE)
                if pos.dim() == 1: pos = pos.unsqueeze(0).expand(img.size(0), -1)
                out = model(img, batch_positions=pos, denormalize=False)
                p = out['agb'].cpu().numpy()
                if p.ndim == 4: p = p.squeeze(1)
                mask = batch.get('valid_mask', None)
                if mask is not None:
                    m = mask.numpy().astype(bool)
                    if m.ndim == 4: m = m.squeeze(1)
                    p = p[m]
                else: p = p.flatten()
                preds.append(p)
        return np.concatenate(preds)
    
    def _predict_reuse(self, info):
        model = ReUseUNet(input_channels=10).to(DEVICE)
        ckpt = torch.load(info['path'], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in tqdm(self.reuse_loader, desc="ReUse"):
                img = batch['image'].to(DEVICE)
                out = model(img).cpu().numpy()
                if out.ndim == 4: out = out.squeeze(1)
                mask = batch.get('valid_mask', None)
                if mask is not None:
                    m = mask.numpy().astype(bool)
                    if m.ndim == 4: m = m.squeeze(1)
                    out = out[m]
                else: out = out.flatten()
                preds.append(out)
        return np.concatenate(preds)
    
    def _predict_ml(self, info):
        loaded = joblib.load(info['path'])
        model = loaded['model'] if isinstance(loaded, dict) else loaded
        scaler = loaded.get('scaler', None) if isinstance(loaded, dict) else None
        f_type = loaded.get('feature_type', 'median') if isinstance(loaded, dict) else 'median'
        preds = []
        for batch in tqdm(self.test_loader, desc=f"{info['subtype']}"):
            imgs = batch['image'].numpy()
            mask = batch.get('valid_mask', None)
            for b in range(imgs.shape[0]):
                img = imgs[b]
                feat = np.median(img, axis=0) if f_type in ['median', 'mean'] else np.mean(img, axis=0)
                X = feat.reshape(feat.shape[0], -1).T
                if scaler: X = scaler.transform(X)
                p = model.predict(X)
                if mask is not None:
                    m = mask[b].numpy().astype(bool).flatten()
                    p = p[m]
                preds.append(p)
        return np.concatenate(preds)

    def run_benchmark(self):
        results = []
        preds_phys = {'Ground Truth': self.y_true_phys}
        
        for name, info in MODEL_PATHS.items():
            if not Path(info['path']).exists(): continue
            print(f"\nEvaluating {name}...")
            start = datetime.now()
            y_p_norm = self._predict_utae(info) if info['type'] == 'utae' else (self._predict_reuse(info) if info['type'] == 'reuse' else self._predict_ml(info))
            dur = (datetime.now() - start).total_seconds()
            
            min_l = min(len(y_p_norm), len(self.y_true_norm))
            y_p_norm, y_t_norm = y_p_norm[:min_l], self.y_true_norm[:min_l]
            y_p_phys, y_t_phys = denormalize_agb(y_p_norm, NORM_STATS), self.y_true_phys[:min_l]

            m_norm = RegressionMetrics.compute_all(y_p_norm.astype(np.float64), y_t_norm)
            m_phys = RegressionMetrics.compute_all(y_p_phys.astype(np.float64), y_t_phys)

            final_m = {
                'Model': name,
                'MAE (Mg/ha)': m_phys['mae'],   
                'RMSE (Mg/ha)': m_phys['rmse'], 
                'R²': m_norm['r2'],     
                'CCC': m_norm['ccc'],   
                'Pearson': m_norm['pearson'],
                'MAPE (%)': m_phys['mape'], 
                'Inference_Time (s)': dur
            }
            results.append(final_m)
            preds_phys[name] = y_p_phys
            print(f"Result: R2: {final_m['R²']:.4f} | MAE: {final_m['MAE (Mg/ha)']:.2f} Mg/ha")

        df = pd.DataFrame(results).set_index('Model')
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(self.output_dir / f'hybrid_metrics_{ts}.csv')
        self.plot_results(df, preds_phys, ts)
        print("\n📊 BENCHMARK RESULTS:\n", df)


    def plot_results(self, df, predictions, timestamp):
        try:
            plt.style.use(['science', 'ieee', 'std-colors', 'no-latex'])
        except:
            plt.style.use('default')

        # Setting Font Global (Standar Jurnal)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['axes.linewidth'] = 1.0

        models = [m for m in predictions.keys() if m != 'Ground Truth']
        max_val = max(self.y_true_phys) * 1.05
        
        # ==============================================================
        # 1. Individual Bar Charts (Metrik Keseluruhan)
        # ==============================================================
        metrics = ['MAE (Mg/ha)', 'RMSE (Mg/ha)', 'R²', 'CCC']
        y_labels = ['Error (Mg/ha)', 'Error (Mg/ha)', 'Score', 'Score']
        clean_labels = [name.replace(" ", "\n") for name in df.index]
        
        for i, met in enumerate(metrics):
            if met in df.columns:
                fig, ax = plt.subplots(figsize=(5.5, 4.0)) 
                x_pos = np.arange(len(df.index))
                bars = ax.bar(x_pos, df[met], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(df)])
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(clean_labels, rotation=0, ha='center', fontsize=10, fontweight='normal')
                
                title_metric = met.split(' ')[0]
                ax.set_title(f"Evaluation Metrics: {title_metric}", fontsize=12, fontweight='bold', pad=15)
                ax.set_ylabel(y_labels[i], fontsize=10, fontweight='normal')
                
                ax.minorticks_on()
                ax.tick_params(axis='x', which='minor', bottom=False) 
                ax.tick_params(axis='both', which='major', direction='in', length=5, width=1.0, right=True, top=True)
                ax.tick_params(axis='y', which='minor', direction='in', length=3, width=0.8, right=True)
                ax.grid(axis='y', linestyle='--', alpha=0.4)
                
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.01), f'{yval:.2f}', 
                            ha='center', va='bottom', fontsize=9)
                             
                plt.tight_layout()
                safe_met = met.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
                plt.savefig(self.output_dir/f'1_metric_{safe_met}_{timestamp}.png', dpi=600, bbox_inches='tight')
                plt.close()

        # ==============================================================
        # 2. Individual Scatter Plots (1:1 Plot)
        # ==============================================================
        n_s = min(8000, len(self.y_true_phys))
        idx = np.random.choice(len(self.y_true_phys), n_s, replace=False)
        
        for name in models:
            fig, ax = plt.subplots(figsize=(5.0, 5.0)) 
            ax.scatter(self.y_true_phys[idx], predictions[name][idx], alpha=0.2, s=4, c='#2c7bb6', edgecolors='none')
            ax.plot([0, max_val], [0, max_val], 'r--', lw=1.5, label='1:1 Line')
            
            ax.set_title(f'{name}', fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel("Observed AGB (Mg/ha)", fontsize=10, fontweight='normal')
            ax.set_ylabel("Predicted AGB (Mg/ha)", fontsize=10, fontweight='normal')
            
            ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
            
            metric_text = f"$R^2 = {df.loc[name, 'R²']:.3f}$\nMAE = {df.loc[name, 'MAE (Mg/ha)']:.1f}"
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray', lw=0.5)
            ax.text(0.05, 0.95, metric_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
            
            ax.tick_params(axis='both', which='major', direction='in', length=5, top=True, right=True, labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='lower right', fontsize=9, frameon=False)
            plt.tight_layout()
            safe_name = name.replace(" ", "_").replace("-", "").lower()
            plt.savefig(self.output_dir/f'2_scatter_{safe_name}_{timestamp}.png', dpi=600, bbox_inches='tight')
            plt.close()

        # ==============================================================
        # 3A. Density Analysis Plot (Sesuai Referensi Gambar)
        # ==============================================================
        best_model = df['RMSE (Mg/ha)'].idxmin()
        y_pred_best = predictions[best_model]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        model_title = f"{best_model} (Proposed)" if best_model == "U-TAE" else best_model
        
        # Menggunakan skala linier, bins=80, dan cmin=1 agar area tanpa piksel menjadi putih
        h2d = ax.hist2d(
            self.y_true_phys, y_pred_best, 
            bins=80, 
            cmap='viridis', 
            cmin=1
        )
        cb = plt.colorbar(h2d[3], ax=ax)
        cb.set_label('Pixel Count', fontsize=10, fontweight='normal')
        cb.ax.tick_params(labelsize=9, direction='in')
        
        ax.plot([0, max_val], [0, max_val], color='red', linestyle='--', lw=2.0, label='1:1 Line')
        ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
        
        ax.set_xlabel('Observed AGB (Mg/ha)', fontsize=10, fontweight='normal')
        ax.set_ylabel('Predicted AGB (Mg/ha)', fontsize=10, fontweight='normal')
        ax.set_title(f'Density Analysis: {model_title}', fontsize=12, fontweight='bold', pad=15)
        
        mae_val, r2_val = df.loc[best_model, 'MAE (Mg/ha)'], df.loc[best_model, 'R²']
        metric_text = f"$R^2 = {r2_val:.3f}$\nMAE = {mae_val:.1f}"
        
        # Box metrik dengan opacity penuh (alpha=1.0) agar tulisan tegas
        props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.8, alpha=1.0)
        ax.text(0.05, 0.95, metric_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        ax.legend(loc='lower right', frameon=False, fontsize=9)
        
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', direction='in', length=5, width=1.0, labelsize=9, top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8, top=True, right=True)
        plt.tight_layout()
        plt.savefig(self.output_dir/f'3A_density_best_{timestamp}.png', dpi=600, bbox_inches='tight')
        plt.close()

        # ==============================================================
        # 4. Error Boxplot 
        # ==============================================================
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        errors = [np.abs(predictions[m] - self.y_true_phys) for m in models]
        box = ax.boxplot(errors, tick_labels=clean_labels, patch_artist=True, showfliers=False)
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(models)]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        for median in box['medians']:
            median.set(color='black', linewidth=1.2)

        ax.set_ylabel('Absolute Error (Mg/ha)', fontsize=10, fontweight='normal')
        ax.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold', pad=15)
        
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', bottom=False) 
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', which='major', direction='in', length=5, right=True, labelsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(self.output_dir/f'4_error_boxplot_{timestamp}.png', dpi=600, bbox_inches='tight')
        plt.close()

        # ==============================================================
        # 5. Residual Density Plot (Dengan bw_adjust pelembut kurva)
        # ==============================================================
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        custom_colors = ['#005b96', '#00b347', '#ff8c00', '#e63946', '#8e44ad']
        
        n_res_samp = min(10000, len(self.y_true_phys))
        idx_res = np.random.choice(len(self.y_true_phys), n_res_samp, replace=False)
        
        for i, name in enumerate(models):
            res_sampled = predictions[name][idx_res] - self.y_true_phys[idx_res]
            label_name = f"{name} (Proposed)" if name == "U-TAE" else name
            
            sns.kdeplot(res_sampled, label=label_name, color=custom_colors[i % len(custom_colors)], 
                        linewidth=2.0, alpha=0.8, bw_adjust=1.5, ax=ax)
        
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax.set_xlim(-100, 100) 
        ax.set_xlabel('Residual Error (Mg/ha)', fontsize=10, fontweight='normal')
        ax.set_ylabel('Density', fontsize=10, fontweight='normal')
        ax.set_title('Residual Density Distribution', fontsize=12, fontweight='bold', pad=15)
        ax.legend(title='Models', loc='upper left', frameon=False, fontsize=9, title_fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(axis='both', which='major', direction='in', length=5, labelsize=9)
        plt.tight_layout()
        plt.savefig(self.output_dir/f'5_residual_density_{timestamp}.png', dpi=600, bbox_inches='tight')
        plt.close()

        # ==============================================================
        # 6. Saturation Analysis 
        # ==============================================================
        bins = np.arange(0, int(max_val) + 50, 50)
        bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        
        for name in models:
            rmse_per_bin = []
            for i in range(len(bins)-1):
                mask = (self.y_true_phys >= bins[i]) & (self.y_true_phys < bins[i+1])
                rmse = np.sqrt(np.mean((predictions[name][mask] - self.y_true_phys[mask])**2)) if np.sum(mask) > 10 else np.nan
                rmse_per_bin.append(rmse)
            ax.plot(bin_labels, rmse_per_bin, marker='o', lw=1.5, markersize=5, label=name)

        ax.set_xlabel('Actual AGB Range (Mg/ha)', fontsize=10, fontweight='normal')
        ax.set_ylabel('RMSE (Mg/ha)', fontsize=10, fontweight='normal')
        ax.set_title('Error Saturation Analysis', fontsize=12, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xticklabels(bin_labels, rotation=30, ha='right', fontsize=9)
        plt.tight_layout()
        plt.savefig(self.output_dir/f'6_saturation_analysis_{timestamp}.png', dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Seluruh Grafik Analisis Visual (6 jenis) berhasil disimpan di {self.output_dir}")

if __name__ == "__main__":
    benchmarker = ModelBenchmarker(DATA_DIR, OUTPUT_DIR, SPLIT_DIR)
    benchmarker.run_benchmark()