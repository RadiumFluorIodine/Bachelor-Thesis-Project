"""
Exploratory Data Analysis (EDA) Script for Biomass Estimation - FULL FIXED VERSION
Menghasilkan 4 visualisasi utama untuk Subbab 4.2 Laporan Skripsi:
1. Distribusi Target AGB (Histogram + KDE)
2. Matriks Korelasi Spektral (Multicollinearity Check)
3. Profil Fenologi Multi-Spektral (Visible, Red Edge, NIR, SWIR)
4. Keseimbangan Partisi Dataset (Train vs Val Distribution)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from tqdm import tqdm
import json
from torch.utils.data import DataLoader

# ==========================================
# SETUP & TYPOGRAPHY STANDAR JURNAL
# ==========================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.linewidth'] = 1.0

# Setup paths
current_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.data.dataset import BiomassDataset, get_or_create_global_split, collate_fn_biomass

class ExploratoryDataAnalyzer:
    def __init__(self, data_dir: str, split_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.split_dir = Path(split_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.band_names = ['B02', 'B03', 'B04', 'B05', 'B06', 
                           'B07', 'B08', 'B8A', 'B11', 'B12']
        self.month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Load Normalization Stats
        norm_path = self.data_dir / 'normalization.json'
        with open(norm_path) as f:
            self.norm_stats = json.load(f)
            
    def denormalize_agb(self, agb_norm):
        mean = self.norm_stats.get('agb_log_mean', 4.1454515)
        std = self.norm_stats.get('agb_log_std', 1.1578547)
        return np.expm1((agb_norm * std) + mean)

    def denormalize_s2(self, s2_norm):
        m_k = 's2_mean_per_band' if 's2_mean_per_band' in self.norm_stats else 'mean'
        s_k = 's2_std_per_band' if 's2_std_per_band' in self.norm_stats else 'std'
        mean = np.array(self.norm_stats[m_k]).reshape(1, 1, 10, 1, 1)
        std = np.array(self.norm_stats[s_k]).reshape(1, 1, 10, 1, 1)
        return (s2_norm * std) + mean

    def extract_data_samples(self, max_patches=5000):
        print(f"\n📥 Mengekstrak {max_patches} sampel data untuk analisis EDA...")
        
        train_files, val_files = get_or_create_global_split(
            data_dir=str(self.data_dir), split_dir=str(self.split_dir), test_size=0.2, random_state=42
        )
        
        if len(train_files) > max_patches:
            train_files = np.random.choice(train_files, max_patches, replace=False).tolist()
        
        val_count = int(max_patches * 0.25)
        if len(val_files) > val_count:
            val_files = np.random.choice(val_files, val_count, replace=False).tolist()
            
        train_dataset = BiomassDataset(str(self.data_dir), mode='train', augment=False, file_list=train_files)
        val_dataset = BiomassDataset(str(self.data_dir), mode='val', augment=False, file_list=val_files)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_biomass, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_biomass, num_workers=8)
        
        def get_arrays(loader, desc):
            all_agb, all_s2 = [], []
            for batch in tqdm(loader, desc=desc, ncols=100, file=sys.stdout):
                s2_raw = self.denormalize_s2(batch['image'].numpy())  
                agb_raw = self.denormalize_agb(batch['label'].numpy()) 
                mask = batch.get('valid_mask', None)
                
                for b in range(s2_raw.shape[0]):
                    m = mask[b].numpy().astype(bool).squeeze() if mask is not None else np.ones((agb_raw.shape[-2], agb_raw.shape[-1]), dtype=bool)
                    agb_map = agb_raw[b].squeeze()
                    valid_agb_pixels = agb_map[m]
                    
                    if len(valid_agb_pixels) > 500:
                        idx = np.random.choice(len(valid_agb_pixels), len(valid_agb_pixels)//4, replace=False)
                        valid_agb_pixels = valid_agb_pixels[idx]

                    patch_s2 = s2_raw[b]
                    s2_temporal_profile = np.mean(patch_s2[:, :, m], axis=2) 
                    all_agb.extend(valid_agb_pixels)
                    all_s2.append(s2_temporal_profile)
            return np.array(all_agb), np.stack(all_s2)

        self.agb_train, self.s2_train = get_arrays(train_loader, "Proses Data Latih")
        self.agb_val, _ = get_arrays(val_loader, "Proses Data Validasi")
        print(f"✅ Selesai! Mean Latih saat ini: {np.mean(self.agb_train):.2f} Mg/ha")

    def plot_1_agb_distribution(self):
        print("🎨 Membuat Plot 1: Distribusi AGB...")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.histplot(self.agb_train, bins=80, color='#2ca02c', kde=True, ax=ax, edgecolor='black', alpha=0.6)
        mean_v, med_v = np.mean(self.agb_train), np.median(self.agb_train)
        ax.axvline(mean_v, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_v:.1f}')
        ax.axvline(med_v, color='blue', linestyle=':', linewidth=1.5, label=f'Median: {med_v:.1f}')
        ax.set_title('Distribusi Data Target AGB (Ground Truth)', fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Above-Ground Biomass (Mg/ha)', fontsize=10); ax.set_ylabel('Pixel Count', fontsize=10)
        ax.legend(); ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_1_agb_distribution.png', dpi=600)
        plt.close()

    def plot_2_correlation_matrix(self):
        print("🎨 Membuat Plot 2: Matriks Korelasi...")
        s2_static = np.mean(self.s2_train, axis=1) 
        df = pd.DataFrame(s2_static, columns=self.band_names)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 8}, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
        ax.set_title('Spectral Bands Correlation Matrix', fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_2_correlation_matrix.png', dpi=600)
        plt.close()

    def plot_3_temporal_dynamics(self):
        print("🎨 Membuat Plot 3: Dinamika Temporal (Multi-Band)...")
        fig, ax = plt.subplots(figsize=(8.5, 5))
        mean_m = np.mean(self.s2_train, axis=0) 
        idx = np.arange(12)
        configs = [
            (0, 'B02 (Blue)', '#1f77b4', '--'), (2, 'B04 (Red)', '#d62728', '-'),
            (3, 'B05 (RE 1)', '#9467bd', '-.'), (4, 'B06 (RE 2)', '#8c564b', '-.'),
            (6, 'B08 (NIR)', '#2ca02c', '-'), (8, 'B11 (SWIR 1)', '#ff7f0e', '--'),
        ]
        for b_idx, label, color, style in configs:
            ax.plot(idx, mean_m[:, b_idx], label=label, color=color, linestyle=style, marker='o', markersize=4)
        ax.set_title('Profil Fenologi Reflektansi Multi-Spektral', fontsize=12, fontweight='bold', pad=15)
        ax.set_xticks(idx); ax.set_xticklabels(self.month_names); ax.set_ylabel('Surface Reflectance (DN)')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_3_temporal_dynamics.png', dpi=600, bbox_inches='tight')
        plt.close()

    def plot_4_split_balance(self):
        print("🎨 Membuat Plot 4: Keseimbangan Dataset...")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.kdeplot(self.agb_train, label='Training Set', fill=True, color='#1f77b4', alpha=0.4, common_norm=False, ax=ax)
        sns.kdeplot(self.agb_val, label='Validation Set', fill=True, color='#ff7f0e', alpha=0.4, common_norm=False, ax=ax)
        ax.set_title('Dataset Split Balance (Train vs Val)', fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Above-Ground Biomass (Mg/ha)'); ax.set_ylabel('Probability Density')
        ax.legend(); ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_4_split_balance.png', dpi=600)
        plt.close()

    def run_all(self):
        self.extract_data_samples(max_patches=5000)
        self.plot_1_agb_distribution()
        self.plot_2_correlation_matrix()
        self.plot_3_temporal_dynamics()
        self.plot_4_split_balance()
        print(f"✅ Sukses! Semua plot disimpan di folder: {self.output_dir}")

if __name__ == "__main__":
    DATA_DIR = "data/processed/lampung/version_2"
    SPLIT_DIR = "data/processed/lampung/splits"
    OUTPUT_DIR = "src/results/eda_analysis"
    analyzer = ExploratoryDataAnalyzer(DATA_DIR, SPLIT_DIR, OUTPUT_DIR)
    analyzer.run_all()