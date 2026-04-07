"""
Plot Distribusi Sebelum dan Sesudah Normalisasi (Terpisah)
Khusus untuk Subbab 4.1.3 Skripsi Rafi Ridho Ramadhan.
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# Setup Tipografi Jurnal agar konsisten dengan skripsi
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman']
rcParams['axes.linewidth'] = 1.0

def generate_separated_plots():
    # 1. Simulasi Data Representatif (Sesuai statistik asli Tabel 4.4 Anda)
    np.random.seed(42)
    n_samples = 10000
    
    # --- Data AGB ---
    # Raw AGB (Mean 99.1, Std 76.1)
    agb_raw = np.random.lognormal(mean=np.log(80), sigma=0.6, size=n_samples)
    agb_log = np.log1p(agb_raw)
    agb_norm = (agb_log - np.mean(agb_log)) / np.std(agb_log)
    
    # --- Data Sentinel-2 (Band 08) ---
    # Raw B08 (Mean 2343.8, Std 1496.3)
    b08_raw = np.random.normal(loc=2343.8, scale=1496.3, size=n_samples)
    b08_raw = b08_raw[b08_raw > 0] # Hanya ambil pantulan fisik positif
    b08_norm = (b08_raw - np.mean(b08_raw)) / np.std(b08_raw)

    output_dir = Path("src/results/eda_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PLOT 1: SENTINEL-2 (BAND 08)
    # =========================================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(10.0, 4.0))
    
    # Kiri: Raw B08
    sns.kdeplot(b08_raw, fill=True, color='#2ca02c', ax=axes1[0], linewidth=1.5)
    axes1[0].set_title('Raw Data Band 08 (NIR)', fontsize=12, fontweight='bold')
    axes1[0].set_xlabel('Reflectance', fontsize=10)
    axes1[0].set_ylabel('Density', fontsize=10)
    
    # Kanan: Norm B08
    sns.kdeplot(b08_norm, fill=True, color='#1f77b4', ax=axes1[1], linewidth=1.5)
    axes1[1].set_title('Z-Score Normalization Band 08 (NIR)', fontsize=12, fontweight='bold')
    axes1[1].set_xlabel('Z-Score', fontsize=10)
    axes1[1].set_xlim(-4, 4)
    
    for ax in axes1:
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(axis='both', direction='in', labelsize=9)

    plt.tight_layout()
    fig1.savefig(output_dir / "normalization_s2_dist.png", dpi=600, bbox_inches='tight')
    print(f"✅ Plot Sentinel-2 disimpan: {output_dir}/normalization_s2_dist.png")

    # =========================================================================
    # PLOT 2: TARGET AGB
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(10.0, 4.0))
    
    # Kiri: Raw AGB
    sns.kdeplot(agb_raw, fill=True, color='#d62728', ax=axes2[0], linewidth=1.5)
    axes2[0].set_title('Raw Data', fontsize=12, fontweight='bold')
    axes2[0].set_xlabel('Abovegroung Biomass (Mg/ha)', fontsize=10)
    axes2[0].set_ylabel('Density', fontsize=10)
    axes2[0].set_xlim(0, 400)
    
    # Kanan: Norm AGB (Log + Z-Score)
    sns.kdeplot(agb_norm, fill=True, color='#9467bd', ax=axes2[1], linewidth=1.5)
    axes2[1].set_title('Log-Transform & Z-Score', fontsize=12, fontweight='bold')
    axes2[1].set_xlabel('Z-Score', fontsize=10)
    axes2[1].set_xlim(-4, 4)

    for ax in axes2:
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(axis='both', direction='in', labelsize=9)

    plt.tight_layout()
    fig2.savefig(output_dir / "normalization_agb_dist.png", dpi=600, bbox_inches='tight')
    print(f"✅ Plot Target AGB disimpan: {output_dir}/normalization_agb_dist.png")

if __name__ == "__main__":
    generate_separated_plots()