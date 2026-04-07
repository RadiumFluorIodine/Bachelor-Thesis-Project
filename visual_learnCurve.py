import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import numpy as np

# ==========================================
# 1. SETUP & TYPOGRAPHY STANDAR JURNAL
# ==========================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.linewidth'] = 1.0

# Lebar 5.5 inci (14cm) dengan tinggi 3.8 inci untuk efek sedikit melebar
FIG_WIDTH = 5.5 
FIG_HEIGHT = 3.8
OUTPUT_DIR = 'src/results/visualizations/training_dynamics'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_training_history(json_file):
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    epochs = range(1, len(data['train_losses']) + 1)
    
    metrics_to_plot = [
        ('mae', 'Mean Absolute Error (MAE)'),
        ('rmse', 'Root Mean Squared Error (RMSE)')
    ]
    
    for metric_key, metric_name in metrics_to_plot:
        train_val = data['train_metrics'][metric_key]
        val_val = data['val_metrics'][metric_key]
        
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
        
        # Plot garis dengan linewidth yang sedikit lebih tebal untuk kejelasan
        ax.plot(epochs, train_val, label='Training', color='#1f77b4', linewidth=2.0)
        ax.plot(epochs, val_val, label='Validation', color='#ff7f0e', linewidth=2.0)
        
        # PERBAIKAN VISUAL: Memulai sumbu Y dari 0 untuk meminimalkan jarak visual
        # Memberikan padding atas sebesar 20% dari nilai maksimal
        max_y = max(max(train_val), max(val_val))
        ax.set_ylim(0, max_y * 1.2)
        
        # Pengaturan Judul (BOLD)
        ax.set_title(f'Learning Curve {metric_name}', fontsize=12, fontweight='bold', pad=15)
        
        # Pengaturan Sumbu (TIDAK BOLD)
        ax.set_xlabel('Epoch', fontsize=10, fontweight='normal')
        ax.set_ylabel('Error Score', fontsize=10, fontweight='normal')
        
        # Styling Grid (Warna abu-abu muda agar tidak mengganggu kurva)
        ax.grid(True, linestyle='--', alpha=0.3, color='#cccccc')
        ax.legend(frameon=True, fontsize=9, loc='upper right')
        
        # Tick parameters (Normal)
        ax.tick_params(axis='both', which='major', labelsize=9, direction='in')
        
        plt.tight_layout()
        
        # Simpan Gambar dengan DPI tinggi untuk laporan skripsi
        save_path = os.path.join(OUTPUT_DIR, f'dynamics_{metric_key}.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ Plot {metric_name} dioptimalkan visualnya di: {save_path}")

if __name__ == "__main__":
    json_path = 'E:/SKRIPSI-RAFI RIDHO RAMADHAN/Bachelor-Thesis-Project-v.1.1/src/results/logs/U-TAE/history.json' 
    plot_training_history(json_path)