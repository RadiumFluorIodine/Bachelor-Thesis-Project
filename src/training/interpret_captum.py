"""
Model Interpretation using Captum (Integrated Gradients).

Analyzes the influence of:
1. Sentinel-2 spectral bands
2. Temporal dimensions (months)
3. Spatial patterns (Overlay & Edge Bias)

on AGB predictions using the trained U-TAE model.
Format visualisasi dioptimalkan penuh untuk standar jurnal dan kertas A4.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients
from matplotlib import rcParams

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

# Import project modules
from src.models.utae import UTAE
from src.data.dataset import BiomassDataset, collate_fn_biomass, get_or_create_global_split

class ModelInterpreter:
    def __init__(
        self,
        model: UTAE,
        device: torch.device,
        output_dir: str = 'src/results/interpretation'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.band_names = ['B2', 'B3', 'B4', 'B5', 'B6', 
                           'B7', 'B8', 'B8A', 'B11', 'B12']
        self.month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
    def _forward_func(self, inputs: torch.Tensor, batch_positions: torch.Tensor):
        def forward(x):
            curr_batch_size = x.shape[0]
            expanded_bp = batch_positions.expand(curr_batch_size, -1)
            outputs = self.model(x, batch_positions=expanded_bp)
            agb_map = outputs['agb']
            
            if agb_map.dim() == 4: 
                return agb_map.sum(dim=(1, 2, 3))
            else: 
                return agb_map.sum(dim=(1, 2))
        return forward
    
    def compute_attributions(
        self,
        input_sample: torch.Tensor,
        batch_positions: torch.Tensor,
        n_steps: int = 50
    ) -> tuple:
        print(f"Input shape: {input_sample.shape}")
        input_sample = input_sample.to(self.device).requires_grad_()
        batch_positions = batch_positions.to(self.device)
        
        forward_func = self._forward_func(input_sample, batch_positions)
        ig = IntegratedGradients(forward_func)
        
        print("Computing attributions...")
        attributions, delta = ig.attribute(
            input_sample,
            n_steps=n_steps,
            internal_batch_size=1,
            return_convergence_delta=True
        )
        
        print(f"Attributions computed | Delta: {delta.item():.6f}")
        return attributions, delta
    
    def analyze_band_importance(self, attributions: torch.Tensor) -> np.ndarray:
        attr_np = attributions.detach().cpu().numpy()
        band_importance = np.mean(np.abs(attr_np), axis=(0, 1, 3, 4))
        band_importance_norm = band_importance / np.sum(band_importance)
        
        self._plot_band_importance(band_importance_norm)
        return band_importance_norm
    
    def _plot_band_importance(self, band_importance: np.ndarray):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        colors = plt.cm.viridis(band_importance / band_importance.max())
        bars = ax.bar(self.band_names, band_importance, color=colors, 
                     edgecolor='black', linewidth=1.0)
        
        for bar, val in zip(bars, band_importance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_title('Spectral Band Importance for AGB Estimation', fontsize=11, fontweight='bold', pad=15)
        ax.set_xlabel('Sentinel-2 Bands', fontsize=10, fontweight='normal')
        ax.set_ylabel('Relative Importance Score', fontsize=10, fontweight='normal')
        
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(0, band_importance.max() * 1.15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'band_importance.png', dpi=600, bbox_inches='tight')
        plt.close()
    
    def analyze_temporal_importance(self, attributions: torch.Tensor) -> np.ndarray:
        attr_np = attributions.detach().cpu().numpy()
        temporal_importance = np.mean(np.abs(attr_np), axis=(0, 2, 3, 4))
        temporal_importance_norm = temporal_importance / np.sum(temporal_importance)
        
        self._plot_temporal_importance(temporal_importance_norm)
        return temporal_importance_norm
    
    def _plot_temporal_importance(self, temporal_importance: np.ndarray):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        months = np.arange(1, 13)
        
        ax.plot(months, temporal_importance, marker='o', linewidth=2.0,
                markersize=6, color='coral', label='Importance')
        ax.fill_between(months, temporal_importance, alpha=0.3, color='coral')
        
        for x, y in zip(months, temporal_importance):
            ax.text(x, y + 0.002, f'{y*100:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_title('Temporal Importance Across 12 Months', fontsize=11, fontweight='bold', pad=15)
        ax.set_xlabel('Month', fontsize=10, fontweight='normal')
        ax.set_ylabel('Relative Importance Score', fontsize=10, fontweight='normal')
        
        ax.set_xticks(months)
        ax.set_xticklabels(self.month_names, rotation=45, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_importance.png', dpi=600, bbox_inches='tight')
        plt.close()

    def analyze_spatial_patterns(self, attributions: torch.Tensor):
        attr_np = attributions.detach().cpu().numpy()
        spatial_attr = np.mean(np.abs(attr_np), axis=(0, 1, 2))
        self._plot_spatial_attribution(spatial_attr)
    
    def _plot_spatial_attribution(self, spatial_attr: np.ndarray):
        """
        Mengganti Heatmap Global dengan Analisis Distribusi Bias Tepi (Edge Bias Check).
        Mengevaluasi apakah model membagi fokusnya secara merata di seluruh bentang 128x128.
        """
        x_profile = np.mean(spatial_attr, axis=0) # Profil sumbu X (Kolom)
        y_profile = np.mean(spatial_attr, axis=1) # Profil sumbu Y (Baris)
        
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        
        ax.plot(x_profile, label='X-Axis (Horizontal) Profile', color='#1f77b4', linewidth=2.0)
        ax.plot(y_profile, label='Y-Axis (Vertical) Profile', color='#ff7f0e', linewidth=2.0, linestyle='--')
        
        ax.set_title('Spatial Focus Distribution (Edge Bias Check)', fontsize=11, fontweight='bold', pad=15)
        ax.set_xlabel('Pixel Position (0 to 127)', fontsize=10, fontweight='normal')
        ax.set_ylabel('Average Attribution Score', fontsize=10, fontweight='normal')
        
        ax.set_xlim(0, 127)
        max_val = max(np.max(x_profile), np.max(y_profile))
        ax.set_ylim(0, max_val * 1.3)
        
        ax.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
        ax.legend(frameon=True, fontsize=9, loc='lower center')
        ax.tick_params(axis='both', which='major', labelsize=9, direction='in')
        
        plt.tight_layout()
        save_path = self.output_dir / 'spatial_bias_profile.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Spatial Bias Profile tersimpan di: {save_path}")
    
    def create_combined_heatmap(self, band_importance: np.ndarray, temporal_importance: np.ndarray):
        """
        Menciptakan heatmap gabungan Band-Temporal dengan grid lines dan anotasi 4 desimal.
        """
        heatmap_data = np.outer(band_importance, temporal_importance)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        
        sns.heatmap(
            heatmap_data, 
            xticklabels=self.month_names, 
            yticklabels=self.band_names,
            cmap='YlOrRd', 
            annot=True,              
            fmt='.4f',               
            linewidths=0.5,          
            linecolor='white',       
            annot_kws={"size": 6.5}, 
            cbar_kws={'label': 'Importance Score'}, 
            ax=ax
        )
        
        ax.set_title('Band-Temporal Importance Heatmap', fontsize=11, fontweight='bold', pad=15)
        ax.set_xlabel('Month', fontsize=10, fontweight='normal')
        ax.set_ylabel('Spectral Band', fontsize=10, fontweight='normal')
        
        plt.xticks(rotation=0) 
        
        plt.tight_layout()
        save_path = self.output_dir / 'combined_importance_heatmap.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ Combined heatmap (4 desimal + grid) tersimpan di: {save_path}")

    def analyze_single_spatial_sample(self, single_attr: torch.Tensor, single_image: torch.Tensor, timestep_idx: int = 1):
        """
        Visualisasi Spasial Lanjutan (Enhanced Spatial Attribution):
        1. RGB Asli
        2. Overlay Heatmap (Transparan di atas Grayscale)
        """
        # 1. Ekstrak RGB
        rgb_tensor = single_image[0, timestep_idx, [2, 1, 0], :, :].cpu()
        rgb_img = rgb_tensor.permute(1, 2, 0).numpy()
        p2, p98 = np.percentile(rgb_img, (2, 98))
        rgb_norm = np.clip((rgb_img - p2) / (p98 - p2 + 1e-8), 0, 1)
        gray_img = np.mean(rgb_norm, axis=2)

        # 2. Ekstrak Atribusi (Hanya Absolut)
        attr_np = single_attr.detach().cpu().numpy()[0] 
        spatial_attr_abs = np.mean(np.abs(attr_np), axis=(0, 1)) 

        # 3. Plotting 2 Panel (Lebar dikembalikan ke 5.5 inci agar pas untuk margin A4)
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
        
        # Panel 1: RGB
        axes[0].imshow(rgb_norm)
        axes[0].set_title(f"Sample Image", fontsize=10, fontweight='bold')
        axes[0].axis('off')
        
        # Panel 2: Overlay
        axes[1].imshow(gray_img, cmap='gray')
        im2 = axes[1].imshow(spatial_attr_abs, cmap='hot', interpolation='bilinear', alpha=0.5)
        axes[1].set_title('Overlay Heatmap', fontsize=10, fontweight='bold')
        axes[1].axis('off')
        
        # Colorbar untuk Overlay Heatmap
        cb = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        save_path = self.output_dir / 'spatial_attribution_enhanced.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Enhanced Spatial Attribution (2 Panel) tersimpan di: {save_path}")

    def save_results(self, band_imp: np.ndarray, temp_imp: np.ndarray, delta: float):
        results = {
            'band_importance': {b: float(s) for b, s in zip(self.band_names, band_imp)},
            'temporal_importance': {f'month_{i+1}': float(s) for i, s in enumerate(temp_imp)},
            'convergence_delta': float(delta),
        }
        with open(self.output_dir / 'interpretation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    def run_interpretation(self, data_loader: DataLoader, n_samples: int = 5, n_steps: int = 50):
        all_attributions, all_deltas = [], []
        first_sample_image, first_sample_attr = None, None
        
        for i, batch in enumerate(data_loader):
            if i >= n_samples: break
            
            image, batch_positions = batch['image'], batch['batch_positions']
            attributions, delta = self.compute_attributions(image, batch_positions, n_steps)
            
            if i == 0:
                first_sample_image = image.clone()
                first_sample_attr = attributions.clone()
            
            all_attributions.append(attributions)
            all_deltas.append(delta.item())
        
        avg_attributions = torch.mean(torch.stack(all_attributions), dim=0)
        avg_delta = np.mean(all_deltas)
        
        band_imp = self.analyze_band_importance(avg_attributions)
        temp_imp = self.analyze_temporal_importance(avg_attributions)
        self.analyze_spatial_patterns(avg_attributions)
        self.create_combined_heatmap(band_imp, temp_imp)
        self.save_results(band_imp, temp_imp, avg_delta)
        
        if first_sample_image is not None:
            self.analyze_single_spatial_sample(first_sample_attr, first_sample_image)

def load_model(model_path: str, config_path: str, device: torch.device) -> UTAE:
    with open(config_path) as f: config = json.load(f)
    model = UTAE(
        input_dim=config['input_dim'], output_dim=config['output_dim'],
        encoder_widths=config['encoder_widths'], decoder_widths=config['decoder_widths'],
        d_model=config['d_model'], n_head=config['n_head'], d_k=config.get('d_k', 4)
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='src/results/checkpoints/U-TAE/best_model.pt')
    parser.add_argument('--config-path', default='src/results/checkpoints/U-TAE/config.json')
    parser.add_argument('--data-dir', default='data/processed/lampung/version_2')
    parser.add_argument('--output-dir', default='src/results/interpretation')
    
    # Nilai optimal untuk efisiensi RTX 4090 dan validitas Skripsi
    parser.add_argument('--n-samples', type=int, default=200)
    parser.add_argument('--n-steps', type=int, default=100)
    
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, args.config_path, device)
    
    split_dir = os.path.join(root_dir, 'data', 'processed', 'lampung', 'splits')
    _, val_files = get_or_create_global_split(args.data_dir, split_dir, 0.2, args.seed)
    
    val_dataset = BiomassDataset(args.data_dir, mode='val', augment=False, file_list=val_files)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, 
                            collate_fn=collate_fn_biomass, num_workers=9, pin_memory=True)
    
    interpreter = ModelInterpreter(model, device, args.output_dir)
    interpreter.run_interpretation(val_loader, n_samples=args.n_samples, n_steps=args.n_steps)

if __name__ == "__main__":
    main()