"""
Model Interpretation using Captum (Integrated Gradients).

Analyzes the influence of:
1. Sentinel-2 spectral bands
2. Temporal dimensions (months)
3. Spatial patterns

on AGB predictions using the trained U-TAE model.

References:
- Sundararajan et al. (2017): Axiomatic Attribution for Deep Networks
- Captum Documentation: https://captum.ai/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from captum.attr import IntegratedGradients
from tqdm import tqdm

# Setup paths
current_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import project modules
from src.models.utae import UTAE
from src.data.dataset import BiomassDataset, collate_fn_biomass, get_or_create_global_split

class ModelInterpreter:
    """
    Interpreter for U-TAE model using Integrated Gradients.
    
    Analyzes feature importance across:
    - Spectral bands (10 Sentinel-2 bands)
    - Temporal steps (12 months)
    - Spatial patterns
    """
    def __init__(
        self,
        model: UTAE,
        device: torch.device,
        output_dir: str = 'src/results/interpretation'
    ):
        """
        Initialize interpreter.
        
        Args:
            model: Trained U-TAE model
            device: torch.device for computation
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sentinel-2 band names
        self.band_names = ['B2', 'B3', 'B4', 'B5', 'B6', 
                           'B7', 'B8', 'B8A', 'B11', 'B12']

        # Month names
        self.month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        

        
    def _forward_func(self, inputs: torch.Tensor, batch_positions: torch.Tensor):
        """
        Wrapper function for Captum.
        
        Args:
            inputs: Input tensor (B, T, C, H, W)
            batch_positions: Temporal positions (B, T)
            
        Returns:
            AGB predictions (B, H, W)
        """
        # Create closure to capture batch_positions
        def forward(x):
            return self.model(x, batch_positions=batch_positions)['agb']
        return forward
    
    def compute_attributions(
        self,
        input_sample: torch.Tensor,
        batch_positions: torch.Tensor,
        n_steps: int = 50
    ) -> tuple:
        """
        Compute integrated gradients attributions.
        
        Args:
            input_sample: Input tensor (B, T, C, H, W)
            batch_positions: Temporal positions (B, T)
            n_steps: Number of integration steps (higher = more accurate)
            
        Returns:
            (attributions, convergence_delta)
        """
        print(f"Input shape: {input_sample.shape}")
        print(f"Integration steps: {n_steps}")

        # Prepare input
        input_sample = input_sample.to(self.device).requires_grad_()
        batch_positions = batch_positions.to(self.device)
        
        # Create forward function
        forward_func = self._forward_func(input_sample, batch_positions)
        
        # Initialize Integrated Gradients
        ig = IntegratedGradients(forward_func)
        
        # Compute attributions
        print("Computing attributions...")
        attributions, delta = ig.attribute(
            input_sample,
            n_steps=n_steps,
            return_convergence_delta=True
        )
        
        print(f"Attributions computed")
        print(f"- Convergence delta: {delta.item():.6f}")
        
        if abs(delta.item()) > 0.1:
            print(f"Warning: Large delta! Consider increasing n_steps.")
        
        return attributions, delta
    
    def analyze_band_importance(
        self,
        attributions: torch.Tensor,
        save_plot: bool = True
    ) -> np.ndarray:
        """
        Analyze spectral band importance.
        
        Args:
            attributions: Attribution tensor (B, T, C, H, W)
            save_plot: Whether to save visualization
            
        Returns:
            band_importance: Array of shape (C,)
        """
        print(f"\n{'='*80}")
        print("BAND IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Convert to numpy
        attr_np = attributions.detach().cpu().numpy()
        
        # Aggregate over batch, time, height, width
        band_importance = np.mean(np.abs(attr_np), axis=(0, 1, 3, 4))
        
        # Normalize to sum to 1
        band_importance_norm = band_importance / np.sum(band_importance)
        
        # Print results
        print("\nBand Importance Scores:")
        for i, (band, score) in enumerate(zip(self.band_names, band_importance_norm)):
            print(f"  {band:5s}: {score:.4f} ({score*100:.2f}%)")
        
        # Identify most important bands
        top_3_idx = np.argsort(band_importance_norm)[-3:][::-1]
        print(f"\nTop 3 Most Important Bands:")
        for rank, idx in enumerate(top_3_idx, 1):
            print(f"  {rank}. {self.band_names[idx]:5s}: {band_importance_norm[idx]*100:.2f}%")
        
        # Visualization
        if save_plot:
            self._plot_band_importance(band_importance_norm)
        
        return band_importance_norm
    
    def _plot_band_importance(self, band_importance: np.ndarray):
        """Plot band importance."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bars with color gradient
        colors = plt.cm.viridis(band_importance / band_importance.max())
        bars = ax.bar(self.band_names, band_importance, color=colors, 
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, band_importance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.3f}\n({val*100:.1f}%)',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title('Spectral Band Importance for AGB Estimation', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Sentinel-2 Bands', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative Importance Score', fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(0, band_importance.max() * 1.15)
        
        plt.tight_layout()
        save_path = self.output_dir / 'band_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nBand importance plot saved: {save_path}")

    
    def analyze_temporal_importance(
        self,
        attributions: torch.Tensor,
        save_plot: bool = True
    ) -> np.ndarray:
        """
        Analyze temporal importance.
        
        Args:
            attributions: Attribution tensor (B, T, C, H, W)
            save_plot: Whether to save visualization
            
        Returns:
            temporal_importance: Array of shape (T,)
        """
        
        # Convert to numpy
        attr_np = attributions.detach().cpu().numpy()
        
        # Aggregate over batch, channel, height, width
        temporal_importance = np.mean(np.abs(attr_np), axis=(0, 2, 3, 4))
        
        # Normalize
        temporal_importance_norm = temporal_importance / np.sum(temporal_importance)
        
        # Print results
        print("\nTemporal Importance Scores:")
        for i, (month, score) in enumerate(zip(self.month_names, temporal_importance_norm), 1):
            print(f"  Month {i:2d} ({month}): {score:.4f} ({score*100:.2f}%)")
        
        # Identify most important months
        top_3_idx = np.argsort(temporal_importance_norm)[-3:][::-1]
        print(f"\nTop 3 Most Important Months:")
        for rank, idx in enumerate(top_3_idx, 1):
            print(f"  {rank}. Month {idx+1:2d} ({self.month_names[idx]}): "
                  f"{temporal_importance_norm[idx]*100:.2f}%")
        
        # Visualization
        if save_plot:
            self._plot_temporal_importance(temporal_importance_norm)
        
        return temporal_importance_norm
    

    def _plot_temporal_importance(self, temporal_importance: np.ndarray):
        """Plot temporal importance."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        months = np.arange(1, 13)
        
        # Line plot with markers
        ax.plot(months, temporal_importance, marker='o', linewidth=2.5,
               markersize=8, color='coral', label='Importance Score')
        
        # Fill area under curve
        ax.fill_between(months, temporal_importance, alpha=0.3, color='coral')
        
        # Add value labels
        for x, y in zip(months, temporal_importance):
            ax.text(x, y + 0.001, f'{y:.3f}', ha='center', va='bottom',
                   fontsize=8, fontweight='bold')
        
        ax.set_title('Temporal Importance Across 12 Months', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative Importance Score', fontsize=12, fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(self.month_names, rotation=45)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        save_path = self.output_dir / 'temporal_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nTemporal importance plot saved: {save_path}")


    def analyze_spatial_patterns(
        self,
        attributions: torch.Tensor,
        save_plot: bool = True
    ):
        """
        Analyze spatial attribution patterns.
        
        Args:
            attributions: Attribution tensor (B, T, C, H, W)
            save_plot: Whether to save visualization
        """
        
        # Convert to numpy
        attr_np = attributions.detach().cpu().numpy()
        
        # Aggregate over batch, time, and channel
        spatial_attr = np.mean(np.abs(attr_np), axis=(0, 1, 2))
        
        print(f"Spatial attribution map shape: {spatial_attr.shape}")
        print(f"Attribution range: [{spatial_attr.min():.6f}, {spatial_attr.max():.6f}]")
        
        # Visualization
        if save_plot:
            self._plot_spatial_attribution(spatial_attr)

    
    def _plot_spatial_attribution(self, spatial_attr: np.ndarray):
        """Plot spatial attribution map."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Heatmap
        im1 = axes[0].imshow(spatial_attr, cmap='hot', interpolation='bilinear')
        axes[0].set_title('Spatial Attribution Heatmap', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Plot 2: Contour plot
        axes[1].contourf(spatial_attr, levels=20, cmap='hot')
        axes[1].set_title('Spatial Attribution Contours', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / 'spatial_attribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSpatial attribution plot saved: {save_path}")

    
    def create_combined_heatmap(
        self,
        band_importance: np.ndarray,
        temporal_importance: np.ndarray
    ):
        """Create combined band-temporal heatmap."""
        print(f"\n{'='*80}")
        print("CREATING COMBINED VISUALIZATION")
        print("="*80)
        
        # Create 2D heatmap: bands × months
        # This shows which band-month combinations are most important
        heatmap_data = np.outer(band_importance, temporal_importance)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(
            heatmap_data,
            xticklabels=self.month_names,
            yticklabels=self.band_names,
            cmap='YlOrRd',
            annot=True,
            fmt='.4f',
            cbar_kws={'label': 'Importance Score'},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title('Band-Temporal Importance Heatmap', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spectral Band', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / 'combined_importance_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined heatmap saved: {save_path}")



    def save_results(
        self,
        band_importance: np.ndarray,
        temporal_importance: np.ndarray,
        convergence_delta: float
    ):
        """Save numerical results to JSON."""
        results = {
            'band_importance': {
                band: float(score)
                for band, score in zip(self.band_names, band_importance)
            },
            'temporal_importance': {
                f'month_{i+1}': float(score)
                for i, score in enumerate(temporal_importance)
            },
            'convergence_delta': float(convergence_delta),
            'metadata': {
                'method': 'Integrated Gradients',
                'n_bands': len(self.band_names),
                'n_timesteps': len(temporal_importance)
            }
        }
        
        save_path = self.output_dir / 'interpretation_results.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nNumerical results saved: {save_path}")



    def run_interpretation(
        self,
        data_loader: DataLoader,
        n_samples: int = 5,
        n_steps: int = 50
    ):
        """
        Run complete interpretation pipeline.
        
        Args:
            data_loader: DataLoader with samples to analyze
            n_samples: Number of samples to analyze (will be averaged)
            n_steps: Number of integration steps
        """
        print(f"Analyzing {n_samples} samples")
        print(f"Integration steps: {n_steps}")
        print("="*80)
        
        all_attributions = []
        all_deltas = []
        
        # Analyze multiple samples
        for i, batch in enumerate(data_loader):
            if i >= n_samples:
                break
            
            print(f"\n--- Sample {i+1}/{n_samples} ---")
            
            image = batch['image']
            batch_positions = batch['batch_positions']
            
            # Compute attributions
            attributions, delta = self.compute_attributions(
                image, batch_positions, n_steps
            )
            
            all_attributions.append(attributions)
            all_deltas.append(delta.item())
        
        # Average attributions across samples
        avg_attributions = torch.mean(torch.stack(all_attributions), dim=0)
        avg_delta = np.mean(all_deltas)
        
        print(f"AVERAGED RESULTS ({n_samples} samples)")
        print(f"Average convergence delta: {avg_delta:.6f}")
        
        # Run analyses
        band_importance = self.analyze_band_importance(avg_attributions)
        temporal_importance = self.analyze_temporal_importance(avg_attributions)
        self.analyze_spatial_patterns(avg_attributions)
        self.create_combined_heatmap(band_importance, temporal_importance)
        self.save_results(band_importance, temporal_importance, avg_delta)
        
        print(f"All results saved to: {self.output_dir}")


def load_model(model_path: str, config_path: str, device: torch.device) -> UTAE:
    """Load trained U-TAE model."""
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"Config loaded from: {config_path}")
    
    # Initialize model
    model = UTAE(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        encoder_widths=config['encoder_widths'],
        decoder_widths=config['decoder_widths'],
        d_model=config['d_model'],
        n_head=config['n_head'],
        d_k=config.get('d_k', 4)
    ).to(device)
    
    # Load weights
    print(f"Loading weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully")
    print(f"Parameters: {n_params:,}")
    
    return model    


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interpret U-TAE model using Integrated Gradients"
    )
    parser.add_argument(
        '--model-path',
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        '--config-path',
        required=True,
        help="Path to model config JSON"
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help="Path to preprocessed data directory"
    )
    parser.add_argument(
        '--output-dir',
        default='src/results/interpretation',
        help="Output directory for results"
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5,
        help="Number of samples to analyze (will be averaged)"
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=50,
        help="Number of integration steps (higher = more accurate)"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help="Batch size for data loading"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, args.config_path, device)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA (Validation Set Only)")
    print("="*80)
    
    # Lokasi direktori split
    split_dir = os.path.join(root_dir, 'data', 'processed', 'lampung', 'splits')
    
    # Panggil fungsi global split (akan membaca file JSON yang sudah ada)
    _, val_files = get_or_create_global_split(
        data_dir=args.data_dir,
        split_dir=split_dir,
        test_size=0.2,
        random_state=args.seed
    )
    
    # Muat HANYA data validasi
    val_dataset = BiomassDataset(
        root_dir=args.data_dir, 
        mode='val', 
        augment=False, 
        file_list=val_files
    )
    
    # DataLoader (Boleh shuffle=True di sini agar sampel yang dianalisis selalu bervariasi tiap kali run)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True, 
        collate_fn=collate_fn_biomass,
        num_workers=0
    )
    
    print(f"Dataset loaded (Validation Only):")
    print(f"- Total Validation Patches: {len(val_dataset)}")
    print(f"- Will analyze: {args.n_samples} samples")

    
    # Run interpretation
    interpreter = ModelInterpreter(model, device, args.output_dir)
    interpreter.run_interpretation(
        val_loader,
        n_samples=args.n_samples,
        n_steps=args.n_steps
    )
    
    print("\nInterpretation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

