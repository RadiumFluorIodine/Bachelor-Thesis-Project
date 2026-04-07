import os
import sys
import torch
import torch.nn.functional as F
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.models.utae import UTAE
from src.data.dataset import BiomassDataset, collate_fn_biomass
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan perangkat: {device}")

    # Konfigurasi Path
    model_path = "src/results/checkpoints/U-TAE/best_model.pt"
    config_path = "src/results/checkpoints/U-TAE/config.json"
    data_dir = "data/processed/lampung/version_2"
    output_dir = "src/results/attention_analysis"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load Config
    with open(config_path) as f:
        config = json.load(f)

    # Model Initialisation
    print("\n[1] Membangun Arsitektur Model...")
    model = UTAE(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        encoder_widths=config['encoder_widths'],
        decoder_widths=config['decoder_widths'],
        d_model=config['d_model'],
        n_head=config['n_head'],
        d_k=config.get('d_k', 4)
    ).to(device)

    # Load Weights 
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Model Summary 
    print("\n[2] Menampilkan Ringkasan Model:")
    model.print_model_summary()

    # Load Data Sampel
    print("\n[3] Memuat Data Sampel dari Lampung...")
    dataset = BiomassDataset(root_dir=data_dir, mode='val')
    # Ambil 1 batch berisi 
    loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_fn_biomass)
    batch = next(iter(loader))

    images = batch['image'].to(device)
    positions = batch['batch_positions'].to(device)

    if positions.dim() == 1:
                positions = positions.unsqueeze(0).expand(images.size(0), -1)

    
    # Run the Forward Pass to extract Attention
    print("\n[4] Mengekstrak Bobot Attention (Forward Pass)...")
    with torch.no_grad():
        output = model(images, batch_positions=positions)
        attn_weights = output['attn_weights'] 
        
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Konfigurasi Index Sampel dan Waktu
    target_sample_idx = 0
    target_timestep_idx = 1

    # Temporal Attention Visualisation
    temporal_out_path = f"{output_dir}/temporal_attention_raw.png"
    print(f"\n[5] Menyimpan Grafik Temporal Attention ke {temporal_out_path}")
    model.visualize_attention(
        attn_weights, 
        timestep_labels=months, 
        save_path=temporal_out_path
    )

    print(f"\n[6] Membuat Peta Komparasi RGB Asli vs Spatial Attention (Bulan {months[target_timestep_idx]})...")
    try:
        # Ekstrak RGB
        rgb_tensor = images[target_sample_idx, target_timestep_idx, [2, 1, 0], :, :].cpu()
        rgb_img = rgb_tensor.permute(1, 2, 0).numpy()
        
        # Lakukan Contrast Stretching agar gambar satelit terlihat cerah
        p2, p98 = np.percentile(rgb_img, (2, 98))
        rgb_norm = np.clip((rgb_img - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Ekstrak Heatmap Langsung dari Tensor attn_weights
        sample_attn = attn_weights[target_sample_idx].cpu()
        
        # Menangani berbagai kemungkinan dimensi output atensi
        if sample_attn.dim() == 4: # Format: (Heads, T, H, W)
            spatial_attn = sample_attn[:, target_timestep_idx, :, :].mean(dim=0)
        elif sample_attn.dim() == 3: # Format: (T, H, W)
            spatial_attn = sample_attn[target_timestep_idx]
        else: 
            spatial_attn = sample_attn[target_timestep_idx].squeeze()

        # Konversi ke float untuk keperluan interpolasi
        spatial_attn = spatial_attn.float()

        # Pada L-TAE, ukuran atensi spasial biasanya lebih kecil (misal 8x8).
        H_asli, W_asli = rgb_norm.shape[:2]
        if spatial_attn.shape != (H_asli, W_asli):
            # Tambahkan dimensi batch dan channel untuk proses interpolasi: (1, 1, H, W)
            spatial_attn_tensor = spatial_attn.unsqueeze(0).unsqueeze(0)
            spatial_attn = F.interpolate(
                spatial_attn_tensor, 
                size=(H_asli, W_asli), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()

        # Konversi hasil akhir kembali ke bentuk Numpy array untuk Matplotlib
        spatial_attn_np = spatial_attn.numpy()

        # 3. Plot secara berdampingan
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot RGB
        axes[0].imshow(rgb_norm)
        axes[0].set_title(f"Citra Satelit RGB - {months[target_timestep_idx]}", fontsize=14, pad=10, fontweight='bold')
        axes[0].axis('off')
        
        # Plot Heatmap dari array secara langsung
        im = axes[1].imshow(spatial_attn_np, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f"Spatial Attention Heatmap - {months[target_timestep_idx]}", fontsize=14, pad=10, fontweight='bold')
        axes[1].axis('off')
        
        # Tambahkan Colorbar sebagai legenda intensitas Atensi
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Attention Weight')
        
        comparison_out_path = f"{output_dir}/rgb_vs_attention_comparison_{months[target_timestep_idx]}.png"
        plt.tight_layout()
        plt.savefig(comparison_out_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"Berhasil! Peta perbandingan tersimpan di: {comparison_out_path}")
        
    except Exception as e:
        print(f"Gagal membuat perbandingan RGB. Error: {e}")

    print("\nEksekusi Selesai! Silakan cek folder", output_dir)

if __name__ == '__main__':
    main()