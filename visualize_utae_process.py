"""
U-TAE Structure Visualization Pipeline - Final 2x6 Grid Edition
Perbaikan: Menggunakan MAX pooling pada Temporal Attention untuk melihat
sebaran fokus spasial model, menghindari "math trap" dari mean temporal.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import os
import sys
from matplotlib import rcParams

# ==========================================
# 1. SETUP & TYPOGRAPHY STANDAR JURNAL
# ==========================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.linewidth'] = 1.2

# Path Adjustment
current_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.models.utae import UTAE
from src.data.dataset import BiomassDataset, collate_fn_biomass

OUTPUT_DIR = 'src/results/visualizations/utae_structure_final'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. CONFIGURATION
# ==========================================
CONFIG = {
    'input_dim': 10, 
    'output_dim': 1,
    'encoder_widths': [64, 64, 128, 128],
    'decoder_widths': [32, 32, 64, 128],
    'd_model': 256, 
    'n_head': 4,
    'd_k': 4
}

MODEL_PATH = 'src/results/checkpoints/U-TAE/best_model.pt'
DATA_DIR = 'data/processed/lampung/version_2'

NORM_STATS = {
    'agb_log_mean': 4.145451545715332,
    'agb_log_std': 1.1578547954559326
}

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================
def get_rgb(img_np, t_idx=6):
    try:
        r, g, b = img_np[t_idx, 2, :, :], img_np[t_idx, 1, :, :], img_np[t_idx, 0, :, :]
        rgb = np.stack([r, g, b], axis=-1)
        p2, p98 = np.percentile(rgb, (2, 98))
        return np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
    except:
        return img_np[t_idx, 0, :, :]

# ==========================================
# 4. CORE ENGINE
# ==========================================
def export_final_pipeline():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("⏳ Initializing Model...")
    model = UTAE(**CONFIG).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    activations = {}
    
    def hook_fn(name):
        return lambda m, i, o: activations.update({name: o.detach()})

    def hook_ltae(m, i, o):
        activations['dec_d4'] = o[0].detach()

    # Sisi Kiri (Encoder)
    model.in_conv.conv.conv.register_forward_hook(hook_fn('enc_L1')) 
    model.down_block[0].conv2.conv.register_forward_hook(hook_fn('enc_L2')) 
    model.down_block[1].conv2.conv.register_forward_hook(hook_fn('enc_L3')) 
    model.down_block[2].conv2.conv.register_forward_hook(hook_fn('enc_L4_seq')) 
    
    # LTAE (Tengah Bawah)
    model.temporal_encoder.register_forward_hook(hook_ltae) 
    
    # Sisi Kanan (Decoder)
    model.up_blocks[0].conv2.conv.register_forward_hook(hook_fn('dec_d3')) 
    model.up_blocks[1].conv2.conv.register_forward_hook(hook_fn('dec_d2')) 
    model.up_blocks[2].conv2.conv.register_forward_hook(hook_fn('dec_d1')) 

    print("⏳ Loading Real Validation Dataset...")
    dataset = BiomassDataset(root_dir=DATA_DIR, mode='val', augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_biomass)

    batch = next(iter(loader))
    img, pos = batch['image'].to(DEVICE), batch['batch_positions'].to(DEVICE)
    gt_norm = batch['label'].numpy()[0] 

    print("🚀 Menjalankan Forward Pass...")
    with torch.no_grad():
        out = model(img, batch_positions=pos, denormalize=False)
        pred_norm = out['agb'].cpu().numpy().squeeze()
        attn = out.get('attn_weights', None) # Shape: (n_head, B, T, H, W)

    # --- DENORMALISASI ---
    pred_phys = np.expm1((pred_norm * NORM_STATS['agb_log_std']) + NORM_STATS['agb_log_mean'])
    gt_phys = np.expm1((gt_norm * NORM_STATS['agb_log_std']) + NORM_STATS['agb_log_mean'])
    
    vmax_val = max(np.percentile(gt_phys, 98), 100) 

    def to_2d_heatmap(tensor):
        dims_to_mean = tuple(range(tensor.ndim - 2))
        return tensor.mean(dim=dims_to_mean).cpu().numpy()

    # --- KOLEKSI 12 KOMPONEN ---
    components = {
        "01_Input_RGB": (get_rgb(img[0].cpu().numpy()), None),
        "02_Enc_L1": (to_2d_heatmap(activations['enc_L1']), 'viridis'),
        "03_Enc_L2": (to_2d_heatmap(activations['enc_L2']), 'viridis'),
        "04_Enc_L3": (to_2d_heatmap(activations['enc_L3']), 'viridis'),
        "05_Enc_L4_Bottleneck": (to_2d_heatmap(activations['enc_L4_seq']), 'inferno'), 
        # FIX: Menggunakan max(dim=2) untuk mencari waktu dengan atensi tertinggi di setiap piksel spasial
        "06_LTAE_Attention": (attn.max(dim=2)[0].mean(dim=0).squeeze().cpu().numpy(), 'hot'),
        "07_Dec_d4": (to_2d_heatmap(activations['dec_d4']), 'inferno'),
        "08_Dec_d3": (to_2d_heatmap(activations['dec_d3']), 'plasma'),
        "09_Dec_d2": (to_2d_heatmap(activations['dec_stage_2'] if 'dec_stage_2' in activations else activations['dec_d2']), 'plasma'),
        "10_Dec_d1": (to_2d_heatmap(activations['dec_d1']), 'magma'),
        "11_Output_AGB": (pred_phys, 'YlGn'),
        "12_Ground_Truth": (gt_phys, 'YlGn')
    }

    # ==========================================
    # KAMUS JUDUL KHUSUS: DIMENSI TENSOR LENGKAP
    # ==========================================
    custom_titles = {
        "01_Input_RGB": "Input Tensor\n(12 x 10 x 128 x 128)",
        "02_Enc_L1": "Encoder L1\n(12 x 64 x 128 x 128)",
        "03_Enc_L2": "Encoder L2\n(12 x 64 x 64 x 64)",
        "04_Enc_L3": "Encoder L3\n(12 x 128 x 32 x 32)",
        "05_Enc_L4_Bottleneck": "Encoder L4 Seq\n(12 x 128 x 16 x 16)",
        "06_LTAE_Attention": "Attention Masks\n(4 x 12 x 16 x 16)", 
        "07_Dec_d4": "Decoder d4\n(128 x 16 x 16)",      
        "08_Dec_d3": "Decoder d3\n(64 x 32 x 32)",
        "09_Dec_d2": "Decoder d2\n(32 x 64 x 64)",
        "10_Dec_d1": "Decoder d1\n(32 x 128 x 128)",
        "11_Output_AGB": "Predicted AGB\n(1 x 128 x 128)",
        "12_Ground_Truth": "Ground Truth\n(1 x 128 x 128)"
    }

    # ==========================================
    # EXPORT 1: GAMBAR INDIVIDUAL (TRANSPARAN)
    # ==========================================
    print(f"🎨 Mengekspor 12 komponen individual ke: {OUTPUT_DIR}")
    for name, (data, cmap) in components.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        
        if "Attention" in name:
            vmin, vmax = None, None
        else:
            vmin = None
            vmax = vmax_val if "AGB" in name or "Truth" in name else np.percentile(data, 99.5)
            
        ax.imshow(data, cmap=cmap, interpolation='antialiased', vmin=vmin, vmax=vmax)
        ax.axis('off')
        
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), 
                    dpi=900, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        print(f"   ✅ {name}.png tersimpan.")

    # ==========================================
    # EXPORT 2: GAMBAR RANGKUMAN (GRID 2x6 UNTUK KERTAS A4)
    # ==========================================
    print("⭐ Membuat Gambar Pipeline Berurutan (Grid 2x6 dirapatkan)...")
    
    fig_sum, axes = plt.subplots(2, 6, figsize=(15, 4.5))
    plt.subplots_adjust(top=0.85, bottom=0.05, wspace=0.35, hspace=0.15) 
    
    sorted_keys = sorted(components.keys())
    
    for i, ax in enumerate(axes.flat):
        if i < len(sorted_keys):
            k = sorted_keys[i]
            d, cm = components[k]
            
            if "Attention" in k:
                vmin, vmax = None, None
            else:
                vmin = None
                vmax = vmax_val if "AGB" in k or "Truth" in k else np.percentile(d, 99.5)
                
            im = ax.imshow(d, cmap=cm, interpolation='antialiased', vmin=vmin, vmax=vmax)
            
            ax.set_title(custom_titles[k], fontsize=10, fontweight='normal', family='serif', pad=10)
            ax.axis('off')
            
            if cm is not None:
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, location='right')
                cb.ax.tick_params(labelsize=8) 
                
                # Mematikan offset notasi ilmiah
                cb.formatter.set_useOffset(False)
                cb.formatter.set_scientific(False)
                cb.update_ticks()
                
                if "AGB" in k or "Truth" in k:
                    cb.set_label('Mg/ha', size=8, family='serif')
        else:
            ax.axis('off')

    plt.suptitle("U-TAE Architecture Processing Pipeline: Tensor Dimensions Flow", 
                 fontsize=16, fontweight='bold', family='serif', y=0.98)
    
    summary_path = os.path.join(OUTPUT_DIR, "0_TOTAL_PIPELINE_A4_OPTIMIZED.png")
    
    plt.savefig(summary_path, dpi=900, bbox_inches='tight', facecolor='white')
    print(f"✅ Selesai! Gambar rangkuman tersimpan di: {summary_path}")

if __name__ == "__main__":
    export_final_pipeline()