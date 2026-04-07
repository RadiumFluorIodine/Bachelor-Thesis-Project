"""
Fine-tuning (Transfer Learning) U-TAE Model for Kalimantan Selatan.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Setup Path
current_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.models.utae import UTAE
from src.data.dataset import BiomassDataset, collate_fn_biomass, get_or_create_global_split
from src.training.training_utils import BiomassRegressionLoss, RegressionMetrics

def plot_finetune_results(history: dict, best_preds: np.ndarray, best_labels: np.ndarray, output_dir: str):
    """Membuat 3 grafik untuk laporan skripsi (Loss Curve, R2 Curve, Scatter Plot)."""
    print("\n🎨 Membuat grafik hasil Fine-Tuning untuk laporan skripsi...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Learning Curve (Loss)
    axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss (MSE)')
    axes[0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss (MSE)')
    axes[0].set_title('Learning Curve (Fine-Tuning Kalsel)', fontweight='bold')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: R-Squared Progression
    axes[1].plot(epochs_range, history['val_r2'], 'g-', linewidth=2, label='Validation R²')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title('Perkembangan Akurasi (R²)', fontweight='bold')
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('R-Squared', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Scatter Plot 1:1 (Best Epoch)
    axes[2].scatter(best_labels, best_preds, alpha=0.5, color='forestgreen', s=10)
    max_val = max(best_labels.max(), best_preds.max())
    axes[2].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 Ideal')
    
    if len(best_labels) > 1:
        z = np.polyfit(best_labels, best_preds, 1)
        p = np.poly1d(z)
        axes[2].plot([0, max_val], p([0, max_val]), 'b-', linewidth=2, label='Regresi Model')
    
    axes[2].set_title('Prediksi vs Aktual (Model Terbaik)', fontweight='bold')
    axes[2].set_xlabel('Aktual AGB (Mg/ha)', fontweight='bold')
    axes[2].set_ylabel('Prediksi AGB (Mg/ha)', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = Path(output_dir) / 'finetune_kalsel_plots.png'
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Grafik tersimpan di: {plot_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Memulai Fine-Tuning di perangkat: {device}")

    # --- KONFIGURASI PATH ---
    pretrained_model_path = "src/results/checkpoints/U-TAE/best_model.pt"
    config_path = "src/results/checkpoints/U-TAE/config.json"
    kalsel_data_dir = "data/processed/kalsel"
    kalsel_norm_path = "data/processed/kalsel/normalization.json"
    kalsel_split_dir = "data/processed/kalsel/splits"
    output_dir = "src/results/checkpoints/U-TAE_Finetuned_Kalsel"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load Normalization Stats Kalsel (untuk denormalisasi metrik)
    with open(kalsel_norm_path) as f:
        norm_stats = json.load(f)
    agb_log_mean = norm_stats.get('agb_log_mean', 4.145451)
    agb_log_std = norm_stats.get('agb_log_std', 1.157854)

    # 1. LOAD PRE-TRAINED MODEL (MODEL LAMPUNG)
    with open(config_path) as f:
        config = json.load(f)

    model = UTAE(
        input_dim=config['input_dim'], output_dim=config['output_dim'],
        encoder_widths=config['encoder_widths'], decoder_widths=config['decoder_widths'],
        d_model=config['d_model'], n_head=config['n_head']
    ).to(device)
    
    checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    print("✅ Pre-trained model (Lampung) berhasil dimuat.")

    # 2. LOAD DATASET KALIMANTAN SELATAN (GLOBAL SPLIT)
    print("\n📦 Menyiapkan Global Split untuk Kalsel...")
    train_files, val_files = get_or_create_global_split(
        data_dir=kalsel_data_dir, 
        split_dir=kalsel_split_dir, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"📊 Total Data Kalsel: {len(train_files) + len(val_files)} patches")
    print(f"   -> Data Training   : {len(train_files)} patches")
    print(f"   -> Data Validation : {len(val_files)} patches")

    train_dataset = BiomassDataset(root_dir=kalsel_data_dir, mode='train', augment=True, file_list=train_files)
    val_dataset = BiomassDataset(root_dir=kalsel_data_dir, mode='val', augment=False, file_list=val_files)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_biomass, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_biomass, num_workers=4, pin_memory=True, persistent_workers=True)

    # 3. SETUP TRAINING DENGAN EARLY STOPPING (FULL PRECISION FP32)
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = BiomassRegressionLoss(smoothness_weight=config.get('smoothness_weight', 0.1))
    
    epochs = 100
    patience = 10
    patience_counter = 0
    best_val_loss = float('inf')
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'val_mae': []}
    best_preds_raw = None
    best_labels_raw = None

    print("\n🔄 Memulai Proses Fine-Tuning (Transfer Learning)...")
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch [{epoch+1:03d}/{epochs}] Training"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            pos = batch['batch_positions'].to(device)
            
            if pos.dim() == 1:
                pos = pos.unsqueeze(0).expand(images.size(0), -1)
            
            optimizer.zero_grad()

            # Forward pass dalam Full Precision (FP32)
            preds = model(images, batch_positions=pos)['agb']
            
            if 'valid_mask' in batch:
                mask = batch['valid_mask'].bool().to(device)
                # Trik Masking: Ubah tebakan di area awan/laut sama dengan label agar error = 0
                preds_loss = torch.where(mask, preds, labels)
                loss = criterion(preds_loss, labels)
            else:
                loss = criterion(preds, labels)
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch [{epoch+1:03d}/{epochs}] Validation"):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                pos = batch['batch_positions'].to(device)
                
                if pos.dim() == 1:
                    pos = pos.unsqueeze(0).expand(images.size(0), -1)
                    
                # Forward pass dalam Full Precision (FP32)
                preds = model(images, batch_positions=pos)['agb']
                
                if 'valid_mask' in batch:
                    mask = batch['valid_mask'].bool().to(device)
                    preds_loss = torch.where(mask, preds, labels)
                    loss = criterion(preds_loss, labels)
                    
                    # Flatten HANYA UNTUK HITUNG R2 dan MAE
                    p = preds.view(-1)[mask.view(-1)]
                    l = labels.view(-1)[mask.view(-1)]
                else:
                    loss = criterion(preds, labels)
                    p = preds.flatten()
                    l = labels.flatten()
                    
                val_loss += loss.item()
                
                # Denormalisasi untuk metrik
                p_raw = torch.expm1((p * agb_log_std) + agb_log_mean).cpu().numpy()
                l_raw = torch.expm1((l * agb_log_std) + agb_log_mean).cpu().numpy()
                all_preds.append(np.clip(p_raw, 0, None))
                all_targets.append(np.clip(l_raw, 0, None))

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        predictions = np.concatenate(all_preds).flatten()
        targets = np.concatenate(all_targets).flatten()
        metrics = RegressionMetrics.compute_all(predictions, targets)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(metrics['r2'])
        history['val_mae'].append(metrics['mae'])
        
        print(f"--> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val R²: {metrics['r2']:.4f} | Val MAE: {metrics['mae']:.2f}")

        # --- EARLY STOPPING LOGIC ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 
            best_preds_raw = predictions
            best_labels_raw = targets
            
            torch.save(model.state_dict(), f"{output_dir}/best_finetuned_model.pt")
            
            with open(f"{output_dir}/config.json", 'w') as f:
                json.dump(config, f)
                
            print("   🌟 Model membaik! Bobot terbaik disimpan.")
        else:
            patience_counter += 1
            print(f"   ⚠️ Val Loss tidak membaik. Early Stopping Counter: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\n🛑 EARLY STOPPING DIPICU! Model berhenti belajar karena Val Loss tidak membaik selama {patience} epoch berturut-turut.")
            break

    # Buat Plot
    if best_preds_raw is not None:
        plot_finetune_results(history, best_preds_raw, best_labels_raw, output_dir)
        
    print("\n✅ Fine-Tuning Selesai! Model siap digunakan untuk inferensi di Kalimantan Selatan.")

if __name__ == '__main__':
    main()