"""
Fine-tuning (Transfer Learning) U-TAE Model for Kalimantan Selatan.
Dilengkapi dengan Fitur Resume (Bisa dicicil), Early Stopping, Progress Bar, Plotting, dan Logging.
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
import logging
from datetime import datetime

current_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.models.utae import UTAE
from src.data.dataset import BiomassDataset, collate_fn_biomass, get_or_create_global_split
from src.training.training_utils import RegressionMetrics

def setup_logger(output_dir: str):
    """Menyiapkan logger untuk menyimpan log ke file dan menampilkan di console."""
    log_file = os.path.join(output_dir, "finetune_kalsel.log")
    
    # Format log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Buat logger
    logger = logging.getLogger('FinetuneLogger')
    logger.setLevel(logging.INFO)
    
    # Hindari duplikasi handler jika fungsi dipanggil ulang
    if not logger.handlers:
        # File Handler 
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console Handler (tampilkan di terminal)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

def plot_finetune_results(history: dict, best_preds: np.ndarray, best_labels: np.ndarray, output_dir: str, logger: logging.Logger):
    """Membuat 3 grafik untuk laporan skripsi (Loss Curve, R2 Curve, Scatter Plot)."""
    logger.info("🎨 Membuat grafik hasil Fine-Tuning untuk laporan skripsi...")
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
    
    axes[2].set_title('Prediksi vs Aktual', fontweight='bold')
    axes[2].set_xlabel('Aktual AGB (Mg/ha)', fontweight='bold')
    axes[2].set_ylabel('Prediksi AGB (Mg/ha)', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = Path(output_dir) / 'finetune_kalsel_plots.png'
    plt.savefig(plot_path, dpi=300)
    logger.info(f"✅ Grafik tersimpan di: {plot_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- KONFIGURASI PATH ---
    pretrained_model_path = "src/results/checkpoints/U-TAE/best_model.pt"
    config_path = "src/results/checkpoints/U-TAE/config.json"
    kalsel_data_dir = "data/processed/kalsel"
    kalsel_norm_path = "data/processed/kalsel/normalization.json"
    kalsel_split_dir = "data/processed/kalsel/splits"
    output_dir = "src/results/checkpoints/U-TAE_Finetuned_Kalsel"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup Logger
    logger = setup_logger(output_dir)
    logger.info(f"🚀 Memulai Fine-Tuning di perangkat: {device}")
    
    # Path untuk file penyambung (Resume Checkpoint)
    resume_checkpoint_path = os.path.join(output_dir, "resume_checkpoint.pt")

    # Load Normalization Stats Kalsel
    try:
        with open(kalsel_norm_path) as f:
            norm_stats = json.load(f)
        agb_log_mean = norm_stats.get('agb_log_mean', 4.145451)
        agb_log_std = norm_stats.get('agb_log_std', 1.157854)
        logger.info("✅ Berhasil memuat statistik normalisasi Kalsel.")
    except Exception as e:
        logger.error(f"Gagal memuat statistik normalisasi: {e}")
        return

    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Gagal memuat konfigurasi model: {e}")
        return

    # Inisialisasi Model Kosong
    model = UTAE(
        input_dim=config['input_dim'], output_dim=config['output_dim'],
        encoder_widths=config['encoder_widths'], decoder_widths=config['decoder_widths'],
        d_model=config['d_model'], n_head=config['n_head']
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    epochs = 100
    patience = 10
    
    # --- LOGIKA RESUME TRAINING ---
    if os.path.exists(resume_checkpoint_path):
        logger.info("♻️ MENEMUKAN CHECKPOINT! Melanjutkan proses Fine-Tuning yang tertunda...")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        history = checkpoint['history']
        
        logger.info(f"✅ Melanjutkan dari Epoch {start_epoch+1}. Best Val Loss sebelumnya: {best_val_loss:.4f}")
    else:
        logger.info("🆕 Memulai Fine-Tuning BARU dari awal (Menggunakan Model Lampung)...")
        try:
            checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
            logger.info("✅ Pre-trained model (Lampung) berhasil dimuat.")
        except Exception as e:
            logger.error(f"Gagal memuat Pre-trained model: {e}")
            return
            
        start_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'val_mae': []}

    # Data Loader Kalsel
    logger.info("📦 Menyiapkan Global Split untuk Kalsel...")
    train_files, val_files = get_or_create_global_split(
        data_dir=kalsel_data_dir, split_dir=kalsel_split_dir, test_size=0.2, random_state=42
    )
    
    logger.info(f"📊 Total Data Kalsel: {len(train_files) + len(val_files)} patches")
    logger.info(f"   -> Data Training   : {len(train_files)} patches")
    logger.info(f"   -> Data Validation : {len(val_files)} patches")
    
    train_dataset = BiomassDataset(root_dir=kalsel_data_dir, mode='train', augment=True, file_list=train_files)
    val_dataset = BiomassDataset(root_dir=kalsel_data_dir, mode='val', augment=False, file_list=val_files)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_biomass, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_biomass, num_workers=0)

    best_preds_raw, best_labels_raw = None, None

    logger.info("🔄 Memulai Proses Fine-Tuning...")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        
        # tqdm digabungkan dengan logging hanya pada awal/akhir epoch agar log tidak kotor
        for batch in tqdm(train_loader, desc=f"Epoch [{epoch+1:03d}/{epochs}] Training", leave=False, ncols=100, file=sys.stdout):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            pos = batch['batch_positions'].to(device)
            
            if pos.dim() == 1:
                pos = pos.unsqueeze(0).expand(images.size(0), -1)
            
            optimizer.zero_grad()
            preds = model(images, batch_positions=pos)['agb']
            
            if 'valid_mask' in batch:
                mask = batch['valid_mask'].bool().to(device).view(-1)
                loss = criterion(preds.view(-1)[mask], labels.view(-1)[mask])
            else:
                loss = criterion(preds.flatten(), labels.flatten())
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch [{epoch+1:03d}/{epochs}] Validation", leave=False, ncols=100, file=sys.stdout):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                pos = batch['batch_positions'].to(device)
                
                if pos.dim() == 1:
                    pos = pos.unsqueeze(0).expand(images.size(0), -1)
                    
                preds = model(images, batch_positions=pos)['agb']
                
                if 'valid_mask' in batch:
                    mask = batch['valid_mask'].bool().to(device).view(-1)
                    p, l = preds.view(-1)[mask], labels.view(-1)[mask]
                else:
                    p, l = preds.flatten(), labels.flatten()
                    
                val_loss += criterion(p, l).item()
                
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
        
        logger.info(f"Epoch [{epoch+1:03d}/{epochs}] -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val R²: {metrics['r2']:.4f} | Val MAE: {metrics['mae']:.2f}")

        # --- UPDATE EARLY STOPPING & SIMPAN MODEL TERBAIK ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 
            best_preds_raw = predictions
            best_labels_raw = targets
            
            torch.save(model.state_dict(), f"{output_dir}/best_finetuned_model.pt")
            with open(f"{output_dir}/config.json", 'w') as f:
                json.dump(config, f)
            logger.info("Model membaik! Bobot terbaik disimpan.")
        else:
            patience_counter += 1
            logger.info(f"Val Loss tidak membaik. Early Stopping Counter: {patience_counter}/{patience}")

        # --- SIMPAN CHECKPOINT SETIAP EPOCH ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'history': history
        }, resume_checkpoint_path)

        if patience_counter >= patience:
            logger.info(f"\nEARLY STOPPING DIPICU! Model berhenti belajar karena Val Loss tidak membaik selama {patience} epoch berturut-turut.")
            break

    # Bikin Plot di akhir
    if best_preds_raw is not None:
        plot_finetune_results(history, best_preds_raw, best_labels_raw, output_dir, logger)
        
    logger.info("Fine-Tuning Selesai! Model siap digunakan untuk inferensi di Kalimantan Selatan.")


if __name__ == '__main__':
    main()