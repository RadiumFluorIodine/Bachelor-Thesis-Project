"""
Data Statistics Extraction Tool
Tujuan: Menghitung dan menampilkan statistik dataset S2 dan AGB
Format disesuaikan untuk Tabel 4.4 Skripsi (Perhitungan Aktual Data Mentah vs Normalisasi)
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import os
import glob
import gc
import sys  # Ditambahkan untuk memperbaiki tampilan tqdm
from tqdm import tqdm
from natsort import natsorted

# Configuration
INPUT_DIR = 'data/raw/lampung'
PATCH_SIZE = 128
STRIDE = 128

def extract_statistics():
    print("🔍 Mencari file citra satelit dan AGB...")
    
    # Load S2 files
    search_pattern = os.path.join(INPUT_DIR, "S2_*_M*.tif")
    raw_files = glob.glob(search_pattern)
    
    if len(raw_files) == 0:
        search_pattern = os.path.join(INPUT_DIR, "S2_M*.tif")
        raw_files = glob.glob(search_pattern)

    s2_files = natsorted(raw_files)
    agb_file = glob.glob(os.path.join(INPUT_DIR, "AGB_*_Label.tif"))
    
    if len(s2_files) != 12 or len(agb_file) == 0:
        print(f"❌ Error: Ditemukan {len(s2_files)} file S2 dan {len(agb_file)} file AGB.")
        return
        
    agb_file = agb_file[0]

    with rasterio.open(s2_files[0]) as src:
        height, width = src.height, src.width
        num_bands = src.count

    # ==================================================================
    # PASS 1: Menghitung Statistik Awal (Mean & Std Mentah)
    # ==================================================================
    print("\n[PASS 1] Menghitung statistik awal data mentah...")
    
    # --- AGB PASS 1 ---
    agb_raw_sum = 0.0
    agb_raw_sum_sq = 0.0 
    agb_raw_count = 0
    agb_log_sum = 0.0
    agb_log_sum_sq = 0.0

    with rasterio.open(agb_file) as agb_src:
        for i in range(0, height, 500):
            for j in range(0, width, 500):
                window_size = min(500, height - i, width - j)
                window = Window(j, i, window_size, window_size)
                
                try:
                    data = agb_src.read(1, window=window).astype(np.float32)
                    valid = data[(data > 0) & np.isfinite(data)]
                    
                    if len(valid) > 0:
                        # Raw
                        agb_raw_sum += np.sum(valid)
                        agb_raw_sum_sq += np.sum(valid ** 2)
                        agb_raw_count += len(valid)
                        
                        # Log1p
                        valid_log = np.log1p(valid)
                        agb_log_sum += np.sum(valid_log)
                        agb_log_sum_sq += np.sum(valid_log ** 2)
                except Exception:
                    pass

    if agb_raw_count > 0:
        agb_raw_mean = agb_raw_sum / agb_raw_count
        agb_raw_var = (agb_raw_sum_sq / agb_raw_count) - (agb_raw_mean ** 2)
        agb_raw_std = np.sqrt(max(agb_raw_var, 1e-8))
        
        agb_log_mean = agb_log_sum / agb_raw_count
        agb_log_var = (agb_log_sum_sq / agb_raw_count) - (agb_log_mean ** 2)
        agb_log_std = np.sqrt(max(agb_log_var, 1e-8))
    else:
        agb_raw_mean = agb_raw_std = agb_log_mean = agb_log_std = 0.0

    # --- S2 PASS 1 ---
    s2_mean_per_band = np.zeros(10)
    s2_sum_sq_per_band = np.zeros(10)
    s2_count_per_band = np.zeros(10)

    # PERBAIKAN TQDM: Menambahkan parameter ncols dan file=sys.stdout
    for s2_file in tqdm(s2_files, desc="Pass 1 - Sentinel-2", unit="month", ncols=100, file=sys.stdout):
        with rasterio.open(s2_file) as src:
            for i in range(0, height, 500):
                for j in range(0, width, 500):
                    window_size = min(500, height - i, width - j)
                    window = Window(j, i, window_size, window_size)
                    
                    try:
                        data = src.read(window=window).astype(np.float32)
                        for band_idx in range(10):
                            band_data = data[band_idx, :, :]
                            valid = band_data[(band_data > 0) & np.isfinite(band_data)]
                            
                            if len(valid) > 0:
                                s2_mean_per_band[band_idx] += np.sum(valid)
                                s2_sum_sq_per_band[band_idx] += np.sum(valid ** 2)
                                s2_count_per_band[band_idx] += len(valid)
                        
                        del data
                        gc.collect()
                    except Exception:
                        pass

    s2_std_per_band = np.zeros(10)
    for band_idx in range(10):
        if s2_count_per_band[band_idx] > 0:
            s2_mean_per_band[band_idx] /= s2_count_per_band[band_idx]
            var = (s2_sum_sq_per_band[band_idx] / s2_count_per_band[band_idx]) - (s2_mean_per_band[band_idx] ** 2)
            s2_std_per_band[band_idx] = np.sqrt(max(var, 1e-8))

    # ==================================================================
    # PASS 2: Transformasi & Kalkulasi Ulang (Data Normalisasi Aktual)
    # ==================================================================
    print("\n[PASS 2] Mengkalkulasi statistik data setelah diterapkan rumus normalisasi (Z-Score)...")
    
    # --- AGB PASS 2 ---
    agb_norm_sum = 0.0
    agb_norm_sum_sq = 0.0
    
    with rasterio.open(agb_file) as agb_src:
        for i in range(0, height, 500):
            for j in range(0, width, 500):
                window_size = min(500, height - i, width - j)
                window = Window(j, i, window_size, window_size)
                try:
                    data = agb_src.read(1, window=window).astype(np.float32)
                    valid = data[(data > 0) & np.isfinite(data)]
                    if len(valid) > 0:
                        # TERAPKAN RUMUS: (Log(X) - Mean) / Std
                        norm_valid = (np.log1p(valid) - agb_log_mean) / agb_log_std
                        
                        agb_norm_sum += np.sum(norm_valid)
                        agb_norm_sum_sq += np.sum(norm_valid ** 2)
                except Exception:
                    pass
                    
    agb_norm_mean = agb_norm_sum / agb_raw_count if agb_raw_count > 0 else 0.0
    agb_norm_var = (agb_norm_sum_sq / agb_raw_count) - (agb_norm_mean ** 2) if agb_raw_count > 0 else 0.0
    agb_norm_std = np.sqrt(max(agb_norm_var, 1e-8)) if agb_raw_count > 0 else 0.0

    # --- S2 PASS 2 ---
    s2_norm_sum_per_band = np.zeros(10)
    s2_norm_sum_sq_per_band = np.zeros(10)

    # PERBAIKAN TQDM: Menambahkan parameter ncols dan file=sys.stdout
    for s2_file in tqdm(s2_files, desc="Pass 2 - Sentinel-2", unit="month", ncols=100, file=sys.stdout):
        with rasterio.open(s2_file) as src:
            for i in range(0, height, 500):
                for j in range(0, width, 500):
                    window_size = min(500, height - i, width - j)
                    window = Window(j, i, window_size, window_size)
                    try:
                        data = src.read(window=window).astype(np.float32)
                        for band_idx in range(10):
                            band_data = data[band_idx, :, :]
                            valid = band_data[(band_data > 0) & np.isfinite(band_data)]
                            if len(valid) > 0:
                                # TERAPKAN RUMUS: (X - Mean) / Std
                                norm_valid = (valid - s2_mean_per_band[band_idx]) / s2_std_per_band[band_idx]
                                
                                s2_norm_sum_per_band[band_idx] += np.sum(norm_valid)
                                s2_norm_sum_sq_per_band[band_idx] += np.sum(norm_valid ** 2)
                        del data
                        gc.collect()
                    except Exception:
                        pass

    s2_norm_mean_per_band = np.zeros(10)
    s2_norm_std_per_band = np.zeros(10)
    for band_idx in range(10):
        if s2_count_per_band[band_idx] > 0:
            s2_norm_mean_per_band[band_idx] = s2_norm_sum_per_band[band_idx] / s2_count_per_band[band_idx]
            var = (s2_norm_sum_sq_per_band[band_idx] / s2_count_per_band[band_idx]) - (s2_norm_mean_per_band[band_idx] ** 2)
            s2_norm_std_per_band[band_idx] = np.sqrt(max(var, 1e-8))

    # =========================================================================
    # 3. PRINT OUTPUT UNTUK TABEL SKRIPSI
    # =========================================================================
    band_names = [
        "B02 (Blue)", "B03 (Green)", "B04 (Red)", "B05 (Red Edge 1)", 
        "B06 (Red Edge 2)", "B07 (Red Edge 3)", "B08 (NIR)", 
        "B8A (NIR Narrow)", "B11 (SWIR 1)", "B12 (SWIR 2)"
    ]

    print("\n" + "="*85)
    print(f"{'Jenis Data':<20} | {'Data Mentah (Mean ± Std)':<28} | {'Data Normalisasi (Mean ± Std)':<30}")
    print("="*85)
    
    # Cetak Baris S2
    for i, name in enumerate(band_names):
        raw_str = f"{s2_mean_per_band[i]:.4f} ± {s2_std_per_band[i]:.4f}"
        norm_str = f"{s2_norm_mean_per_band[i]:.4f} ± {s2_norm_std_per_band[i]:.4f}"
        print(f"{name:<20} | {raw_str:<28} | {norm_str:<30}")
    
    # Cetak Baris AGB
    agb_raw_str = f"{agb_raw_mean:.4f} ± {agb_raw_std:.4f}"
    agb_norm_str = f"{agb_norm_mean:.4f} ± {agb_norm_std:.4f}"
    print(f"{'ESA CCI AGB':<20} | {agb_raw_str:<28} | {agb_norm_str:<30}")
    print("="*85)
    
if __name__ == "__main__":
    extract_statistics()