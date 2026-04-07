"""
Preprocess Spatiotemporal with no cliping [-3, 3]

Objective:
- Log-transform AGB for handle right-skewed distribution
- Per-band z-score normalization for Sentinel-2
- Valid mask combines S2 + AGB validity
- Proper nodata handling (0 → NaN)

Expected output:
- data/processed/
  - 00000.npz, 00001.npz, ... (patch files)
  - normalization.json (per-band statistics)

"""

import rasterio
from rasterio.windows import Window 
import numpy as np
import os
import glob
import json
from tqdm import tqdm
import gc
from natsort import natsorted


# Configuration
INPUT_DIR = 'data/raw/lampung'
OUTPUT_DIR = 'data/processed/lampung/no_cliping'
PATCH_SIZE = 128
STRIDE = 128
NODATA_THRESHOLD = 0.1  
MIN_VALID_MONTHS = 6    


def compute_normalization_stats_per_band(s2_files, agb_file):
    """
    Compute per-band statistics for spectral consistency.
    
    
    Args:
        s2_files: List of S2 files
        agb_file: Path to AGB label file
    
    Returns:
        stats: Dictionary with per-band mean/std
    """
    print(" Computing normalization AGB with log-transform")
    
    # Step 1: AGB Statistics
    agb_log_sum = 0.0
    agb_log_sum_sq = 0.0
    agb_log_count = 0

    agb_raw_sum = 0.0
    agb_raw_count = 0
    
    with rasterio.open(agb_file) as agb_src:
        height, width = agb_src.height, agb_src.width
        
        # Samples in windows for efficiency
        for i in range(0, height, 500):
            for j in range(0, width, 500):
                window_size = min(500, height - i, width - j)
                window = Window(j, i, window_size, window_size)
                
                try:
                    data = agb_src.read(1, window=window).astype(np.float32)
                    
                    valid_mask = (data > 0) & np.isfinite(data)
                    valid = data[valid_mask]
                    
                    if len(valid) > 0:
                        # Log-transform: log(1 + x)
                        valid_log = np.log1p(valid)
                        
                        agb_log_sum += np.sum(valid_log)
                        agb_log_sum_sq += np.sum(valid_log ** 2)
                        agb_log_count += len(valid_log)
                        
                        # Raw stats for reference
                        agb_raw_sum += np.sum(valid)
                        agb_raw_count += len(valid)
                    
                    del data, valid
                    gc.collect()
                    
                except Exception as e:
                    print(f"⚠️ Warning: Read AGB window ({i}, {j}): {e}")
                    continue
    
    print(f"   ✓ AGB samples processed: {agb_log_count:,}")
    
    if agb_log_count == 0:
        raise ValueError("✗ No valid AGB data found!")
    
    # Compute log-space statistics
    agb_log_mean = agb_log_sum / agb_log_count
    agb_log_var = (agb_log_sum_sq / agb_log_count) - (agb_log_mean ** 2)
    agb_log_std = np.sqrt(max(agb_log_var, 1e-8))

    # Raw statistics for reporting
    agb_raw_mean = agb_raw_sum / agb_raw_count

    print(f"\n AGB Distribution:")
    print(f"   Raw space:  Mean={agb_raw_mean:.1f} Mg/ha")
    print(f"   Log space:  Mean={agb_log_mean:.3f}, Std={agb_log_std:.3f}")


    # Sentinel 2 Per-Band Statistics (Z-Score)
    s2_mean_per_band = np.zeros((10,))
    s2_sum_sq_per_band = np.zeros((10,))
    s2_count_per_band = np.zeros((10,))

    print("   Processing S2 files for per-band statistics...")
    
    for s2_file in tqdm(s2_files, desc="S2 Stats", unit="month"):
        with rasterio.open(s2_file) as src:
            num_bands = src.count
            if num_bands != 10:
                print(f"WARNING: Expected 10 bands di {s2_file}, got {num_bands}")
            
            height, width = src.height, src.width
            
            # Sample in windows
            for i in range(0, height, 500):
                for j in range(0, width, 500):
                    window_size = min(500, height - i, width - j)
                    window = Window(j, i, window_size, window_size)
                    
                    try:
                        # Read all bands: (10, H, W)
                        data = src.read(window=window).astype(np.float32)
                        
                        # Process each band separately
                        for band_idx in range(10):
                            band_data = data[band_idx, :, :]
                            valid = band_data[
                                (band_data > 0) & np.isfinite(band_data)]
                            
                            if len(valid) > 0:
                                s2_mean_per_band[band_idx] += np.sum(valid)
                                s2_sum_sq_per_band[band_idx] += np.sum(valid ** 2)
                                s2_count_per_band[band_idx] += len(valid)
                        
                        del data
                        gc.collect()
                    except Exception as e:
                        continue
    
    print(f"   ✓ S2 per-band samples: {s2_count_per_band.astype(int)}")
    
    # Compute mean and std per band
    s2_std_per_band = np.zeros((10,))
    for band_idx in range(10):
        if s2_count_per_band[band_idx] > 0:
            s2_mean_per_band[band_idx] /= s2_count_per_band[band_idx]
            var = (
                (s2_sum_sq_per_band[band_idx] / s2_count_per_band[band_idx]) -
                (s2_mean_per_band[band_idx] ** 2)
            )
            s2_std_per_band[band_idx] = np.sqrt(max(var, 1e-8))
        else:
            s2_std_per_band[band_idx] = 1.0  # Default fallback
    
    # Return statistics
    stats = {
        # Per-band S2 statistics (z-score)
        's2_mean_per_band': s2_mean_per_band.tolist(),
        's2_std_per_band': s2_std_per_band.tolist(),
        's2_mean': float(np.mean(s2_mean_per_band)),
        's2_std': float(np.mean(s2_std_per_band)),
        
        # AGB LOG-SPACE statistics
        'agb_log_mean': float(agb_log_mean),
        'agb_log_std': float(agb_log_std),
        
        # AGB raw statistics (for reference only)
        'agb_raw_mean': float(agb_raw_mean),
        'agb_raw_min': 0.0,
        
        # Metadata
        'normalization_method': 'log-transform',
        'transform_function': 'np.log1p(x)',
        'inverse_function': 'np.expm1(x)',
        'n_s2_samples_per_band': s2_count_per_band.tolist(),
        'n_agb_samples': int(agb_log_count),
    }
    
    return stats
    

def check_patch_validity(s2_np, agb_np, threshold=NODATA_THRESHOLD, min_months=MIN_VALID_MONTHS):
    """
    Check valid patch.
    
    s2_np: (T=12, C=10, H, W) S2 data (with NaN for nodata)
        agb_np: (H, W) AGB label
        threshold: Max fraction of invalid pixels allowed
        min_months: Minimum valid months required per pixel
    
    Returns:
        valid_mask: (H, W) boolean mask
        is_valid: bool, True if patch meets quality threshold
    """
    
    # Sentinel-2 Quality Assessment  
    s2_valid_per_band = np.isfinite(s2_np)
    s2_valid_per_month = s2_valid_per_band.all(axis=1) # (T=12, H, W)
    valid_month_count = s2_valid_per_month.sum(axis=0) # (H, W)

    # Pixels are considered healthy if they have at least 6 months of clean data.
    s2_spatial_valid = valid_month_count >= min_months
    s2_health_ratio = s2_spatial_valid.sum() / s2_spatial_valid.size

    # AGB Quality Assessment
    agb_valid = np.isfinite(agb_np) & (agb_np >= 0)
    agb_health_ratio = agb_valid.sum() / agb_valid.size

    # Both must pass their respective thresholds.
    s2_pass = s2_health_ratio >= (1 - threshold)
    agb_pass = agb_health_ratio >= (1 - threshold)

    is_valid = s2_pass and agb_pass

    # Masking for Training
    valid_pixels = s2_spatial_valid & agb_valid
    
    return valid_pixels, is_valid


def main():
    """Main preprocessing pipeline."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load S2 files
    search_pattern = os.path.join(INPUT_DIR, "S2_*_M*.tif")
    raw_files = glob.glob(search_pattern)
    
    # Fallback 
    if len(raw_files) == 0:
        print(" Main pattern not found, trying fallback...")
        search_pattern = os.path.join(INPUT_DIR, "S2_M*.tif")
        raw_files = glob.glob(search_pattern)

    # Sorting 
    s2_files = natsorted(raw_files)
    
    agb_file = os.path.join(INPUT_DIR, "AGB_*_Label.tif")
    agb_matches = glob.glob(agb_file)
    
    if len(agb_matches) == 0:
        print(f"✗ CRITICAL: AGB file not found in {INPUT_DIR}")
        return

    agb_file = agb_matches[0]
    
    if len(s2_files) != 12:
        print(f"✗ CRITICAL: Expected 12 S2 files, found {len(s2_files)}")
        return
    
    print("✓ Data Files Loaded:")
    print(f"   Start (Jan): {os.path.basename(s2_files[0])}")
    print(f"   End (Dec): {os.path.basename(s2_files[-1])}")
    print(f"   AGB Label: {os.path.basename(agb_file)}")

    
    # Step 1: Compute normalization stats
    print("\n" + "-"*70)
    print("STEP 1: Computing Per-Band Normalization Statistics")
    print("-"*70)
    
    norm_stats = compute_normalization_stats_per_band(s2_files, agb_file)
    
    print("\n Per-Band Statistics:")
    for band_idx in range(10):
        print(f"      Band {band_idx}: "
              f"μ={norm_stats['s2_mean_per_band'][band_idx]:.1f}, "
              f"σ={norm_stats['s2_std_per_band'][band_idx]:.1f}")
    
    print(f"\n   AGB (Log-Transform):")
    print(f"      Log-space: μ={norm_stats['agb_log_mean']:.3f}, σ={norm_stats['agb_log_std']:.3f}")
    print(f"      Raw-space: μ={norm_stats['agb_raw_mean']:.1f} Mg/ha (reference)")
    
    
    # Step 2: Open data sources
    print("\n" + "-"*70)
    print("STEP 2: Opening Data Sources")
    print("-"*70)
    
    s2_srcs = [rasterio.open(f) for f in s2_files]
    agb_src = rasterio.open(agb_file)
    
    height = s2_srcs[0].height
    width = s2_srcs[0].width
    num_bands = s2_srcs[0].count
    
    print(f"Image dimensions: {width} × {height} pixels")
    print(f"S2 bands per image: {num_bands}")
    
    # Step 3: Generate patch positions
    print("\n" + "-"*70)
    print("STEP 3: Generating Patch Positions")
    print("-"*70)
    
    patch_positions_r = list(range(0, height - PATCH_SIZE + 1, STRIDE))
    patch_positions_c = list(range(0, width - PATCH_SIZE + 1, STRIDE))
    
    # Add the last patch if it is not there yet
    if (height - PATCH_SIZE) not in patch_positions_r:
        patch_positions_r.append(height - PATCH_SIZE)
    if (width - PATCH_SIZE) not in patch_positions_c:
        patch_positions_c.append(width - PATCH_SIZE)
    
    total_patches_estimate = len(patch_positions_r) * len(patch_positions_c)
    print(f"Estimated total patches: {total_patches_estimate}")
    
    # Step 4: Main processing loop
    print("\n" + "-"*70)
    print("STEP 4: Processing Patches")
    print("-"*70)
    
    patch_id = 0
    valid_count = 0
    skipped_count = 0
    
    with tqdm(total=total_patches_estimate, desc="Generating Patches",
              unit="patch") as pbar:
        for r in patch_positions_r:
            for c in patch_positions_c:
                window = Window(c, r, PATCH_SIZE, PATCH_SIZE)
                
                try:
                    # Read all 12 months
                    stack = []
                    for src in s2_srcs:
                        arr = src.read(window=window).astype(np.float32)
                        
                        if arr.shape[0] != 10:
                            raise ValueError(
                                f"Expected 10 bands, got {arr.shape[0]}"
                            )
                        
                        # Convert 0 (nodata) to NaN
                        arr[arr == 0] = np.nan
                        
                        stack.append(arr)
                    
                    # Stack to (T=12, C=10, H=128, W=128)
                    s2_np = np.stack(stack, axis=0)
                    
                    # Read AGB label
                    agb_np = agb_src.read(1, window=window).astype(np.float32)
                    agb_np[agb_np < 0] = np.nan


                    # Check validity
                    valid_mask, is_valid = check_patch_validity(s2_np, agb_np)
                    
                    if not is_valid:
                        skipped_count += 1
                        pbar.update(1)
                        continue
                    
                    # Normalize
                    # S2: Per-band z-score
                    s2_norm = np.zeros_like(s2_np, dtype=np.float32)
                    for band_idx in range(10):
                        mean = norm_stats['s2_mean_per_band'][band_idx]
                        std = norm_stats['s2_std_per_band'][band_idx] + 1e-8
                        
                        band = s2_np[:, band_idx, :, :]
                        valid = np.isfinite(band)
                        
                        s2_norm[:, band_idx, :, :] = 0.0
                        s2_norm[:, band_idx, :, :][valid] = (band[valid] - mean) / std
                    
                    # s2_norm = np.clip(s2_norm, -3, 3) No cliping
                    
                    # AGB: Log-transform + z-score
                    agb_norm = np.zeros_like(agb_np, dtype=np.float32)
                    valid_agb = np.isfinite(agb_np)
                    
                    if valid_agb.sum() > 0:
                        # Step 1: log(1 + x)
                        agb_log = np.log1p(agb_np[valid_agb])
                        
                        # Step 2: z-score in log-space
                        agb_log_norm = (agb_log - norm_stats['agb_log_mean']) / (norm_stats['agb_log_std'] + 1e-8)
                        
                        # No Clip outliers
                        # agb_log_norm = np.clip(agb_log_norm, -3, 3)
                        
                        agb_norm[valid_agb] = agb_log_norm
                    
                    # Save patch
                    output_file = os.path.join(OUTPUT_DIR, f"{patch_id:05d}.npz")
                    np.savez_compressed(
                        output_file,
                        image=s2_norm.astype(np.float32),       # (12, 10, 128, 128)
                        label=agb_norm.astype(np.float32),      # (128, 128)
                        valid_mask=valid_mask.astype(bool)      # (128, 128)
                    )
                    
                    patch_id += 1
                    valid_count += 1
                    
                    if patch_id % 1000 == 0:
                        gc.collect()
                    
                except Exception as e:
                    print(f"\n Error at ({r}, {c}): {e}")
                    skipped_count += 1
                
                pbar.update(1)
    
    # Cleanup
    for src in s2_srcs:
        src.close()
    agb_src.close()
    
    # Save normalization stats
    norm_path = os.path.join(OUTPUT_DIR, 'normalization.json')
    with open(norm_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    
    # Summary
    print("\n" + "-"*70)
    print("PREPROCESSING SUMMARY (LOG-TRANSFORM)")
    print("-"*70)
    print(f"✓ Total Patches Saved    : {valid_count:,}")
    print(f"  Invalid Patches Skipped: {skipped_count:,}")
    print(f"  Processing Rate        : {100*valid_count/total_patches_estimate:.1f}%")
    print(f"\n✓ Output Shape: (T=12, C=10, H=128, W=128)")
    print(f"✓ S2: Per-band z-score normalization")
    print(f"✓ AGB: Log-transform + z-score (log(1+x))")
    print(f"✓ Valid mask saved (S2 + AGB combined)")
    print(f"✓ Min valid months: {MIN_VALID_MONTHS}/12")
    print(f"\n✓ Normalization stats: {norm_path}")
    print(f"\n Denormalization formula:")
    print(f"   AGB_raw = exp(z*σ + μ) - 1")
    print(f"   where z = normalized value, μ = {norm_stats['agb_log_mean']:.3f}, σ = {norm_stats['agb_log_std']:.3f}")
    print("-"*70)


if __name__ == "__main__":
    main()