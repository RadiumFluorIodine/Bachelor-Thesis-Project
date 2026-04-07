"""
AGB Prediction Inference - Mode 3: Smart Masked Inference with GEE
Menggunakan Google Earth Engine untuk Region-of-Interest (ROI) Masking.

Fitur Utama:
- Mengunduh batas provinsi dari GEE otomatis
- Sinkronisasi CRS otomatis antara GEE (WGS84) dan GeoTIFF (UTM/Lainnya)
- Smart Tiling: Melewati (skip) potongan gambar yang berada di lautan
- Memory-efficient batch processing
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime
import ee

# Setup paths
current_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.models.utae import UTAE

# Ganti dengan Project ID GEE Anda
PROJECT_ID = 'data-skripsi-473712'

class GeoTIFFInferenceEngine:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        normalization_stats_path: str,
        roi_name: str = "Hulu Sungai Utara",
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.roi_name = roi_name
        
        # Inisialisasi Google Earth Engine
        self._init_gee()
        
        # Load components
        self.config = self._load_config(config_path)
        self.norm_stats = self._load_normalization_stats(normalization_stats_path)
        self.model = self._load_model(model_path)
        
        # Tiling parameters
        self.tile_size = 128  
        self.overlap = 16  
        self.batch_size = 128  
        
        self.logger.info(f"✅ Smart GEE Inference Engine initialized")
        self.logger.info(f"    ROI Target: {self.roi_name}")
        self.logger.info(f"    Device: {self.device}")

    def _init_gee(self):
        self.logger.info("Menginisialisasi Earth Engine...")
        try:
            ee.Initialize(project=PROJECT_ID)
            print("✓ Connect to Earth Engine.")
        except Exception:
            print("Requires GEE authentication....")
            ee.Authenticate()
            ee.Initialize(project=PROJECT_ID)
            print("✓ Authentication complete.")

    def _load_config(self, config_path: str) -> dict:
        with open(config_path) as f:
            return json.load(f)
    
    def _load_normalization_stats(self, stats_path: str) -> dict:
        with open(stats_path) as f:
            stats = json.load(f)
        mean_key = 's2_mean_per_band' if 's2_mean_per_band' in stats else 'mean'
        std_key = 's2_std_per_band' if 's2_std_per_band' in stats else 'std'
        return {
            'mean': stats[mean_key],
            'std': stats[std_key],
            'agb_log_mean': stats.get('agb_log_mean', 4.1454515),
            'agb_log_std': stats.get('agb_log_std', 1.1578547)
        }
    
    def _load_model(self, model_path: str) -> UTAE:
        model = UTAE(
            input_dim=self.config['input_dim'],
            output_dim=self.config['output_dim'],
            encoder_widths=self.config['encoder_widths'],
            decoder_widths=self.config['decoder_widths'],
            d_model=self.config['d_model'],
            n_head=self.config['n_head']
        ).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model.eval()
        return model

    def normalize_tile(self, tile: np.ndarray) -> np.ndarray:
        mean = np.array(self.norm_stats['mean'], dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.array(self.norm_stats['std'], dtype=np.float32).reshape(1, -1, 1, 1)
        return (tile - mean) / (std + 1e-8)

    def generate_tiles(self, raster_shape: Tuple[int, int], roi_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        height, width = raster_shape
        stride = self.tile_size - self.overlap
        tiles = []
        skipped_tiles = 0
        for row in range(0, height, stride):
            for col in range(0, width, stride):
                h = min(self.tile_size, height - row)
                w = min(self.tile_size, width - col)
                if h < self.tile_size // 2 or w < self.tile_size // 2: continue
                # Skip jika 100% lautan berdasarkan mask GEE
                if not roi_mask[row:row+h, col:col+w].any():
                    skipped_tiles += 1
                    continue
                tiles.append((row, col, h, w))
        self.logger.info(f"✅ Tiling: {len(tiles)} tiles daratan, {skipped_tiles} tiles lautan dilewati.")
        return tiles

    def read_tile(self, paths, row, col, h, w):
        window = Window(col, row, w, h)
        data_list = []
        for p in paths:
            with rasterio.open(p) as src:
                d = src.read(window=window).astype(np.float32)
                if d.shape[1] < self.tile_size or d.shape[2] < self.tile_size:
                    d = np.pad(d, ((0,0), (0, self.tile_size - h), (0, self.tile_size - w)), mode='reflect')
                data_list.append(d)
        return np.array(data_list) # (T, C, H, W)

    def infer_on_geotiffs(self, geotiff_paths: List[str], output_path: str):
        # Ambil metadata dari file pertama
        with rasterio.open(geotiff_paths[0]) as src:
            meta = src.meta.copy()
            height, width = src.height, src.width
            crs = src.crs
            transform = src.transform

        # 1. GEE Masking & Coordinate Transform
        self.logger.info(f"📥 Mengambil ROI {self.roi_name} dari GEE...")
        fc = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq("ADM1_NAME", self.roi_name))
        roi_geom_wgs84 = fc.geometry().getInfo()

        self.logger.info(f"🔄 Menyingkronkan CRS GEE (WGS84) ke Raster ({crs.to_string()})...")
        roi_geom_projected = transform_geom('EPSG:4326', crs, roi_geom_wgs84)
        roi_mask = geometry_mask([roi_geom_projected], out_shape=(height, width), transform=transform, invert=True)

        if np.sum(roi_mask) == 0:
            self.logger.error("❌ ROI Mask kosong! Koordinat GEE dan GeoTIFF tidak beririsan.")
            return

        # 2. Tiling & Inference
        tile_coords = self.generate_tiles((height, width), roi_mask)
        output_array = np.full((height, width), -9999.0, dtype=np.float32)
        count_array = np.zeros((height, width), dtype=np.float32)

        batch_positions = torch.tensor([list(range(1, len(geotiff_paths)+1))], dtype=torch.long).to(self.device)
        
        buffer, coords_buffer = [], []
        for i, (r, c, h, w) in enumerate(tqdm(tile_coords, desc="GPU Processing")):
            tile_data = self.read_tile(geotiff_paths, r, c, h, w)
            buffer.append(self.normalize_tile(tile_data))
            coords_buffer.append((r, c, h, w))

            if len(buffer) == self.batch_size or i == len(tile_coords) - 1:
                batch_in = torch.from_numpy(np.array(buffer)).to(self.device).float()
                b_pos = batch_positions.repeat(len(buffer), 1)
                
                with torch.no_grad():
                    pred = self.model(batch_in, batch_positions=b_pos)['agb']
                    # Denormalisasi
                    pred = torch.expm1((pred * self.norm_stats['agb_log_std']) + self.norm_stats['agb_log_mean'])
                    pred = pred.cpu().numpy()

                for j, (br, bc, bh, bw) in enumerate(coords_buffer):
                    p_tile = pred[j, 0, :bh, :bw]
                    mask_area = roi_mask[br:br+bh, bc:bc+bw]
                    # Hanya isi area yang masuk dalam ROI GEE
                    p_tile[~mask_area] = -9999.0
                    
                    target_area = output_array[br:br+bh, bc:bc+bw]
                    target_area[mask_area] = np.where(count_array[br:br+bh, bc:bc+bw][mask_area] == 0, 
                                                      p_tile[mask_area], 
                                                      target_area[mask_area] + p_tile[mask_area])
                    count_array[br:br+bh, bc:bc+bw] += mask_area.astype(float)
                buffer, coords_buffer = [], []

        # 3. Finalize & Save
        output_array[count_array > 1] /= count_array[count_array > 1]
        output_array[count_array == 0] = -9999.0

        meta.update(dtype='float32', count=1, nodata=-9999.0, compress='lzw')
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(output_array, 1)
        
        self.logger.info(f"✅ Prediksi selesai. Output: {output_path}")

def main():
    # Setup Default Paths
    MODEL = 'src/results/checkpoints/U-TAE_Finetuned_Kalsel/best_finetuned_model.pt'
    CONFIG = 'src/results/checkpoints/U-TAE/config.json'
    STATS = 'data/processed/kalsel/normalization.json'
    IN_DIR = 'data/raw/kalimantan-selatan'
    OUT_FILE = 'src/results/inference/kalimantan-selatan.tif'

    geotiff_files = sorted(Path(IN_DIR).glob('*.tif'))
    if not geotiff_files:
        print(f"❌ GeoTIFF tidak ditemukan di {IN_DIR}")
        return

    engine = GeoTIFFInferenceEngine(MODEL, CONFIG, STATS, roi_name="Lampung")
    engine.infer_on_geotiffs([str(f) for f in geotiff_files], OUT_FILE)

if __name__ == '__main__':
    main()