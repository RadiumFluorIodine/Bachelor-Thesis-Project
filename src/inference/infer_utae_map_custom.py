"""
AGB Prediction Inference - Mode 3: Custom Asset GEE Masking
Menggunakan Google Earth Engine untuk Custom Region-of-Interest (ROI) Masking.

Fitur Utama:
- Mengunduh batas poligon langsung dari GEE Custom Asset (sawit-pdl)
- Sinkronisasi CRS otomatis antara GEE (WGS84) dan GeoTIFF (UTM/Lainnya)
- Smart Tiling: Melewati (skip) potongan gambar di luar area sawit
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
        asset_id: str,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
        self.logger = logging.getLogger(__name__)
        
        self.asset_id = asset_id
        
        # Load components
        self.config = self._load_config(config_path)
        self.norm_stats = self._load_normalization_stats(normalization_stats_path)
        self.model = self._load_model(model_path)
        
        # Tiling parameters
        self.tile_size = 128  
        self.overlap = 16  
        self.batch_size = 128  
        
        self.logger.info(f"✅ Smart GEE Inference Engine initialized")
        self.logger.info(f"    Asset Target : {self.asset_id}")
        self.logger.info(f"    Device       : {self.device}")

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
                
                # Skip jika potongan (tile) tidak beririsan dengan mask
                if not roi_mask[row:row+h, col:col+w].any():
                    skipped_tiles += 1
                    continue
                tiles.append((row, col, h, w))
                
        self.logger.info(f"✅ Tiling: {len(tiles)} tiles diproses, {skipped_tiles} tiles dilewati (di luar mask).")
        return tiles

    def read_tile(self, paths, row, col, h, w):
        window = Window(col, row, w, h)
        data_list = []
        for p in paths:
            with rasterio.open(p) as src:
                d = src.read(window=window).astype(np.float32)
                
                # Safety Net 1: Jika kotak window berada di luar raster (kosong)
                if d.size == 0:
                    d = np.zeros((src.count, self.tile_size, self.tile_size), dtype=np.float32)
                else:
                    # Safety Net 2: Hitung padding berdasarkan SHAPE ASLI dari data yang ditarik
                    pad_h = self.tile_size - d.shape[1]
                    pad_w = self.tile_size - d.shape[2]
                    
                    # Lakukan padding HANYA jika matriks kurang dari 128x128
                    if pad_h > 0 or pad_w > 0:
                        d = np.pad(d, (
                            (0, 0), 
                            (0, max(0, pad_h)), 
                            (0, max(0, pad_w))
                        ), mode='reflect')
                    
                    # Safety Net 3: Pastikan ukuran mutlak 128x128 (potong jika berlebih)
                    d = d[:, :self.tile_size, :self.tile_size]
                    
                data_list.append(d)
                
        # Deteksi dini jika ada file GeoTIFF yang kehilangan band/channel
        channels = [x.shape[0] for x in data_list]
        if len(set(channels)) > 1:
            self.logger.error(f"❌ FATAL: Jumlah band tidak konsisten antar GeoTIFF: {channels}")
            raise ValueError("Dimensi channel (bands) tidak konsisten antar 13 file GeoTIFF Anda.")
            
        return np.array(data_list) # (T, C, H, W)

    def infer_on_geotiffs(self, geotiff_paths: List[str], output_path: str):
        # Ambil metadata dari file pertama
        with rasterio.open(geotiff_paths[0]) as src:
            meta = src.meta.copy()
            height, width = src.height, src.width
            crs = src.crs
            transform = src.transform

        # 1. GEE Masking & Coordinate Transform (Menggunakan Custom Asset)
        self.logger.info(f"📥 Mengambil vektor Asset dari GEE: {self.asset_id}...")
        fc = ee.FeatureCollection(self.asset_id)
        roi_geom_wgs84 = fc.geometry().getInfo()

        self.logger.info(f"🔄 Menyingkronkan CRS GEE (WGS84) ke Raster ({crs.to_string()})...")
        roi_geom_projected = transform_geom('EPSG:4326', crs, roi_geom_wgs84)
        roi_mask = geometry_mask([roi_geom_projected], out_shape=(height, width), transform=transform, invert=True)

        if np.sum(roi_mask) == 0:
            self.logger.error("❌ ROI Mask kosong! Poligon Asset dan GeoTIFF tidak beririsan.")
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
                    # Hanya isi area yang masuk dalam Custom Asset GEE
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

def init_gee():
    print("\nMenginisialisasi Earth Engine...")
    try:
        ee.Initialize(project=PROJECT_ID)
        print("✓ Terhubung ke Earth Engine.\n")
    except Exception:
        print("Membutuhkan Autentikasi GEE....")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
        print("✓ Autentikasi Selesai.\n")

def main():
    # Inisialisasi Earth Engine
    init_gee()

    # Setup Paths
    MODEL = 'src/results/checkpoints/U-TAE_Finetuned_Kalsel/best_finetuned_model.pt'
    CONFIG = 'src/results/checkpoints/U-TAE/config.json'
    STATS = 'data/processed/kalsel/normalization.json'
    IN_DIR = 'data/raw/kalimantan-selatan'
    
    # Path Output disesuaikan dengan nama asset
    OUT_DIR = 'src/results/inference/'
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_FILE = os.path.join(OUT_DIR, "sawit_pdl_inference.tif")

    # TARGET ASSET GEE
    TARGET_ASSET = 'projects/data-skripsi-473712/assets/sawit-pdl'

    geotiff_files = sorted(Path(IN_DIR).glob('*.tif'))
    if not geotiff_files:
        print(f"❌ GeoTIFF tidak ditemukan di {IN_DIR}")
        return

    print(f"\n{'='*60}")
    print(f"🚀 MEMULAI INFERENSI: SAWIT-PDL")
    print(f"{'='*60}")

    engine = GeoTIFFInferenceEngine(
        model_path=MODEL, 
        config_path=CONFIG, 
        normalization_stats_path=STATS, 
        asset_id=TARGET_ASSET
    )
    
    engine.infer_on_geotiffs([str(f) for f in geotiff_files], OUT_FILE)

if __name__ == '__main__':
    main()