"""
End-to-End Pipeline: Inference & Field Validation for Hutan Mangrove Petengoran.

Alur Kerja:
1. INFERENSI: Membaca raw GeoTIFF Petengoran -> Proses U-TAE (Seluruh Area Tanpa Masking) -> Simpan Peta
2. VALIDASI: Download Polygon GEE -> Ekstrak rata-rata piksel Peta AGB -> Bandingkan dengan Aktual (Absolut)
Format visualisasi dioptimalkan untuk kertas A4 dan plot terpisah.
"""

import sys
import os
import torch
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import rasterio
import rasterio.mask
from rasterio.windows import Window
from shapely.geometry import shape
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
import json
import ee

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

from src.models.utae import UTAE
from src.training.training_utils import RegressionMetrics

# Konfigurasi GEE
PROJECT_ID = 'data-skripsi-473712'


class GeoTIFFInferenceEngine:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        normalization_stats_path: str,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger("InferenceEngine")
        
        self.config = self._load_config(config_path)
        self.norm_stats = self._load_normalization_stats(normalization_stats_path)
        self.model = self._load_model(model_path)
        
        self.tile_size = 128  
        self.overlap = 16  
        self.batch_size = 128  
        
        self.logger.info(f"✅ Inference Engine initialized (Device: {self.device})")
        self.logger.info("ℹ️ Mode Inferensi: Full Area (Tanpa ROI Masking)")

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

    def generate_tiles(self, raster_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        height, width = raster_shape
        stride = self.tile_size - self.overlap
        tiles = []
        for row in range(0, height, stride):
            for col in range(0, width, stride):
                h = min(self.tile_size, height - row)
                w = min(self.tile_size, width - col)
                if h < self.tile_size // 2 or w < self.tile_size // 2: continue
                # Tanpa pengecekan ROI, semua tile diproses
                tiles.append((row, col, h, w))
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
        return np.array(data_list) 

    def infer_on_geotiffs(self, geotiff_paths: List[str], output_path: str):
        with rasterio.open(geotiff_paths[0]) as src:
            meta = src.meta.copy()
            height, width = src.height, src.width

        self.logger.info(f"🔄 Melakukan inferensi pada seluruh area citra ({width}x{height} piksel)...")

        tile_coords = self.generate_tiles((height, width))
        output_array = np.zeros((height, width), dtype=np.float32)
        count_array = np.zeros((height, width), dtype=np.float32)

        batch_positions = torch.tensor([list(range(1, len(geotiff_paths)+1))], dtype=torch.long).to(self.device)
        
        buffer, coords_buffer = [], []
        for i, (r, c, h, w) in enumerate(tqdm(tile_coords, desc="GPU Processing", ncols=100)):
            tile_data = self.read_tile(geotiff_paths, r, c, h, w)
            buffer.append(self.normalize_tile(tile_data))
            coords_buffer.append((r, c, h, w))

            if len(buffer) == self.batch_size or i == len(tile_coords) - 1:
                batch_in = torch.from_numpy(np.array(buffer)).to(self.device).float()
                b_pos = batch_positions.repeat(len(buffer), 1)
                
                with torch.no_grad():
                    pred = self.model(batch_in, batch_positions=b_pos)['agb']
                    pred = torch.expm1((pred * self.norm_stats['agb_log_std']) + self.norm_stats['agb_log_mean'])
                    pred = pred.cpu().numpy()

                for j, (br, bc, bh, bw) in enumerate(coords_buffer):
                    p_tile = pred[j, 0, :bh, :bw]
                    target_area = output_array[br:br+bh, bc:bc+bw]
                    
                    # Langsung masukkan ke array karena tidak ada mask daratan/lautan
                    target_area[:] = np.where(count_array[br:br+bh, bc:bc+bw] == 0, 
                                              p_tile, 
                                              target_area + p_tile)
                    count_array[br:br+bh, bc:bc+bw] += 1.0
                buffer, coords_buffer = [], []

        # Rata-ratakan area yang overlapping (overlap blending)
        output_array[count_array > 1] /= count_array[count_array > 1]
        
        # Area yang sama sekali tidak tertutup (jika ada) di-set ke NoData
        output_array[count_array == 0] = -9999.0

        meta.update(dtype='float32', count=1, nodata=-9999.0, compress='lzw')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(output_array, 1)
        
        self.logger.info(f"✅ Prediksi selesai. Output: {output_path}")


class PetengoranGEEValidation:
    def __init__(self, prediction_map_path: str):
        self.pred_map_path = Path(prediction_map_path)
        self.logger = logging.getLogger("Validator")
        
        self.logger.info("Menginisialisasi Earth Engine untuk Validasi...")
        try:
            ee.Initialize(project=PROJECT_ID)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=PROJECT_ID)
            
        self.zones_info = {
            'Zone-A': {'asset_id': 'projects/data-skripsi-473712/assets/Zone-A', 'agb_actual': 164.324},
            'Zone-B': {'asset_id': 'projects/data-skripsi-473712/assets/Zone-B', 'agb_actual': 160.381},
            'Zone-C': {'asset_id': 'projects/data-skripsi-473712/assets/Zone-C', 'agb_actual': 90.962},
            'Zone-D': {'asset_id': 'projects/data-skripsi-473712/assets/Zone-D', 'agb_actual': 272.030}
        }

    def load_gee_data(self) -> gpd.GeoDataFrame:
        self.logger.info("📥 Mengunduh batas poligon Zona dari GEE Asset...")
        features = []
        for zone_name, info in self.zones_info.items():
            try:
                fc = ee.FeatureCollection(info['asset_id'])
                geom_geojson = fc.geometry().getInfo()
                shapely_geom = shape(geom_geojson)
                features.append({
                    'Zone': zone_name,
                    'geometry': shapely_geom,
                    'agb_actual': info['agb_actual']
                })
            except Exception as e:
                self.logger.error(f" ❌ Gagal memuat {zone_name}: {e}")
        return gpd.GeoDataFrame(features, crs="EPSG:4326")

    def extract_predictions(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        predicted_values = []
        with rasterio.open(self.pred_map_path) as src:
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
                
            self.logger.info("✂️ Mengekstrak nilai rata-rata piksel di dalam poligon...")
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == 'Point':
                    geom = geom.buffer(15.0) 
                    
                try:
                    out_image, _ = rasterio.mask.mask(src, [geom], crop=True)
                    out_image = out_image[0]
                    valid_pixels = out_image[(out_image != -9999.0) & (out_image > 0) & (~np.isnan(out_image))]
                    
                    if len(valid_pixels) > 0:
                        mean_val = np.mean(valid_pixels)
                        predicted_values.append(mean_val)
                    else:
                        predicted_values.append(np.nan)
                except ValueError:
                    predicted_values.append(np.nan)
        return np.array(predicted_values, dtype=float)

    def validate(self, output_dir: str):
        gdf = self.load_gee_data()
        pred_values = self.extract_predictions(gdf)
        meas_values = gdf['agb_actual'].values
        zones_list = gdf['Zone'].values
        
        valid_mask = ~(np.isnan(pred_values) | np.isnan(meas_values))
        pred_valid = pred_values[valid_mask]
        meas_valid = meas_values[valid_mask]
        zones_valid = zones_list[valid_mask]
        
        if len(pred_valid) == 0:
            self.logger.error("❌ Tidak ada data yang valid untuk dibandingkan.")
            return
            
        df_results = pd.DataFrame({
            'Zona': zones_valid,
            'AGB_Aktual_Paper': meas_valid,
            'AGB_Prediksi_UTAE': pred_valid
        })
        
        # Menghitung Absolute Error (Sama dengan MAE/RMSE per zona tunggal)
        df_results['Absolut_Error'] = np.abs(df_results['AGB_Prediksi_UTAE'] - df_results['AGB_Aktual_Paper'])
        df_results['Error_Persen'] = (df_results['Absolut_Error'] / df_results['AGB_Aktual_Paper']) * 100
        
        self.logger.info("\n" + "="*70)
        self.logger.info("TABEL KOMPARASI PER ZONA (Mg/ha)")
        self.logger.info("="*70)
        self.logger.info("\n" + df_results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        self.logger.info("="*70)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'tabel_validasi_per_zona.csv'
        df_results.to_csv(csv_path, index=False)
        
        metrics = RegressionMetrics.compute_all(pred_valid, meas_valid)
        bias_total = float(np.mean(pred_valid - meas_valid))
        metrics.update({'bias': bias_total})
        
        self.logger.info(f"\nMAE Keseluruhan  : {metrics['mae']:.2f} Mg/ha")
        self.logger.info(f"RMSE Keseluruhan : {metrics['rmse']:.2f} Mg/ha")
        
        self._plot_validation(df_results, metrics, output_dir)
        
    def _plot_validation(self, df: pd.DataFrame, metrics: Dict, output_dir: Path):
        """Membuat dua grafik terpisah dengan standar A4."""
        self.logger.info("\n🎨 Membuat grafik visualisasi terpisah...")
        
        # --- Fungsi Bantuan untuk Label Bar ---
        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        # ==========================================
        # Plot 1: Bar Chart Aktual vs Prediksi
        # ==========================================
        fig1, ax1 = plt.subplots(figsize=(5.5, 4.0)) # Ukuran pas untuk margin A4
        x = np.arange(len(df))
        width = 0.35
        
        rects1 = ax1.bar(x - width/2, df['AGB_Aktual_Paper'], width, label='Aktual (In-situ)', color='#55a868', edgecolor='black', linewidth=1)
        rects2 = ax1.bar(x + width/2, df['AGB_Prediksi_UTAE'], width, label='Prediksi (U-TAE)', color='#4c72b0', edgecolor='black', linewidth=1)
        
        # Label & Title (Font weight terpisah)
        ax1.set_ylabel('Above-Ground Biomass (Mg/ha)', fontsize=10, fontweight='normal')
        ax1.set_title('Komparasi AGB Aktual vs Prediksi per Zona', fontsize=12, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Zona'], fontsize=10, fontweight='normal')
        
        # Penyesuaian Sumbu & Grid
        max_y = max(df['AGB_Aktual_Paper'].max(), df['AGB_Prediksi_UTAE'].max())
        ax1.set_ylim(0, max_y * 1.25) # Beri ruang untuk legend
        ax1.legend(fontsize=9, frameon=True, loc='upper left')
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        ax1.tick_params(axis='both', which='major', direction='in', labelsize=9)
        
        autolabel(rects1, ax1)
        autolabel(rects2, ax1)

        plt.tight_layout()
        save_path_bar = output_dir / 'validasi_lapangan_barchart.png'
        fig1.savefig(save_path_bar, dpi=600, bbox_inches='tight')
        plt.close(fig1)

        # ==========================================
        # Plot 2: Bar Chart Absolute Error per Zona
        # ==========================================
        fig2, ax2 = plt.subplots(figsize=(5.5, 4.0))
        
        error_per_zone = df['Absolut_Error'].values
        rects3 = ax2.bar(x, error_per_zone, width=0.45, label='Absolute Error', color='#d62728', edgecolor='black', linewidth=1)
        
        ax2.set_ylabel('Besaran Error (Mg/ha)', fontsize=10, fontweight='normal')
        ax2.set_title('Distribusi Kesalahan (Error) per Zona', fontsize=12, fontweight='bold', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['Zona'], fontsize=10, fontweight='normal')
        
        max_err = error_per_zone.max()
        ax2.set_ylim(0, max_err * 1.25)
        ax2.grid(axis='y', linestyle='--', alpha=0.4)
        ax2.tick_params(axis='both', which='major', direction='in', labelsize=9)
        
        autolabel(rects3, ax2)
        
        # Kotak Metrik Keseluruhan 
        metrics_text = (
            "Metrik Keseluruhan:\n"
            f"RMSE = {metrics['rmse']:.2f} Mg/ha\n"
            f"MAE  = {metrics['mae']:.2f} Mg/ha\n"
            f"Bias = {metrics['bias']:+.2f} Mg/ha"
        )
        
        props = dict(boxstyle='square,pad=0.6', facecolor='white', alpha=1.0, edgecolor='black', linewidth=0.8)
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, 
                 verticalalignment='top', horizontalalignment='left',
                 bbox=props, fontsize=9, family='monospace')
        
        plt.tight_layout()
        save_path_err = output_dir / 'validasi_lapangan_error.png'
        fig2.savefig(save_path_err, dpi=600, bbox_inches='tight')
        plt.close(fig2)
        
        self.logger.info(f"📸 Bar chart (Aktual vs Prediksi) tersimpan di: {save_path_bar}")
        self.logger.info(f"📸 Bar chart (Distribusi Error) tersimpan di: {save_path_err}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Configuration
    MODEL = 'src/results/checkpoints/U-TAE/best_model.pt'
    CONFIG = 'src/results/checkpoints/U-TAE/config.json'
    STATS = 'data/processed/lampung/version_2/normalization.json'
    IN_DIR = 'data/raw/petengoran' 
    OUT_FILE = 'src/results/inference/agb_map_petengoran_2023.tif'
    VAL_OUT_DIR = 'src/results/field_validation'

    # Inference
    print("\n" + "="*50)
    print("🚀 FASE 1: INFERENSI PETA AGB MENGGUNAKAN U-TAE")
    print("="*50)
    geotiff_files = sorted(Path(IN_DIR).glob('*.tif'))
    if not geotiff_files:
        print(f"❌ GeoTIFF tidak ditemukan di direktori {IN_DIR}. Pastikan data mentah tersedia.")
        return

    engine = GeoTIFFInferenceEngine(MODEL, CONFIG, STATS)
    engine.infer_on_geotiffs([str(f) for f in geotiff_files], OUT_FILE)

    # Field Validation
    print("\n" + "="*50)
    print("📊 FASE 2: VALIDASI LAPANGAN (GOOGLE EARTH ENGINE)")
    print("="*50)
    
    validator = PetengoranGEEValidation(prediction_map_path=OUT_FILE)
    validator.validate(output_dir=VAL_OUT_DIR)

if __name__ == '__main__':
    main()