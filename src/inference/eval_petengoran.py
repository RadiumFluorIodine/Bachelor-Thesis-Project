"""
Field Validation for Hutan Mangrove Petengoran (GEE Integrated).

Membandingkan hasil prediksi AGB U-TAE dengan data in-situ dari paper referensi
menggunakan Asset Region-of-Interest (ROI) dari Google Earth Engine.
Format visualisasi dioptimalkan untuk kertas A4 dan plot terpisah.
"""

import sys
import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from typing import Dict, Tuple
import json
import logging
import rasterio
import rasterio.mask
import ee
from shapely.geometry import shape
import pandas as pd

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

from src.training.training_utils import RegressionMetrics

class PetengoranGEEValidation:
    def __init__(self, prediction_map_path: str):
        self.pred_map_path = Path(prediction_map_path)
        self.logger = logging.getLogger(__name__)
        
        if not self.pred_map_path.exists():
            raise FileNotFoundError(f"Peta prediksi tidak ditemukan: {self.pred_map_path}")
            
        # 1. Inisialisasi GEE
        self.logger.info("Menginisialisasi Google Earth Engine...")
        try:
            ee.Initialize()
        except Exception:
            ee.Authenticate()
            ee.Initialize()

        # 2. Hardcode Ground Truth dari Paper (Tabel 8)
        # Zona C adalah total AGB dari R. apiculata, S. alba, dan R. mucronata
        self.zones_info = {
            'Zone-A': {'asset_id': 'projects/data-skripsi-473712/assets/Zone-A', 'agb_actual': 164.324},
            'Zone-B': {'asset_id': 'projects/data-skripsi-473712/assets/Zone-B', 'agb_actual': 160.381},
            'Zone-C': {'asset_id': 'projects/data-skripsi-473712/assets/Zone-C', 'agb_actual': 90.962},
            'Zone-D': {'asset_id': 'projects/data-skripsi-473712/assets/Zone-D', 'agb_actual': 272.030}
        }

    def load_gee_data(self) -> gpd.GeoDataFrame:
        """Mengunduh geometri Zona dari GEE dan mengubahnya ke GeoDataFrame."""
        self.logger.info("\n📥 Mengunduh batas poligon Zona dari GEE Asset...")
        features = []
        
        for zone_name, info in self.zones_info.items():
            try:
                # Panggil FeatureCollection dari GEE
                fc = ee.FeatureCollection(info['asset_id'])
                # Ekstrak koordinat GeoJSON
                geom_geojson = fc.geometry().getInfo()
                # Konversi ke geometri Shapely (Python)
                shapely_geom = shape(geom_geojson)
                
                features.append({
                    'Zone': zone_name,
                    'geometry': shapely_geom,
                    'agb_actual': info['agb_actual']
                })
                self.logger.info(f"   ✅ {zone_name} berhasil diunduh.")
            except Exception as e:
                self.logger.error(f"   ❌ Gagal memuat {zone_name}: {e}")
        
        # Buat GeoDataFrame dengan sistem koordinat standar WGS84 (Lat/Lon)
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        return gdf

    def extract_predictions(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Ekstraksi rata-rata piksel AGB prediksi yang jatuh di dalam poligon."""
        predicted_values = []
        
        with rasterio.open(self.pred_map_path) as src:
            # Samakan proyeksi jika berbeda (Misal dari LatLon WGS84 ke UTM raster)
            if gdf.crs != src.crs:
                self.logger.info(f"🔄 Menyelaraskan proyeksi koordinat dari {gdf.crs} ke {src.crs}...")
                gdf = gdf.to_crs(src.crs)
                
            self.logger.info("✂️ Mengekstrak nilai piksel di dalam poligon...")
            
            for idx, row in gdf.iterrows():
                geom = row.geometry
                
                # Jika inputnya berupa titik, kita beri radius 15 meter.
                # Jika sudah berupa poligon, kita pakai poligon aslinya.
                if geom.geom_type == 'Point':
                    geom = geom.buffer(15.0) 
                    
                try:
                    # Potong raster mengikuti bentuk poligon
                    out_image, _ = rasterio.mask.mask(src, [geom], crop=True)
                    out_image = out_image[0]
                    
                    # Ambil hanya piksel valid (bukan nol, bukan No-Data)
                    valid_pixels = out_image[(out_image != -9999.0) & (out_image > 0) & (~np.isnan(out_image))]
                    
                    if len(valid_pixels) > 0:
                        mean_val = np.mean(valid_pixels)
                        predicted_values.append(mean_val)
                        self.logger.info(f"   📍 {row['Zone']}: Rata-rata Prediksi = {mean_val:.2f} Mg/ha (dari {len(valid_pixels)} piksel)")
                    else:
                        self.logger.warning(f"   ⚠️ {row['Zone']}: Tidak ada piksel valid di dalam poligon.")
                        predicted_values.append(np.nan)
                        
                except ValueError:
                    self.logger.warning(f"   ❌ {row['Zone']}: Poligon berada di luar batas peta raster!")
                    predicted_values.append(np.nan)
                    
        return np.array(predicted_values, dtype=float)

    def validate(self, output_dir: str):
        """Menjalankan seluruh proses validasi per zona dan total, serta membuat plot."""
        # 1. Load Data
        gdf = self.load_gee_data()
        
        # 2. Extract Predictions
        pred_values = self.extract_predictions(gdf)
        meas_values = gdf['agb_actual'].values
        zones_list = gdf['Zone'].values
        
        # Filter NaN (jika ada zona yang meleset)
        valid_mask = ~(np.isnan(pred_values) | np.isnan(meas_values))
        pred_valid = pred_values[valid_mask]
        meas_valid = meas_values[valid_mask]
        zones_valid = zones_list[valid_mask]
        
        if len(pred_valid) == 0:
            self.logger.error("Tidak ada data yang valid untuk dibandingkan.")
            return
            
        # 3. ANALISIS PER ZONA (Dataframe)
        df_results = pd.DataFrame({
            'Zona': zones_valid,
            'AGB_Aktual_Paper': meas_valid,
            'AGB_Prediksi_UTAE': pred_valid
        })
        
        # Hitung Error per Zona
        df_results['Selisih_Error'] = df_results['AGB_Prediksi_UTAE'] - df_results['AGB_Aktual_Paper']
        df_results['Error_Persen'] = (df_results['Selisih_Error'] / df_results['AGB_Aktual_Paper']) * 100
        
        # Cetak Tabel Per Zona ke Terminal
        self.logger.info("\n" + "="*70)
        self.logger.info("Tabel Komparasi Per Zona (Mg/ha)")
        self.logger.info("="*70)
        self.logger.info(df_results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        self.logger.info("="*70)
        
        # Simpan ke CSV untuk Lampiran Skripsi
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'tabel_validasi_per_zona.csv'
        df_results.to_csv(csv_path, index=False)
        self.logger.info(f"💾 Tabel per zona disimpan ke: {csv_path}")

        # 4. ANALISIS TOTAL (Keseluruhan)
        metrics = RegressionMetrics.compute_all(pred_valid, meas_valid)
        bias_total = float(np.mean(pred_valid - meas_valid))
        
        self.logger.info("\n" + "="*50)
        self.logger.info("HASIL VALIDASI TOTAL (KESELURUHAN ZONA)")
        self.logger.info("="*50)
        self.logger.info(f"Rata-rata Aktual   : {np.mean(meas_valid):.2f} Mg/ha")
        self.logger.info(f"Rata-rata Prediksi : {np.mean(pred_valid):.2f} Mg/ha")
        self.logger.info(f"MAE Keseluruhan    : {metrics['mae']:.2f} Mg/ha")
        self.logger.info(f"RMSE Keseluruhan   : {metrics['rmse']:.2f} Mg/ha")
        self.logger.info(f"Bias Total         : {bias_total:+.2f} Mg/ha")
        self.logger.info("="*50)
        
        # Gabungkan data untuk plot
        metrics.update({'bias': bias_total})
        
        # 5. Buat Visualisasi (Terpisah)
        self._plot_validation(df_results, metrics, output_dir)
        
    def _plot_validation(self, df: pd.DataFrame, metrics: Dict, output_dir: Path):
        """Membuat dua grafik terpisah dengan standar A4."""
        self.logger.info("\n🎨 Membuat grafik visualisasi...")
        
        # --- PLOT 1: BAR CHART PER ZONA ---
        fig1, ax1 = plt.subplots(figsize=(5.5, 4.0)) # Ukuran pas untuk margin A4
        x = np.arange(len(df))
        width = 0.35
        
        rects1 = ax1.bar(x - width/2, df['AGB_Aktual_Paper'], width, label='Aktual (In-situ)', color='#55a868', edgecolor='black', linewidth=1)
        rects2 = ax1.bar(x + width/2, df['AGB_Prediksi_UTAE'], width, label='Prediksi (U-TAE)', color='#4c72b0', edgecolor='black', linewidth=1)
        
        # Label & Title
        ax1.set_ylabel('Above-Ground Biomass (Mg/ha)', fontsize=10, fontweight='normal')
        ax1.set_title('Aktual vs Prediksi per Zona (Petengoran)', fontsize=12, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Zona'], fontsize=10, fontweight='normal')
        
        # Penyesuaian Sumbu & Grid
        max_y = max(df['AGB_Aktual_Paper'].max(), df['AGB_Prediksi_UTAE'].max())
        ax1.set_ylim(0, max_y * 1.25) # Beri ruang untuk teks dan legend
        ax1.legend(fontsize=9, frameon=True, loc='upper right')
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        ax1.tick_params(axis='both', which='major', direction='in', labelsize=9)
        
        # Tambahkan label angka di atas batang
        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        autolabel(rects1, ax1)
        autolabel(rects2, ax1)

        plt.tight_layout()
        save_path_bar = output_dir / 'validasi_lapangan_barchart.png'
        fig1.savefig(save_path_bar, dpi=600, bbox_inches='tight')
        plt.close(fig1)

        # --- PLOT 2: SCATTER PLOT 1:1 ---
        fig2, ax2 = plt.subplots(figsize=(5.0, 5.0)) # Bujur sangkar untuk scatter plot 1:1
        meas = df['AGB_Aktual_Paper'].values
        pred = df['AGB_Prediksi_UTAE'].values
        zones = df['Zona'].values
        
        ax2.scatter(meas, pred, color='#4c72b0', s=100, edgecolors='black', alpha=0.8, zorder=5)
        
        # Anotasi teks untuk setiap titik zona
        for i, txt in enumerate(zones):
            ax2.annotate(txt, (meas[i], pred[i]), xytext=(8, -4), textcoords='offset points', 
                         fontsize=9, fontweight='normal', color='black')
            
        max_val = max(meas.max(), pred.max()) + 20
        ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='1:1 Ideal Line')
        
        if len(meas) > 1:
            z = np.polyfit(meas, pred, 1)
            p = np.poly1d(z)
            ax2.plot([0, max_val], p([0, max_val]), 'k-', linewidth=1.5, alpha=0.6, label='Trend Line')
            
        ax2.set_xlabel('Measured AGB In-situ (Mg/ha)', fontsize=10, fontweight='normal')
        ax2.set_ylabel('Predicted AGB U-TAE (Mg/ha)', fontsize=10, fontweight='normal')
        ax2.set_title('Field Validation Scatter Plot', fontsize=12, fontweight='bold', pad=15)
        
        ax2.set_xlim(0, max_val)
        ax2.set_ylim(0, max_val)
        
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.tick_params(axis='both', which='major', direction='in', labelsize=9)
        ax2.legend(loc='lower right', fontsize=9, frameon=False)
        
        # Box Metrik
        metrics_text = f"RMSE = {metrics['rmse']:.1f}\nMAE  = {metrics['mae']:.1f}\nBias = {metrics['bias']:+.1f}"
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray', lw=0.5)
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, verticalalignment='top', 
                 bbox=props, fontsize=10)
        
        plt.tight_layout()
        save_path_scatter = output_dir / 'validasi_lapangan_scatter.png'
        fig2.savefig(save_path_scatter, dpi=600, bbox_inches='tight')
        plt.close(fig2)
        
        self.logger.info(f"📸 Bar chart tersimpan di: {save_path_bar}")
        self.logger.info(f"📸 Scatter plot tersimpan di: {save_path_scatter}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-map', default='E:/SKRIPSI-RAFI RIDHO RAMADHAN/Bachelor-Thesis-Project-v.1.1/src/results\inference/agb_map_petengoran_2023.tif')
    parser.add_argument('--output-dir', default='src/results/field_validation')
    args = parser.parse_args()
    
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    validator = PetengoranGEEValidation(prediction_map_path=args.pred_map)
    validator.validate(output_dir=args.output_dir)

if __name__ == '__main__':
    sys.exit(main())