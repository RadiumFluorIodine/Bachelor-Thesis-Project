"""
Script untuk membuat Layout Peta Distribusi AGB dari file GeoTIFF.
Menghasilkan gambar (PNG/JPEG) beresolusi tinggi dengan standar publikasi jurnal akademik.
Fitur: Basemap Terrain, North Arrow, Graticule Profesional, Transparent NoData.
"""

import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from pathlib import Path
import contextily as cx  # Library untuk menambahkan basemap

def create_agb_map(tif_path: str, output_path: str, title: str):
    print(f"📥 Membaca file raster: {tif_path}")
    
    if not os.path.exists(tif_path):
        print(f"❌ Error: File {tif_path} tidak ditemukan!")
        return

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        nodata = src.nodata
        crs = src.crs
        left, bottom, right, top = src.bounds
        extent = [left, right, bottom, top]

    # 1. Masking nilai NoData agar Transparan
    if nodata is None:
        nodata = -9999.0
    
    data_masked = np.ma.masked_where((data == nodata) | (data <= 0) | np.isnan(data), data)

    if data_masked.count() == 0:
        print("❌ Error: Semua piksel bernilai NoData.")
        return

    # Menentukan batas warna (persentil 2 hingga 98 untuk kontras maksimal)
    vmin_val = np.nanpercentile(data_masked.filled(np.nan), 2)
    vmax_val = np.nanpercentile(data_masked.filled(np.nan), 98)

    print(f"📊 Rentang AGB asli: {np.nanmin(data_masked.filled(np.nan)):.2f} - {np.nanmax(data_masked.filled(np.nan)):.2f} Mg/ha")
    print(f"🎨 Rentang visualisasi diset pada: {vmin_val:.2f} - {vmax_val:.2f} Mg/ha")

    # 2. Setup Plot & Elemen Profesional
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('white') 

    # Colormap
    cmap = plt.cm.YlGn.copy()
    cmap.set_bad(color='white', alpha=0.0) # Membuat area NoData 100% transparan

    # Plot Raster (alpha=0.85 agar basemap sedikit tembus jika diperlukan)
    im = ax.imshow(data_masked, cmap=cmap, extent=extent, vmin=vmin_val, vmax=vmax_val, zorder=2)

    # 3. Menambahkan Basemap (Peta Latar)
    try:
        print("🗺️ Mengunduh basemap... (Pastikan ada koneksi internet)")
        # Menggunakan Esri World Terrain (cx.providers.Esri.WorldTerrain) (Bisa diganti ke cx.providers.Esri.WorldImagery untuk Satelit)
        cx.add_basemap(ax, crs=crs.to_string(), source=cx.providers.Esri.WorldImagery, alpha=0.7, zorder=1)
    except Exception as e:
        print(f"⚠️ Gagal memuat basemap: {e}")

    # 4. Colorbar (Legenda)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax, extend='both') 
    
    cbar.set_label('Above-Ground Biomass (Mg/ha)', fontsize=12, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=11, direction='in')
    cbar.outline.set_linewidth(1)
    

    # 5. Elemen Kartografi (Grid, Ticks, North Arrow)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    

    # Style Tick & Grid Profesional (Garis putus-putus tipis hitam)
    ax.tick_params(axis='both', which='major', labelsize=11, direction='in', length=6, width=1.2, zorder=3)
    ax.grid(color='black', linestyle=':', linewidth=0.6, alpha=0.6, zorder=3)

    # Bingkai Peta Tebal
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        spine.set_zorder(4)

    # Menambahkan Arah Mata Angin (North Arrow) di sudut kiri atas
    x_arrow, y_arrow = 0.06, 0.94
    ax.annotate('N', xy=(x_arrow, y_arrow), xytext=(x_arrow, y_arrow - 0.07),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='black', width=4, headwidth=12, edgecolor='white', lw=1),
                ha='center', va='center', fontsize=18, fontweight='bold', zorder=5,
                bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

    # 6. Simpan Hasil
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=900, bbox_inches='tight') 
    print(f"✅ Peta berhasil disimpan di: {output_path}")
    plt.close()

if __name__ == '__main__':
    # Konfigurasi Path
    INPUT_TIFF = 'src/results/inference/agb_map_lampung_2025.tif'  
    OUTPUT_PNG = 'src/results/maps/layout_peta_lampung2022.png'
    
    JUDUL_PETA = "Peta Distribusi Above-Ground Biomass Provinsi Lampung Tahun 2022"

    create_agb_map(INPUT_TIFF, OUTPUT_PNG, JUDUL_PETA)