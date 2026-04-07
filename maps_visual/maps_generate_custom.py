"""
Script untuk membuat Layout Peta Distribusi AGB dari file GeoTIFF.
Menghasilkan gambar (PNG/JPEG) beresolusi tinggi dengan standar publikasi jurnal akademik.
Fitur: Auto-Crop (Zoom In), Basemap Satelit, Landscape Layout, North Arrow.
"""

import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import contextily as cx  
from pathlib import Path

def create_agb_map(tif_path: str, output_path: str, title: str):
    print(f"📥 Membaca file raster: {tif_path}")
    
    if not os.path.exists(tif_path):
        print(f"❌ Error: File {tif_path} tidak ditemukan!")
        return

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        nodata = src.nodata
        crs = src.crs
        
        # Ekstrak resolusi dan transformasi untuk konversi indeks array ke koordinat peta
        transform = src.transform

    # 1. Masking nilai NoData
    if nodata is None:
        nodata = -9999.0
    
    data_masked = np.ma.masked_where((data == nodata) | (data <= 0) | np.isnan(data), data)

    if data_masked.count() == 0:
        print("❌ Error: Semua piksel bernilai NoData.")
        return

    # 2. AUTO-CROP (ZOOM IN) KE AREA ACTUAL
    # Mencari baris dan kolom yang memiliki data valid
    valid_rows, valid_cols = np.where(~data_masked.mask)
    
    # Ambil indeks batas ekstrim dari data yang valid
    min_row, max_row = np.min(valid_rows), np.max(valid_rows)
    min_col, max_col = np.min(valid_cols), np.max(valid_cols)
    
    # Berikan sedikit margin tambahan (padding) agar gambar tidak terlalu mentok ke garis tepi (misal 50 piksel)
    padding = 50
    min_row = max(0, min_row - padding)
    max_row = min(data.shape[0], max_row + padding)
    min_col = max(0, min_col - padding)
    max_col = min(data.shape[1], max_col + padding)

    # Konversi indeks baris/kolom ekstrem ke koordinat peta (Extent yang baru)
    x_min, y_max = transform * (min_col, min_row)  # Titik kiri atas (X_min, Y_max)
    x_max, y_min = transform * (max_col, max_row)  # Titik kanan bawah (X_max, Y_min)
    
    # Tentukan extent asli untuk seluruh raster (tetap digunakan untuk imshow)
    left, bottom, right, top = src.bounds
    full_extent = [left, right, bottom, top]

    # Rentang Warna
    vmin_val = np.nanpercentile(data_masked.filled(np.nan), 2)
    vmax_val = np.nanpercentile(data_masked.filled(np.nan), 98)

    print(f"🔍 Auto-Zoom Aktif: Koordinat dipotong ke area bervegetasi.")

    # 3. Setup Plot & Elemen Profesional
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white') 

    # Colormap
    cmap = plt.cm.YlGn.copy()
    cmap.set_bad(color='white', alpha=0.0) 

    # Plot Raster 
    im = ax.imshow(data_masked, cmap=cmap, extent=full_extent, vmin=vmin_val, vmax=vmax_val, zorder=2, alpha=0.85)

    # PERBAIKAN UTAMA: Batasi sumbu X dan Y hanya pada koordinat data yang ada (Zoom In)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 4. Menambahkan Basemap (Peta Latar)
    try:
        print("🗺️ Mengunduh basemap satelit resolusi tinggi...")
        cx.add_basemap(ax, crs=crs.to_string(), source=cx.providers.Esri.WorldImagery, alpha=0.7, zorder=1)
    except Exception as e:
        print(f"⚠️ Gagal memuat basemap: {e}")

    # 5. Colorbar (Legenda)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax, extend='both') 
    
    cbar.set_label('Above-Ground Biomass (Mg/ha)', fontsize=12, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=10, direction='in')
    cbar.outline.set_linewidth(1)

    # 6. Elemen Kartografi
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    # Style Tick & Grid Profesional
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in', length=6, width=1.2, zorder=3)
    ax.grid(color='white', linestyle=':', linewidth=0.8, alpha=0.5, zorder=3)

    # Memutar teks koordinat X
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_horizontalalignment('right')

    # Bingkai Peta
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        spine.set_zorder(4)

    # Menambahkan Arah Mata Angin
    x_arrow, y_arrow = 0.04, 0.90
    ax.annotate('N', xy=(x_arrow, y_arrow), xytext=(x_arrow, y_arrow - 0.08),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='black', width=4, headwidth=12, edgecolor='white', lw=1),
                ha='center', va='center', fontsize=16, fontweight='bold', zorder=5,
                bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    # 7. Simpan Hasil
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight') 
    print(f"✅ Peta berhasil disimpan di: {output_path}")
    plt.close()

if __name__ == '__main__':
    # Konfigurasi Path
    INPUT_TIFF = 'src/results/inference/sawit_pdl_inference.tif'  
    OUTPUT_PNG = 'src/results/maps/layout_peta_pdl.png'
    
    JUDUL_PETA = "Peta Distribusi Above-Ground Biomass\nPerkebunan Sawit PT. PDL 2022"

    create_agb_map(INPUT_TIFF, OUTPUT_PNG, JUDUL_PETA)