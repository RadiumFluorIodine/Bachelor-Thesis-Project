"""
Script untuk membuat Layout Peta Distribusi AGB dari file GeoTIFF.
Menghasilkan gambar (PNG/JPEG) beresolusi tinggi dengan standar publikasi jurnal akademik.
Fitur: Basemap Satelit Resolusi Tinggi, North Arrow, Graticule Profesional, Transparent NoData.
"""

import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import contextily as cx

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
    # Menggunakan rasio Portrait agar sesuai dengan bentuk lahan
    fig, ax = plt.subplots(figsize=(8, 11))
    fig.patch.set_facecolor('white') 

    # Colormap
    cmap = plt.cm.YlGn.copy()
    cmap.set_bad(color='white', alpha=0.0) # Area NoData 100% transparan

    # Plot Raster (Alpha 0.8 agar citra satelit di belakangnya sedikit terlihat)
    im = ax.imshow(data_masked, cmap=cmap, extent=extent, vmin=vmin_val, vmax=vmax_val, zorder=2, alpha=0.85)

    # Memaksa batas X dan Y agar tepat seukuran raster (menghilangkan ruang putih sisa)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # 3. Menambahkan Basemap (Peta Latar)
    try:
        print("🗺️ Mengunduh basemap satelit resolusi tinggi...")
        # PERBAIKAN: Menggunakan WorldImagery yang mendukung zoom hingga level 19
        cx.add_basemap(ax, crs=crs.to_string(), source=cx.providers.Esri.WorldImagery, alpha=0.6, zorder=1)
    except Exception as e:
        print(f"⚠️ Gagal memuat basemap: {e}")

    # 4. Colorbar (Legenda)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax, extend='both') 
    
    cbar.set_label('Above-Ground Biomass (Mg/ha)', fontsize=12, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=10, direction='in')
    cbar.outline.set_linewidth(1)

    # 5. Elemen Kartografi (Grid, Ticks, North Arrow)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Style Tick & Grid Profesional
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in', length=6, width=1.2, zorder=3)
    ax.grid(color='white', linestyle=':', linewidth=0.8, alpha=0.5, zorder=3) # Grid diubah putih agar kontras dengan satelit

    # Memutar teks koordinat X agar tidak bertumpuk
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment('right')

    # Bingkai Peta Tebal
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        spine.set_zorder(4)

    # Menambahkan Arah Mata Angin (North Arrow)
    x_arrow, y_arrow = 0.08, 0.92
    ax.annotate('N', xy=(x_arrow, y_arrow), xytext=(x_arrow, y_arrow - 0.05),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='black', width=4, headwidth=12, edgecolor='white', lw=1),
                ha='center', va='center', fontsize=18, fontweight='bold', zorder=5,
                bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    # 6. Simpan Hasil
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight') 
    print(f"✅ Peta berhasil disimpan di: {output_path}")
    plt.close()

if __name__ == '__main__':
    # Konfigurasi Path
    INPUT_TIFF = 'src/results/inference/petengoran2023_all_zones.tif'  
    OUTPUT_PNG = 'src/results/maps/layout_peta_petengoran2023.png'
    
    # Gunakan \n untuk merapikan judul yang panjang
    JUDUL_PETA = "Peta Distribusi Above-Ground Biomass\nMangrove Petengoran 2023"

    create_agb_map(INPUT_TIFF, OUTPUT_PNG, JUDUL_PETA)