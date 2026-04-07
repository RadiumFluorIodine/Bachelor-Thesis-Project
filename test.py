import numpy as np
import rasterio
from rasterio.windows import Window

# Ganti dengan path file Anda
s2_file = "data/raw/lampung/S2_Lampung_M05.tif" # Ambil bulan tengah
agb_file = "data/raw/lampung/AGB_Lampung_Label.tif"

# Ambil sample window kecil (misal di tengah gambar)
# Ganti col_off, row_off ke koordinat yang Anda curigai hitam
win = Window(col_off=5000, row_off=5000, width=128, height=128)

with rasterio.open(s2_file) as src_s2, rasterio.open(agb_file) as src_agb:
    s2_data = src_s2.read(window=win)
    agb_data = src_agb.read(1, window=win)

print("="*40)
print("🔍 DIAGNOSA DATA (128x128 Patch)")
print("="*40)

# 1. Cek Sentinel-2
print(f"Sentinel-2 (M05) Stats:")
print(f"  - Min: {s2_data.min()}, Max: {s2_data.max()}")
print(f"  - Apakah ada NaN? {np.isnan(s2_data).any()}")
print(f"  - Apakah ada 0? {np.any(s2_data == 0)}")
print(f"  - Jumlah Pixel Finite (Valid S2): {np.isfinite(s2_data).sum()} / {s2_data.size}")

# 2. Cek AGB
print(f"\nAGB Label Stats:")
print(f"  - Min: {agb_data.min()}, Max: {agb_data.max()}")
print(f"  - Apakah ada NaN? {np.isnan(agb_data).any()}")
print(f"  - Jumlah Pixel Valid AGB: {np.isfinite(agb_data).sum()}")

# 3. Simulasi Logika Mask
s2_valid = np.isfinite(s2_data).all(axis=0) # Asumsi 1 bulan ini saja
agb_valid = np.isfinite(agb_data) & (agb_data >= 0)
final_mask = s2_valid & agb_valid

print(f"\nFinal Mask Stats:")
print(f"  - Total Pixel Valid (Putih): {final_mask.sum()}")
print(f"  - Persentase Valid: {final_mask.sum() / final_mask.size * 100:.2f}%")

if final_mask.sum() == 0:
    print("\n❌ KESIMPULAN: Mask Hitam Total.")
    if np.isfinite(s2_data).sum() == 0:
        print("   -> Penyebab: Data Sentinel-2 Kosong/NaN.")
    elif np.isfinite(agb_data).sum() == 0:
        print("   -> Penyebab: Data AGB Label Kosong/NaN.")
    else:
        print("   -> Penyebab: Tidak ada irisan (S2 ada, AGB tidak ada, atau sebaliknya).")
else:
    print("\n✅ KESIMPULAN: Mask SEHARUSNYA TIDAK HITAM. Cek cara visualisasi Anda (dikali 255 belum?).")