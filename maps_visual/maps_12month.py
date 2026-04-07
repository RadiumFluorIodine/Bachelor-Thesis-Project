"""
Plot citra Sentinel-2 (RGB) dari file GeoTIFF dan simpan ke PNG.

Output:
1) S2_Lampung_RGB_12_bulan.png  -> gabungan 12 bulan (Januari-Desember)
2) AGB_Lampung_Label.png        -> label terpisah

Catatan:
- Sentinel-2 RGB umumnya memakai band 4, 3, 2 (B04, B03, B02).
- Jika urutan band di file Anda berbeda, ubah RGB_BANDS di bawah.
"""

from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from tqdm import tqdm
from matplotlib import rcParams

# ==========================================
# TYPOGRAPHY STANDAR JURNAL
# ==========================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.linewidth'] = 1.0

rcParams['axes.titlesize'] = 10
rcParams['figure.titlesize'] = 14
rcParams['axes.titleweight'] = 'bold'

rcParams['savefig.dpi'] = 900
rcParams['figure.dpi'] = 600


# =========================
# PATH
# =========================
INPUT_DIR = Path(r"E:/SKRIPSI-RAFI RIDHO RAMADHAN/Bachelor-Thesis-Project-v.1.1/data/raw/lampung")
OUTPUT_DIR = Path(r"E:/SKRIPSI-RAFI RIDHO RAMADHAN/Bachelor-Thesis-Project-v.1.1/src/results/visualizations/maps")

MONTH_FILES = [
    ("Januari","S2_Lampung_M01.tif"),
    ("Februari","S2_Lampung_M02.tif"),
    ("Maret","S2_Lampung_M03.tif"),
    ("April","S2_Lampung_M04.tif"),
    ("Mei","S2_Lampung_M05.tif"),
    ("Juni","S2_Lampung_M06.tif"),
    ("Juli","S2_Lampung_M07.tif"),
    ("Agustus","S2_Lampung_M08.tif"),
    ("September","S2_Lampung_M09.tif"),
    ("Oktober","S2_Lampung_M10.tif"),
    ("November","S2_Lampung_M11.tif"),
    ("Desember","S2_Lampung_M12.tif"),
]

AGB_FILE = "AGB_Lampung_Label.tif"
MAX_DISPLAY_SIZE = 1200


# =========================
# HELPER
# =========================
def stretch_band(band, pmin=2, pmax=98):
    band = band.astype(np.float32)
    valid = np.isfinite(band)

    if not np.any(valid):
        return np.zeros_like(band)

    vmin, vmax = np.percentile(band[valid], (pmin, pmax))
    return np.clip((band - vmin) / (vmax - vmin + 1e-6), 0, 1)


def read_rgb_preview(path):
    with rasterio.open(path) as src:
        scale = max(src.width / MAX_DISPLAY_SIZE, src.height / MAX_DISPLAY_SIZE, 1)
        w, h = int(src.width/scale), int(src.height/scale)

        data = src.read(
            indexes=[3,2,1],
            out_shape=(3,h,w),
            resampling=Resampling.bilinear
        )

    img = np.dstack([
        stretch_band(data[0]),
        stretch_band(data[1]),
        stretch_band(data[2])
    ])

    return img


def read_agb(path):
    with rasterio.open(path) as src:
        scale = max(src.width / MAX_DISPLAY_SIZE, src.height / MAX_DISPLAY_SIZE, 1)
        w, h = int(src.width/scale), int(src.height/scale)

        return src.read(1, out_shape=(h,w), resampling=Resampling.bilinear)


def save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR/name, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR/name}")


# =========================
# PLOT 12 BULAN (4x3)
# =========================
def plot_12():
    fig, axes = plt.subplots(3, 4, figsize=(11.69, 7.5))
    axes = axes.ravel()

    for i,(m,f) in enumerate(tqdm(MONTH_FILES)):
        img = read_rgb_preview(INPUT_DIR/f)

        axes[i].imshow(img)
        axes[i].set_title(m, pad=4)
        axes[i].axis("off")

    fig.suptitle("Citra Sentinel-2 RGB - Lampung 2022")

    plt.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.90,
        bottom=0.05,
        wspace=0.03,
        hspace=0.18
    )

    save(fig, "S2_Lampung_RGB_12_bulan.png")


# =========================
# PLOT AGB
# =========================
def plot_agb():
    img = read_agb(INPUT_DIR/AGB_FILE)

    fig, ax = plt.subplots(figsize=(11.69, 7))
    im = ax.imshow(img, cmap="viridis")

    ax.set_title("AGB Lampung Label")
    ax.axis("off")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    save(fig, "AGB_Lampung_Label.png")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Start plotting...")
    plot_12()
    plot_agb()
    print("Done.")