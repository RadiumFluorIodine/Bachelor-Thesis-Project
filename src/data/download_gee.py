"""
Google Earth Engine Data Download Script
Downloads Sentinel-2 and ESA CCI AGB data for specified regions.
"""

import ee 
import geemap
import os
import sys
import json
from datetime import datetime
from tqdm import tqdm


# Configuration
"""
Asset list:
- Lampung: projects/data-skripsi-473712/assets/Lampung
- Kalimantan-selatan: projects/data-skripsi-473712/assets/Kalimantan-selatan
- Petengoran: projects/data-skripsi-473712/assets/Petengoran
"""
PROJECT_ID = 'data-skripsi-473712'
ASSET_ID = 'projects/data-skripsi-473712/assets/Petengoran'
OUTPUT_DIR = 'data/raw/petengoran'
REGION_NAME = 'Petengoran'

# Year
YEAR_S2 = 2023
YEAR_AGB = 2023

# Cloud threshold
CLOUD_THRESHOLD = 70
USE_CLOUD_MASK = False


def main ():
    # Authentification
    try:
        ee.Initialize(project=PROJECT_ID)
        print("✓ Connect to Earth Engine.")
    except Exception:
        print("Requires GEE authentication....")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
        print("✓ Authentication complete.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}")

    # Region of Interest
    roi = ee.FeatureCollection(ASSET_ID).geometry()
    print(f"✓ ROI loaded: {REGION_NAME}")

    # Download Data Sentinel-2
    def mask_s2(image):
        """Masking cloud, cloud shadow, and cirrus"""
        scl = image.select('SCL')
        # cloud shadow(3), cloud medium(8), cloud high(9), cirrus(10)
        mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        
        return image.updateMask(mask).select([
            'B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'
        ])
    
    def select_bands_only(image):
        """Only select bands without masking"""
        return image.select([
            'B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'
        ])


    def save_metadata(output_dir, region_name, year, cloud_threshold, ref_proj):
        """Save download metadata"""
        metadata = {
            'region': region_name,
            'year': year,
            'download_date': datetime.now().isoformat(),
            'sentinel2': {
                'bands': ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'],
                'cloud_threshold': cloud_threshold,
                'num_months': 12,
                'resolution': '10m',
                'compositing': 'median'
            },
            'agb': {
                'source': 'ESA CCI AGB v6.0',
                'year': year,
                'resolution': '10m',
                'resampling': 'bilinear'
            },
            'projection': str(ref_proj.getInfo())
        }
    
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
        print(f"✓ Metadata saved: {metadata_path}")

    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(f'{YEAR_S2}-01-01', f'{YEAR_S2}-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD))
    
    # Apply masking option
    if USE_CLOUD_MASK:
        s2_col = s2_col.map(mask_s2)
        print("✓ Cloud masking ENABLED.")
    else:
        s2_col = s2_col.map(select_bands_only)
        print("✓ Cloud masking DISABLED.")

    ref_proj = s2_col.first().select('B2').projection()
    print(f"✓ Reference projection extracted from Sentinel-2")

    print(f"Mulai download Sentinel-2 ({YEAR_S2}) - Region: {REGION_NAME}...")
    for m in tqdm(range(1, 13), desc="Downloading months"):
        start = ee.Date.fromYMD(YEAR_S2, m, 1)
        end = start.advance(1, 'month')

        # Check available image
        monthly_col = s2_col.filterDate(start, end)
        img_count = monthly_col.size().getInfo() 
        print(f"   [Info] Month {m}: {img_count} images passed the filter.")

        img = s2_col.filterDate(start, end).median().clip(roi)
        fname = os.path.join(OUTPUT_DIR, f"S2_{REGION_NAME}_M{m:02d}.tif")
        
        # Check if the file already exists
        if not os.path.exists(fname):
            print(f"   Processing Month {m}...")
            geemap.download_ee_image(
                img, fname, region=roi, crs=ref_proj, scale=10, 
                dtype='uint16', overwrite=True
            )
        else:
            print(f"   ✓ The month {m} already exists")

    
    
    # Data ESA CCI AGB v6.0
    print(f"Start downloading ESA CCI AGB v6.0 ({YEAR_AGB}) - Region: {REGION_NAME}...")


    # Use the Collection ID from the Community Catalog ()
    agb_col = ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")


    # Year Filter
    agb_img = agb_col.filterDate(f'{YEAR_AGB}-01-01', f'{YEAR_AGB}-12-31').first()

    # Check the data
    if agb_col is None:
        print("✗ ESA CCI data is not available for that year!")
        return
    
    # Notes:
    # Band 1 = AGB (Mg/ha)

    # Download Data ESA CCI AGB v6.0
    agb_selected = agb_img.select(['AGB'])
    agb_resampled = agb_selected \
        .resample('bilinear') \
        .reproject(crs=ref_proj, scale=10) \
        .clip(roi)
    

    out_agb_path = os.path.join(OUTPUT_DIR, f"AGB_{REGION_NAME}_Label.tif")
    geemap.download_ee_image(
        agb_resampled, 
        out_agb_path, 
        region=roi, 
        crs=ref_proj, 
        scale=10, 
        dtype='float32', 
        overwrite=True
    )
    print(f"✓ AGB Label successfully downloaded: {out_agb_path}")
    
    # Save metadata
    save_metadata(OUTPUT_DIR, REGION_NAME, 2022, CLOUD_THRESHOLD, ref_proj)

    # Summary of Results
    print("\n" + "-"*60)
    print("Ringkasan Hasil")
    print("-"*60)
    print(f"Region: {REGION_NAME}")
    print(f"Year: {YEAR_S2}")
    print(f"Sentinel-2: 10 bands × 12 months")
    print(f"ESA CCI AGB: 1 label band")
    print(f"Resolution: 10m × 10m")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-"*60)
    print("✓ Download complete!")
    print(f"Ready for preprocessing.")

if __name__ == "__main__":
    main()

