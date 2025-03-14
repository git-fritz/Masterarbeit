# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:56:31 2025

@author: Felix
"""

import geopandas as gpd
import pandas as pd
# Define input file paths
gpkg1_path = r"E:\Thesis\seedlings_east_cut.gpkg"  # Replace with your first GPKG file
gpkg2_path =r"E:\Thesis\LiDea1-west_ortho.tif_Model25_drone1.5cm_preds_filt.gpkg"  # Replace with your second GPKG file
output_gpkg_path = r"E:\Thesis\merged_seedlings.gpkg"  # Output file

# Load the two GeoPackages
gdf1 = gpd.read_file(gpkg1_path)
gdf2 = gpd.read_file(gpkg2_path)

# Ensure both have the same CRS
if gdf1.crs != gdf2.crs:
    print("⚠️ CRS mismatch detected! Converting to match.")
    gdf2 = gdf2.to_crs(gdf1.crs)

# Merge the two datasets
merged_gdf = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True))

# Save to a new GPKG
merged_gdf.to_file(output_gpkg_path, driver="GPKG")

print(f"✅ Merged GPKG saved at: {output_gpkg_path}")

# %%

# add max height
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import box
from tqdm import tqdm  # ✅ Import tqdm for progress bar

# Define input file paths
merged_gpkg_path = r"E:\Thesis\merged_seedlings.gpkg"   # Merged GPKG file
chm_raster_path = r"E:\Thesis\data\CHM\merged_chm.tif"  # CHM raster file
output_gpkg_path = r"E:\Thesis\merged_seedlings_height.gpkg"   # Output GPKG file


# Load the merged GeoPackage
gdf = gpd.read_file(merged_gpkg_path)
print(f"✅ Loaded {len(gdf)} polygons from GPKG.")

# Open CHM raster to get bounds and CRS
with rasterio.open(chm_raster_path) as src:
    chm_bounds = box(*src.bounds)  # Create a polygon from raster bounds
    chm_crs = src.crs

# Ensure the GPKG and CHM raster have the same CRS
if gdf.crs != chm_crs:
    print("⚠️ CRS mismatch detected! Converting to match CHM raster.")
    gdf = gdf.to_crs(chm_crs)

# 🔹 Filter polygons that are inside the CHM raster bounds
gdf = gdf[gdf.geometry.intersects(chm_bounds)]
print(f"✅ {len(gdf)} polygons remain after filtering out-of-bounds shapes.")

# Compute max CHM height per polygon with a forced console-friendly progress bar
max_heights = []
tqdm_bar = tqdm(total=len(gdf), desc="Calculating max height", dynamic_ncols=True, leave=True, ascii=True)

for stat in zonal_stats(gdf, chm_raster_path, stats=["max"]):
    max_heights.append(stat["max"] if stat["max"] is not None else 0)
    tqdm_bar.update(1)  # ✅ Ensure progress bar updates correctly

tqdm_bar.close()  # ✅ Close progress bar properly

# Add max height values to the GeoDataFrame
gdf["max_height"] = max_heights

# Save the updated GeoPackage
gdf.to_file(output_gpkg_path, driver="GPKG")

print(f"✅ Updated GPKG saved with max_height column (filtered): {output_gpkg_path}")
# %%


import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import box
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

# Define input file paths
merged_gpkg_path = r"E:\Thesis\merged_seedlings.gpkg"   # Merged GPKG file
chm_raster_path = r"E:\Thesis\data\CHM\merged_chm.tif"  # CHM raster file
output_gpkg_path = r"E:\Thesis\merged_seedlings_height.gpkg"   # Output GPKG file

# Load GeoPackage
gdf = gpd.read_file(merged_gpkg_path)
print(f"✅ Loaded {len(gdf)} polygons from GPKG.")

# Open CHM raster to get bounds & CRS
with rasterio.open(chm_raster_path) as src:
    chm_bounds = box(*src.bounds)
    chm_crs = src.crs

# Ensure CRS match
if gdf.crs != chm_crs:
    print("⚠️ CRS mismatch detected! Converting to match CHM raster.")
    gdf = gdf.to_crs(chm_crs)

# Filter polygons inside CHM extent
gdf = gdf[gdf.geometry.intersects(chm_bounds)]
print(f"✅ {len(gdf)} polygons remain after filtering.")

# Function to process a batch of polygons
def process_batch(batch):
    return zonal_stats(batch, chm_raster_path, stats=["max"], all_touched=True)

# Split data into chunks
num_cores = max(1, cpu_count() - 2)  # Use all but 2 cores
batch_size = len(gdf) // num_cores
batches = [gdf.iloc[i:i+batch_size] for i in range(0, len(gdf), batch_size)]

# Run in parallel
print(f"⚡ Running zonal stats with {num_cores} cores...")
with Pool(num_cores) as pool:
    results = list(tqdm(pool.imap(process_batch, batches), total=len(batches), desc="Processing batches", unit="batch"))

# Flatten results
max_heights = [stat["max"] if stat and "max" in stat else 0 for batch in results for stat in batch]

# Add results to GeoDataFrame
gdf["max_height"] = max_heights

# Save output
gdf.to_file(output_gpkg_path, driver="GPKG")
print(f"✅ Updated GPKG saved: {output_gpkg_path}")
