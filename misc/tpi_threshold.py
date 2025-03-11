# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:02:19 2025

@author: Felix
"""
# %%
# THIS CODE THRESHOLDS A RASTER BY A SET VALUE: IN THIS CASE SPECIFICALLY TO 
# THRESHOLD TPI LAYERS TO THEN BE POLYGONIZED LATER. THE TPI IS THRESHOLD IS 
# IS CHOSEN IN A MANNER THAT RESULTS IN THE BEST REPRESENTATION OF MICRO-
# TOPOGRAPHY (MOUNDS)
# %%



import rasterio
import numpy as np

# Input and output file paths
input_raster = r"E:\Thesis\data\DEM_TPI\merged_raster_clean9999_tpi35.tif"  # Change this to your actual file
output_raster = r"E:\Thesis\data\DEM_TPI\merged_raster_clean9999_tpi35_filter002.tif"  # Output file
threshold = 0.02   # Change this to your desired threshold

# Open the raster file
with rasterio.open(input_raster) as src:
    profile = src.profile  # Get metadata
    tpi_data = src.read(1)  # Read the first band

# Apply threshold: Keep values above the threshold, set others to NaN
filtered_tpi = np.where(tpi_data > threshold, tpi_data, np.nan)

# Ensure NaN values are written correctly by changing dtype
profile.update(dtype=rasterio.float32, nodata=np.nan)

# Save the new raster
with rasterio.open(output_raster, 'w', **profile) as dst:
    dst.write(filtered_tpi.astype(rasterio.float32), 1)

print(f"Filtered raster saved to {output_raster}")
