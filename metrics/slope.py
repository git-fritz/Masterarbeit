"""
Created on Tue Feb 11 12:04:17 2025

@author: Felix
"""
# %%
# CODE TO CALCULATE SLOPE FROM A DTM 
# %%


import rasterio
import richdem as rd
import numpy as np

def calculate_slope(input_raster, output_raster):
    # Load DEM
    with rasterio.open(input_raster) as src:
        dem = src.read(1)
        profile = src.profile

    # Compute slope
    dem_rd = rd.rdarray(dem, no_data=profile['nodata'])
    slope = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')

    # Save output
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(slope.astype(np.float32), 1)

# Example usage in Spyder:
input_raster = r"E:\Thesis\data\DEM\merged_raster_clean.tif"
output_raster = r"E:\Thesis\testing\metrics\slope_check.tif"
calculate_slope(input_raster, output_raster)
