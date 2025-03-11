"""
Created on Tue Feb 11 12:18:15 2025

@author: Felix
"""
# %%
# CODE TO CALCULATE ASPECT FROM DTM
# %%


import rasterio
import richdem as rd
import numpy as np

def calculate_aspect(input_raster, output_raster):
    with rasterio.open(input_raster) as src:
        dem = src.read(1)
        profile = src.profile

    dem_rd = rd.rdarray(dem, no_data=profile['nodata'])
    aspect = rd.TerrainAttribute(dem_rd, attrib='aspect')

    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(aspect.astype(np.float32), 1)

# Example usage in Spyder:
input_raster = r"E:\Thesis\data\DEM\merged_raster_clean9999.tif"
output_raster = r"E:\Thesis\testing\metrics\aspect.tif"
calculate_aspect(input_raster, output_raster)
