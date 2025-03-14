# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:03:29 2025

@author: Felix
"""

 
import rasterio
import numpy as np
from rasterio.windows import Window
from scipy.ndimage import generic_filter

def calculate_roughness(dtm_path, output_path, window_size=3):
    """
    Calculates surface roughness from a DTM using a moving window standard deviation.
    
    Args:
        dtm_path (str): Path to the input DTM (GeoTIFF).
        output_path (str): Path to save the roughness raster (GeoTIFF).
        window_size (int): Size of the moving window (e.g., 3 for 3x3 kernel).
    
    Returns:
        None
    """

    # Open DTM raster
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1, masked=True)  # Read first band, keeping NoData values
        profile = src.profile  # Save metadata

    # Convert NoData to NaN to avoid affecting calculations
    dtm = dtm.filled(np.nan)

    # Define a NaN-safe standard deviation function
    def nan_safe_std(values):
        """ Compute standard deviation, but return NaN if all values are NaN. """
        if np.all(np.isnan(values)):  # Check if all values in the window are NaN
            return np.nan
        return np.nanstd(values)

    # Apply moving window standard deviation
    roughness = generic_filter(dtm, nan_safe_std, size=window_size, mode='nearest')

    # Update metadata for the output raster
    profile.update(dtype=rasterio.float32, nodata=np.nan)

    # Save roughness raster
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(roughness.astype(rasterio.float32), 1)

    print(f"✅ Roughness raster saved to: {output_path}")

# Example usage
dtm_path = r"E:\Thesis\data\DEM\nDTM_clip.tif" # Replace with your DTM file path
output_path = r"E:\Thesis\data\DEM\roughness.tif"  # Replace with desired output path

calculate_roughness(dtm_path, output_path, window_size=3)
