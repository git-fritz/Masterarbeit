# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:35:24 2025

@author: Felix
"""

# %%
# CODE TO CALULATE TPI AT DIFFRENT SCALES: 25, 35, 55
# %%


import numpy as np
import rasterio
from scipy.ndimage import generic_filter

def calculate_tpi_exclude_nodata(input_path, output_path, window_size):
    """
    Calculate the Topographic Position Index (TPI) for a given DTM, excluding NoData values.

    Parameters:
        input_path (str): Path to the input DTM GeoTIFF file.
        output_path (str): Path to save the resulting TPI GeoTIFF file.
        window_size (int): Size of the moving window (must be an odd number).
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")

    # Open the input GeoTIFF file
    with rasterio.open(input_path) as src:
        # Read the data and metadata
        data = src.read(1)
        meta = src.meta.copy()
        nodata_value = src.nodata

        if nodata_value is None:
            raise ValueError("Input file does not have a defined NoData value.")

        # Define a function to calculate TPI for a window
        def tpi_calculator(window):
            # Exclude NoData values from the calculation
            valid_values = window[window != nodata_value]
            if valid_values.size == 0:
                return nodata_value  # Return NoData value if the entire window is invalid
            center = valid_values[len(valid_values) // 2]
            mean = np.nanmean(valid_values)
            return center - mean

        # Apply the TPI calculation with the chosen window size
        tpi = generic_filter(data, tpi_calculator, size=window_size, mode='constant', cval=nodata_value)

        # Update metadata to match the TPI output
        meta.update({"dtype": "float32", "nodata": nodata_value})

        # Save the TPI result to a new GeoTIFF
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(tpi.astype(np.float32), 1)

if __name__ == "__main__":
    # Example usage
    input_file = r"E:\Thesis\data\DEM\merged_raster_clean9999.tif"
    output_file = r"E:\Thesis\data\merged_raster_clean9999_tpi399.tif"
    window_size = 399  # Example window size

    calculate_tpi_exclude_nodata(input_file, output_file, window_size)
    print(f"TPI calculated with window size {window_size} and saved to {output_file}")
