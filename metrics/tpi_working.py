# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:35:24 2025

@author: Felix
"""
# %%
# import os
# os.environ["TEMP"] = "D:\\SpyderTemp"
# os.environ["TMP"] = "D:\\SpyderTemp"

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
    input_file = r"E:\Thesis\data\DEM\nDTM_clip.tif"
    output_file = r"E:\Thesis\data\DEM_TPI\tpi55\tpi55_ndtm\tpi55_ndtm.tif"
    window_size = 55  # Example window size

    calculate_tpi_exclude_nodata(input_file, output_file, window_size)
    print(f"TPI calculated with window size {window_size} and saved to {output_file}")


# %% multiprocessing

import numpy as np
import rasterio
from scipy.ndimage import generic_filter
from joblib import Parallel, delayed, cpu_count

def tpi_calculator(window, nodata_value):
    """Compute TPI for a given window, ignoring NoData values."""
    valid_values = window[window != nodata_value]
    if valid_values.size == 0:
        return nodata_value  # Return NoData if the entire window is invalid
    center = valid_values[len(valid_values) // 2]
    mean = np.nanmean(valid_values)
    return center - mean

def parallel_generic_filter(data, window_size, nodata_value):
    """Apply `generic_filter` in parallel across rows."""
    def process_row(row):
        return generic_filter(row, tpi_calculator, size=window_size, mode='constant', cval=nodata_value, extra_arguments=(nodata_value,))

    num_cores = cpu_count()
    print(f"Using {num_cores} cores for processing...")
    
    # Apply function in parallel to each row
    tpi_result = Parallel(n_jobs=num_cores)(delayed(process_row)(row) for row in data)
    return np.array(tpi_result)

def calculate_tpi_parallel(input_path, output_path, window_size):
    """
    Calculate the Topographic Position Index (TPI) in parallel using joblib.

    Parameters:
        input_path (str): Path to the input DTM GeoTIFF file.
        output_path (str): Path to save the resulting TPI GeoTIFF file.
        window_size (int): Size of the moving window (must be an odd number).
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")

    with rasterio.open(input_path) as src:
        data = src.read(1)
        meta = src.meta.copy()
        nodata_value = src.nodata

        if nodata_value is None:
            raise ValueError("Input file does not have a defined NoData value.")

        # Compute TPI using all cores
        tpi_result = parallel_generic_filter(data, window_size, nodata_value)

        # Update metadata and save output
        meta.update({"dtype": "float32", "nodata": nodata_value})
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(tpi_result.astype(np.float32), 1)

if __name__ == "__main__":
    # Example usage
    input_file = r"E:\Thesis\data\DEM\nDTM_clip.tif"
    output_file = r"E:\Thesis\data\DEM_TPI\tpi35\tpi35_ndtm\tpi35_ndtm.tif"
    window_size = 35  # Example window size

    calculate_tpi_parallel(input_file, output_file, window_size)
    print(f"✅ TPI calculated in parallel with window size {window_size} and saved to {output_file}")

# %%




import numpy as np
import rasterio
from scipy.ndimage import generic_filter

def calculate_tpi_exclude_nodata(input_path, output_paths, window_sizes):
    """
    Calculate the Topographic Position Index (TPI) at multiple scales, excluding NoData values.

    Parameters:
        input_path (str): Path to the input DTM GeoTIFF file.
        output_paths (dict): Dictionary mapping window sizes to output file paths.
        window_sizes (list): List of window sizes to compute TPI (must be odd numbers).
    """
    for size in window_sizes:
        if size % 2 == 0:
            raise ValueError(f"Window size {size} must be an odd number.")

    # Open the input GeoTIFF file
    with rasterio.open(input_path) as src:
        # Read the data and metadata
        data = src.read(1)
        meta = src.meta.copy()
        nodata_value = src.nodata

        if nodata_value is None:
            raise ValueError("Input file does not have a defined NoData value.")

        # Function to calculate TPI for a window
        def tpi_calculator(window):
            valid_values = window[window != nodata_value]  # Exclude NoData values
            if valid_values.size == 0:
                return nodata_value  # Return NoData value if the entire window is invalid
            center = valid_values[len(valid_values) // 2]
            mean = np.nanmean(valid_values)
            return center - mean

        # Compute and save TPI for each window size
        for size in window_sizes:
            print(f"Processing TPI for window size {size}...")
            tpi = generic_filter(data, tpi_calculator, size=size, mode='constant', cval=nodata_value)

            # Update metadata
            meta.update({"dtype": "float32", "nodata": nodata_value})

            # Save TPI raster
            with rasterio.open(output_paths[size], "w", **meta) as dst:
                dst.write(tpi.astype(np.float32), 1)

            print(f"✅ TPI for window size {size} saved to: {output_paths[size]}")

if __name__ == "__main__":
    # Example usage
    input_file = r"E:\Thesis\data\DEM\output\merged_raster_clean9999_nDTM.tif"
    
    output_files = {
        25: r"E:\Thesis\data\DEM_TPI\tpi25\tpi25_ndtm.tif",
        35: r"E:\Thesis\data\DEM_TPI\tpi35\tpi35_ndtm.tif",
        55: r"E:\Thesis\data\DEM_TPI\tpi55\tpi55_ndtm.tif"
    }
    
    window_sizes = [25, 35, 55]

    calculate_tpi_exclude_nodata(input_file, output_files, window_sizes)
    print("🎉 All TPI calculations completed successfully!")