# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:57:11 2025

@author: Felix
"""
import rasterio
import numpy as np

def normalize_raster_with_nan(input_raster, output_raster, new_nodata=np.nan):
    try:
        with rasterio.open(input_raster) as src:
            # Read the raster data
            data = src.read(1)
            nodata = src.nodata
            
            # Ensure NoData values are correctly masked
            mask = np.isnan(data)  # Explicitly identify NaN values
            
            # Debugging: Check mask
            print(f"Number of masked (NoData) pixels: {mask.sum()}")
            print(f"Total pixels: {data.size}")
            print(f"Valid pixels: {data.size - mask.sum()}")
            
            # Ensure there is valid data for normalization
            valid_data = data[~mask]
            if valid_data.size == 0:
                raise ValueError("No valid data found in the raster for normalization.")
            
            # Calculate Min and Max using nan-safe functions
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)
            print(f"Min Value: {min_val}, Max Value: {max_val}")
            
            # Normalize valid data
            normalized_data = np.copy(data)
            normalized_data[~mask] = (data[~mask] - min_val) / (max_val - min_val)
            
            # Set NoData values back
            if np.isnan(new_nodata):
                normalized_data[mask] = np.nan
            else:
                normalized_data[mask] = new_nodata
            
            # Update metadata
            metadata = src.meta
            metadata.update(dtype='float32', nodata=new_nodata)
            
            # Save the normalized raster
            with rasterio.open(output_raster, 'w', **metadata) as dest:
                dest.write(normalized_data, 1)
        
        print(f"Normalized raster saved to: {output_raster}")
    except Exception as e:
        print(f"Error: {e}")

# Example Usage
input_raster = r"E:\BERA\BERA_Aufenthalt\LiDEA_Pilot\0000_Results\XY_Results\Aug\LideaPilot-L1-ROADWEST-2\17_Raster\dtm.tif"
output_raster = r"E:\BERA\BERA_Aufenthalt\LiDEA_Pilot\0000_Results\XY_Results\Aug\LideaPilot-L1-ROADWEST-2\17_Raster\dtm_norm.tif"

normalize_raster_with_nan(input_raster, output_raster, new_nodata=np.nan)

# "E:\BERA\BERA_Aufenthalt\LiDEA_Pilot\0000_Results\XY_Results\Aug\LideaPilot-L1-ROADWEST-4\17_Raster\dtm.tif"
#%%
# WORKING SCRIPT

import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.transform import Affine
from scipy.ndimage import sobel, uniform_filter
import os
from tqdm import tqdm

def process_chunk(dtm_path, window):
    with rasterio.open(dtm_path) as src:
        dem = src.read(1, window=window)
        print(f"Processing window at {window}")

        slope = calculate_slope(dem)
        print("Slope calculation done")

        aspect = calculate_aspect(dem)
        print("Aspect calculation done")

        profile_curvature = calculate_profile_curvature(dem)
        print("Profile curvature calculation done")

        planform_curvature = calculate_planform_curvature(dem)
        print("Planform curvature calculation done")

        roughness_small = calculate_roughness_window(dem, size=3)
        print("Small-scale roughness calculation done")

        roughness_medium = calculate_roughness_window(dem, size=10)
        print("Medium-scale roughness calculation done")

        roughness_large = calculate_roughness_window(dem, size=50)
        print("Large-scale roughness calculation done")

        tpi_small = calculate_tpi_window(dem, size=3)
        print("Small-scale TPI calculation done")

        tpi_medium = calculate_tpi_window(dem, size=10)
        print("Medium-scale TPI calculation done")

        tpi_large = calculate_tpi_window(dem, size=50)
        print("Large-scale TPI calculation done")

        tri_small = calculate_tri_window(dem, size=3)
        print("Small-scale TRI calculation done")

        tri_medium = calculate_tri_window(dem, size=10)
        print("Medium-scale TRI calculation done")

        tri_large = calculate_tri_window(dem, size=50)
        print("Large-scale TRI calculation done")

        return slope, aspect, profile_curvature, planform_curvature, roughness_small, roughness_medium, roughness_large, tpi_small, tpi_medium, tpi_large, tri_small, tri_medium, tri_large

def calculate_slope(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    return np.sqrt(dzdx**2 + dzdy**2)

def calculate_aspect(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    return np.arctan2(dzdy, dzdx) * 180 / np.pi

def calculate_profile_curvature(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    dzdxx = sobel(dzdx, axis=1)
    dzdyy = sobel(dzdy, axis=0)
    return dzdxx + dzdyy

def calculate_planform_curvature(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    dzdxx = sobel(dzdx, axis=1)
    dzdyy = sobel(dzdy, axis=0)
    return dzdxx - dzdyy

def calculate_roughness_window(dem, size):
    mean_elevation = uniform_filter(dem, size=size)
    elevation_diff = dem - mean_elevation
    return uniform_filter(elevation_diff**2, size=size)**0.5

def calculate_tpi_window(dem, size):
    mean_elevation = uniform_filter(dem, size=size)
    return dem - mean_elevation

def calculate_tri_window(dem, size):
    mean_elevation = uniform_filter(dem, size=size)
    elevation_diff = dem - mean_elevation
    return uniform_filter(np.abs(elevation_diff), size=size)

if __name__ == "__main__":
    dtm_path = r"E:\Thesis\merge\merged_raster9999.tif"
    output_dir = r"E:\Thesis\testing\output_metrics\merge_metrics"
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(dtm_path) as src:
        meta = src.meta.copy()
        tile_size = 256  # Reduced tile size to lower memory usage
        width, height = src.width, src.height
        windows = [Window(j, i, min(tile_size, width - j), min(tile_size, height - i))
                   for i in range(0, height, tile_size)
                   for j in range(0, width, tile_size)]

        results = []
        for window in tqdm(windows, desc="Processing chunks"):
            result = process_chunk(dtm_path, window)
            results.append(result)

        # Save outputs
        metrics = [
            "slope", "aspect", "profile_curvature", "planform_curvature", 
            "roughness_small", "roughness_medium", "roughness_large", 
            "tpi_small", "tpi_medium", "tpi_large", 
            "tri_small", "tri_medium", "tri_large"
        ]
        for i, metric in enumerate(metrics):
            output_path = os.path.join(output_dir, f"{metric}.tif")
            meta.update(dtype="float32")

            # Reconstruct full array for the metric
            full_array = np.zeros((height, width), dtype="float32")
            for result, window in zip(results, windows):
                data = result[i]
                row_start = int(window.row_off)
                row_end = row_start + data.shape[0]
                col_start = int(window.col_off)
                col_end = col_start + data.shape[1]
                full_array[row_start:row_end, col_start:col_end] = data

            # Write to the output file
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(full_array, 1)

        print("All metrics have been successfully calculated and saved.")

#%%

## SAME SCRIPT BUT WITHOUT CHUNKING
import rasterio
import numpy as np
from scipy.ndimage import sobel, uniform_filter
import os

def process_dtm(dtm_path):
    """
    Process the entire DTM, calculating various metrics.

    Parameters:
        dtm_path (str): Path to the input DTM file.

    Returns:
        results (dict): Dictionary of calculated metrics.
    """
    with rasterio.open(dtm_path) as src:
        dem = src.read(1)  # Read the entire DTM

        # Calculate metrics
        results = {
            "slope": calculate_slope(dem),
            "aspect": calculate_aspect(dem),
            "profile_curvature": calculate_profile_curvature(dem),
            "planform_curvature": calculate_planform_curvature(dem),
            "roughness_small": calculate_roughness_window(dem, size=3),
            "roughness_medium": calculate_roughness_window(dem, size=10),
            "roughness_large": calculate_roughness_window(dem, size=50),
            "tpi_small": calculate_tpi_window(dem, size=3),
            "tpi_medium": calculate_tpi_window(dem, size=10),
            "tpi_large": calculate_tpi_window(dem, size=50),
            "tri_small": calculate_tri_window(dem, size=3),
            "tri_medium": calculate_tri_window(dem, size=10),
            "tri_large": calculate_tri_window(dem, size=50)
        }

    return results

def calculate_slope(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    return np.sqrt(dzdx**2 + dzdy**2)

def calculate_aspect(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    return np.arctan2(dzdy, dzdx) * 180 / np.pi

def calculate_profile_curvature(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    dzdxx = sobel(dzdx, axis=1)
    dzdyy = sobel(dzdy, axis=0)
    return dzdxx + dzdyy

def calculate_planform_curvature(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    dzdxx = sobel(dzdx, axis=1)
    dzdyy = sobel(dzdy, axis=0)
    return dzdxx - dzdyy

def calculate_roughness_window(dem, size):
    mean_elevation = uniform_filter(dem, size=size)
    elevation_diff = dem - mean_elevation
    return uniform_filter(elevation_diff**2, size=size)**0.5

def calculate_tpi_window(dem, size):
    mean_elevation = uniform_filter(dem, size=size)
    return dem - mean_elevation

def calculate_tri_window(dem, size):
    mean_elevation = uniform_filter(dem, size=size)
    elevation_diff = dem - mean_elevation
    return uniform_filter(np.abs(elevation_diff), size=size)

if __name__ == "__main__":
    dtm_path = r"E:\Thesis\merge\merged_raster9999.tif"
    output_dir = r"E:\Thesis\testing\output_metrics\merge_metrics"
    os.makedirs(output_dir, exist_ok=True)

    results = process_dtm(dtm_path)

    # Save outputs
    metrics = [
        "slope", "aspect", "profile_curvature", "planform_curvature", 
        "roughness_small", "roughness_medium", "roughness_large", 
        "tpi_small", "tpi_medium", "tpi_large", 
        "tri_small", "tri_medium", "tri_large"
    ]

    with rasterio.open(dtm_path) as src:
        meta = src.meta.copy()
        meta.update(dtype="float32")

    for metric in metrics:
        output_path = os.path.join(output_dir, f"{metric}.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(results[metric].astype("float32"), 1)

    print("All metrics have been successfully calculated and saved.")

#%%

import psutil

print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")

#%%

import rasterio
import numpy as np
from scipy.ndimage import sobel, uniform_filter
import os
from joblib import Parallel, delayed

def process_dtm_parallel(dtm_path, metrics, n_jobs=-1):
    """
    Process the DTM in parallel, calculating various metrics.

    Parameters:
        dtm_path (str): Path to the input DTM file.
        metrics (list): List of metric calculation functions.
        n_jobs (int): Number of parallel jobs (-1 uses all available cores).

    Returns:
        results (dict): Dictionary of calculated metrics.
    """
    with rasterio.open(dtm_path) as src:
        dem = src.read(1)  # Read the entire DTM

    # Parallelize the metric calculations
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_metric)(metric, dem) for metric in metrics
    )

    return {name: result for name, result in results}

def calculate_metric(metric, dem):
    """
    Helper function to calculate a metric and return its name and result.

    Parameters:
        metric (tuple): A tuple containing the metric name and function.
        dem (ndarray): DEM array.

    Returns:
        tuple: Metric name and calculated result.
    """
    name, func = metric
    return name, func(dem)

# Metric calculation functions
def calculate_slope(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    return np.sqrt(dzdx**2 + dzdy**2)

def calculate_aspect(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    return np.arctan2(dzdy, dzdx) * 180 / np.pi

def calculate_profile_curvature(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    dzdxx = sobel(dzdx, axis=1)
    dzdyy = sobel(dzdy, axis=0)
    return dzdxx + dzdyy

def calculate_planform_curvature(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    dzdxx = sobel(dzdx, axis=1)
    dzdyy = sobel(dzdy, axis=0)
    return dzdxx - dzdyy

def calculate_roughness_window(dem, size):
    mean_elevation = uniform_filter(dem, size=size)
    elevation_diff = dem - mean_elevation
    return uniform_filter(elevation_diff**2, size=size)**0.5

def calculate_tpi_window(dem, size):
    mean_elevation = uniform_filter(dem, size=size)
    return dem - mean_elevation

def calculate_tri_window(dem, size):
    mean_elevation = uniform_filter(dem, size=size)
    elevation_diff = dem - mean_elevation
    return uniform_filter(np.abs(elevation_diff), size=size)

if __name__ == "__main__":
    dtm_path = r"E:\Thesis\merge\merged_raster9999.tif"
    output_dir = r"E:\Thesis\testing\output_metrics\merge_metrics"
    os.makedirs(output_dir, exist_ok=True)

    # Define the metrics to calculate
    metrics = [
        ("slope", calculate_slope),
        ("aspect", calculate_aspect),
        ("profile_curvature", calculate_profile_curvature),
        ("planform_curvature", calculate_planform_curvature),
        ("roughness_small", lambda dem: calculate_roughness_window(dem, size=3)),
        ("roughness_medium", lambda dem: calculate_roughness_window(dem, size=10)),
        ("roughness_large", lambda dem: calculate_roughness_window(dem, size=50)),
        ("tpi_small", lambda dem: calculate_tpi_window(dem, size=3)),
        ("tpi_medium", lambda dem: calculate_tpi_window(dem, size=10)),
        ("tpi_large", lambda dem: calculate_tpi_window(dem, size=50)),
        ("tri_small", lambda dem: calculate_tri_window(dem, size=3)),
        ("tri_medium", lambda dem: calculate_tri_window(dem, size=10)),
        ("tri_large", lambda dem: calculate_tri_window(dem, size=50))
    ]

    # Process the DTM in parallel
    results = process_dtm_parallel(dtm_path, metrics)

    # Save outputs
    with rasterio.open(dtm_path) as src:
        meta = src.meta.copy()
        meta.update(dtype="float32")

    for metric, data in results.items():
        output_path = os.path.join(output_dir, f"{metric}.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(data.astype("float32"), 1)

    print("All metrics have been successfully calculated and saved.")


#%% OVERLAPPING CHUNKS

import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.transform import Affine
from scipy.ndimage import sobel, uniform_filter
import os
from tqdm import tqdm

def process_chunk(dtm_path, window, overlap):
    with rasterio.open(dtm_path) as src:
        # Erweiterter Chunk mit Overlap lesen
        expanded_window = Window(
            max(0, window.col_off - overlap),
            max(0, window.row_off - overlap),
            min(src.width - window.col_off + overlap, window.width + 2 * overlap),
            min(src.height - window.row_off + overlap, window.height + 2 * overlap)
        )
        dem = src.read(1, window=expanded_window)

        # Metriken berechnen
        slope = calculate_slope(dem)
        aspect = calculate_aspect(dem)
        profile_curvature = calculate_profile_curvature(dem)
        planform_curvature = calculate_planform_curvature(dem)
        roughness_small = calculate_roughness_window(dem, size=3)
        roughness_medium = calculate_roughness_window(dem, size=10)
        roughness_large = calculate_roughness_window(dem, size=50)
        tpi_small = calculate_tpi_window(dem, size=3)
        tpi_medium = calculate_tpi_window(dem, size=10)
        tpi_large = calculate_tpi_window(dem, size=50)
        tri_small = calculate_tri_window(dem, size=3)
        tri_medium = calculate_tri_window(dem, size=10)
        tri_large = calculate_tri_window(dem, size=50)

        # Overlap abschneiden
        trimmed_slope = slope[overlap:-overlap, overlap:-overlap]
        trimmed_aspect = aspect[overlap:-overlap, overlap:-overlap]
        trimmed_profile_curvature = profile_curvature[overlap:-overlap, overlap:-overlap]
        trimmed_planform_curvature = planform_curvature[overlap:-overlap, overlap:-overlap]
        trimmed_roughness_small = roughness_small[overlap:-overlap, overlap:-overlap]
        trimmed_roughness_medium = roughness_medium[overlap:-overlap, overlap:-overlap]
        trimmed_roughness_large = roughness_large[overlap:-overlap, overlap:-overlap]
        trimmed_tpi_small = tpi_small[overlap:-overlap, overlap:-overlap]
        trimmed_tpi_medium = tpi_medium[overlap:-overlap, overlap:-overlap]
        trimmed_tpi_large = tpi_large[overlap:-overlap, overlap:-overlap]
        trimmed_tri_small = tri_small[overlap:-overlap, overlap:-overlap]
        trimmed_tri_medium = tri_medium[overlap:-overlap, overlap:-overlap]
        trimmed_tri_large = tri_large[overlap:-overlap, overlap:-overlap]

        return (
            trimmed_slope, trimmed_aspect, trimmed_profile_curvature, trimmed_planform_curvature,
            trimmed_roughness_small, trimmed_roughness_medium, trimmed_roughness_large,
            trimmed_tpi_small, trimmed_tpi_medium, trimmed_tpi_large,
            trimmed_tri_small, trimmed_tri_medium, trimmed_tri_large
        )

# Define metric functions (unchanged)
def calculate_slope(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    return np.sqrt(dzdx**2 + dzdy**2)

def calculate_aspect(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    return np.arctan2(dzdy, dzdx) * 180 / np.pi

def calculate_profile_curvature(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    dzdxx = sobel(dzdx, axis=1)
    dzdyy = sobel(dzdy, axis=0)
    return dzdxx + dzdyy

def calculate_planform_curvature(dem):
    dzdx = sobel(dem, axis=1)
    dzdy = sobel(dem, axis=0)
    dzdxx = sobel(dzdx, axis=1)
    dzdyy = sobel(dzdy, axis=0)
    return dzdxx - dzdyy

def calculate_roughness_window(dem, size):
    mean_elevation = uniform_filter(dem, size=size)
    elevation_diff = dem - mean_elevation
    return uniform_filter(elevation_diff**2, size=size)**0.5

def calculate_tpi_window(dem, size):
    mean_elevation = uniform_filter(dem, size=size)
    return dem - mean_elevation

def calculate_tri_window(dem, size):
    mean_elevation = uniform_filter(dem, size=size)
    elevation_diff = dem - mean_elevation
    return uniform_filter(np.abs(elevation_diff), size=size)

if __name__ == "__main__":
    dtm_path = r"E:\Thesis\merge\merged_west.tif"
    output_dir = r"E:\Thesis\testing\output_metrics\merge_west"
    os.makedirs(output_dir, exist_ok=True)

    overlap = 10  # Overlap in Pixeln
    tile_size = 256  # Größe der Kacheln ohne Overlap

    with rasterio.open(dtm_path) as src:
        meta = src.meta.copy()
        width, height = src.width, src.height

        windows = [
            Window(j, i, min(tile_size, width - j), min(tile_size, height - i))
            for i in range(0, height, tile_size)
            for j in range(0, width, tile_size)
        ]

        results = []
        for window in tqdm(windows, desc="Processing chunks"):
            result = process_chunk(dtm_path, window, overlap)
            results.append(result)

        # Save outputs
        metrics = [
            "slope", "aspect", "profile_curvature", "planform_curvature", 
            "roughness_small", "roughness_medium", "roughness_large", 
            "tpi_small", "tpi_medium", "tpi_large", 
            "tri_small", "tri_medium", "tri_large"
        ]

        for i, metric in enumerate(metrics):
            output_path = os.path.join(output_dir, f"{metric}.tif")
            meta.update(dtype="float32")

            # Reconstruct full array for the metric
            full_array = np.zeros((height, width), dtype="float32")
            for result, window in zip(results, windows):
                data = result[i]
                row_start = int(window.row_off)
                row_end = row_start + data.shape[0]
                col_start = int(window.col_off)
                col_end = col_start + data.shape[1]
                full_array[row_start:row_end, col_start:col_end] = data

            # Write to the output file
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(full_array, 1)

        print("All metrics have been successfully calculated and saved.")

