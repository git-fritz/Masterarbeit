# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:07:29 2025

@author: Felix
"""

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
        tri_large = calculate_tri_window(dem, size=399)

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
            trimmed_slope, trimmed_aspect, trimmed_profile_curvature, trimmed_planform_curvature
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
    dtm_path = r"E:\Thesis\data\DEM\merged_raster_clean9999.tif"
    output_dir = r"E:\Thesis\data\DEM\tpi399.tif"
    os.makedirs(output_dir, exist_ok=True)

    overlap = 10  # Overlap in Pixeln
    tile_size = 128  # Größe der Kacheln ohne Overlap

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
            #"slope", 
            # "aspect", "profile_curvature", "planform_curvature", 
            # "roughness_small", "roughness_medium", "roughness_large", 
            # "tpi_small", "tpi_medium", "tpi_large", 
            # "tri_small", "tri_medium", 
            "tri_large"
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

import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.transform import Affine
from scipy.ndimage import sobel, uniform_filter, gaussian_filter
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
        slope = calculate_slope_with_window(dem, sigma=5)  # Calculate slope over a broader window
        aspect = calculate_aspect(dem)
        profile_curvature = calculate_profile_curvature(dem)
        planform_curvature = calculate_planform_curvature(dem)

        # Overlap abschneiden
        trimmed_slope = slope[overlap:-overlap, overlap:-overlap]
        trimmed_aspect = aspect[overlap:-overlap, overlap:-overlap]
        trimmed_profile_curvature = profile_curvature[overlap:-overlap, overlap:-overlap]
        trimmed_planform_curvature = planform_curvature[overlap:-overlap, overlap:-overlap]

        return (
            trimmed_slope, trimmed_aspect, trimmed_profile_curvature, trimmed_planform_curvature
        )

def calculate_slope_with_window(dem, sigma=5):
    """
    Calculate slope over a broader moving window using Gaussian smoothing.

    Parameters:
        dem (numpy.ndarray): Input DEM.
        sigma (float): Standard deviation for Gaussian kernel, controlling the window size.

    Returns:
        numpy.ndarray: Slope array.
    """
    # Smooth the DEM using a Gaussian filter
    smoothed_dem = gaussian_filter(dem, sigma=sigma)

    # Calculate derivatives on the smoothed DEM
    dzdx = sobel(smoothed_dem, axis=1)
    dzdy = sobel(smoothed_dem, axis=0)

    # Calculate the slope
    slope = np.sqrt(dzdx**2 + dzdy**2)
    return slope

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
            "slope"#, "aspect", "profile_curvature", "planform_curvature"
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
