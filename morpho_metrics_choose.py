# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:44:27 2024

@author: Felix
"""

import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.transform import Affine
from scipy.ndimage import sobel, uniform_filter
import os
from tqdm import tqdm

# Define which metrics to calculate
metrics_to_calculate = {
    "slope": True,
    "aspect": True,
    "profile_curvature": True,
    "planform_curvature": True,
    "roughness_small": True,
    "roughness_medium": True,
    "roughness_large": True,
    "tpi_small": True,
    "tpi_medium": True,
    "tpi_large": True,
    "tri_small": True,
    "tri_medium": True,
    "tri_large": True
}

def process_chunk(dtm_path, window):
    with rasterio.open(dtm_path) as src:
        dem = src.read(1, window=window)
        print(f"Processing window at {window}")

        results = {}

        if metrics_to_calculate["slope"]:
            results["slope"] = calculate_slope(dem)
            print("Slope calculation done")

        if metrics_to_calculate["aspect"]:
            results["aspect"] = calculate_aspect(dem)
            print("Aspect calculation done")

        if metrics_to_calculate["profile_curvature"]:
            results["profile_curvature"] = calculate_profile_curvature(dem)
            print("Profile curvature calculation done")

        if metrics_to_calculate["planform_curvature"]:
            results["planform_curvature"] = calculate_planform_curvature(dem)
            print("Planform curvature calculation done")

        if metrics_to_calculate["roughness_small"]:
            results["roughness_small"] = calculate_roughness_window(dem, size=3)
            print("Small-scale roughness calculation done")

        if metrics_to_calculate["roughness_medium"]:
            results["roughness_medium"] = calculate_roughness_window(dem, size=10)
            print("Medium-scale roughness calculation done")

        if metrics_to_calculate["roughness_large"]:
            results["roughness_large"] = calculate_roughness_window(dem, size=50)
            print("Large-scale roughness calculation done")

        if metrics_to_calculate["tpi_small"]:
            results["tpi_small"] = calculate_tpi_window(dem, size=3)
            print("Small-scale TPI calculation done")

        if metrics_to_calculate["tpi_medium"]:
            results["tpi_medium"] = calculate_tpi_window(dem, size=10)
            print("Medium-scale TPI calculation done")

        if metrics_to_calculate["tpi_large"]:
            results["tpi_large"] = calculate_tpi_window(dem, size=50)
            print("Large-scale TPI calculation done")

        if metrics_to_calculate["tri_small"]:
            results["tri_small"] = calculate_tri_window(dem, size=3)
            print("Small-scale TRI calculation done")

        if metrics_to_calculate["tri_medium"]:
            results["tri_medium"] = calculate_tri_window(dem, size=10)
            print("Medium-scale TRI calculation done")

        if metrics_to_calculate["tri_large"]:
            results["tri_large"] = calculate_tri_window(dem, size=50)
            print("Large-scale TRI calculation done")

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
    dtm_path = r"E:\BERA\BERA_Aufenthalt\LiDEA_Pilot\0000_Results\XY_Results\Aug\LideaPilot-L1-ROADWEST-2\17_Raster\dtm.tif"  # Update this path to your DTM file
    output_dir = r"E:\Thesis\testing\output_metrics"
    os.makedirs(output_dir, exist_ok=True)

    use_chunks = True  # Toggle chunk processing on/off

    with rasterio.open(dtm_path) as src:
        meta = src.meta.copy()
        tile_size = 512  # Reduced tile size to lower memory usage
        width, height = src.width, src.height

        if use_chunks:
            windows = [Window(j, i, min(tile_size, width - j), min(tile_size, height - i))
                       for i in range(0, height, tile_size)
                       for j in range(0, width, tile_size)]

            results = []
            for window in tqdm(windows, desc="Processing chunks"):
                result = process_chunk(dtm_path, window)
                results.append(result)

        else:
            results = [process_chunk(dtm_path, None)]  # Process entire DTM

        # Save outputs
        for metric in metrics_to_calculate.keys():
            if not metrics_to_calculate[metric]:
                continue

            output_path = os.path.join(output_dir, f"{metric}.tif")
            meta.update(dtype="float32")

            # Reconstruct full array for the metric
            full_array = np.zeros((height, width), dtype="float32")
            if use_chunks:
                for result, window in zip(results, windows):
                    if metric in result:
                        data = result[metric]
                        row_start = int(window.row_off)
                        row_end = row_start + data.shape[0]
                        col_start = int(window.col_off)
                        col_end = col_start + data.shape[1]
                        full_array[row_start:row_end, col_start:col_end] = data
            else:
                full_array = results[0][metric]

            # Write to the output file
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(full_array, 1)

        print("All metrics have been successfully calculated and saved.")
