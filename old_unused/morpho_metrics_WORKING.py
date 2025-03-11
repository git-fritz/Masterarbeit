
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
