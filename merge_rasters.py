# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:19:57 2025

@author: Felix
"""

import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os

def merge_rasters(input_folder, output_path):
    """
    Merge all raster files in a folder into a single raster file.

    Parameters:
        input_folder (str): Path to the folder containing raster files to merge.
        output_path (str): Path to save the merged raster file.
    """
    # Find all raster files in the folder
    raster_files = glob.glob(os.path.join(input_folder, "*.tif"))
    if not raster_files:
        raise ValueError("No raster files found in the specified folder.")

    # Open all rasters as datasets
    datasets = [rasterio.open(file) for file in raster_files]

    # Merge the datasets
    merged_data, merged_transform = merge(datasets)

    # Use metadata from the first raster for the output
    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": merged_data.shape[1],
        "width": merged_data.shape[2],
        "transform": merged_transform
    })

    # Write the merged raster to a file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(merged_data)

    # Close all the input datasets
    for dataset in datasets:
        dataset.close()

if __name__ == "__main__":
    # Example usage
    input_folder = r"E:\Thesis\merge_chm"
    output_file = r"E:\Thesis\merge_chm\merged_chm.tif"

    merge_rasters(input_folder, output_file)
    print(f"Merged raster saved to {output_file}")
