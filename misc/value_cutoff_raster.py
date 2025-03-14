# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:48:38 2025

@author: Felix
"""
# %%
# THIS CODE TAKES A CHM (OR ANY RASTER) AND GIVES THE OPTION TO CUTOFF VALUES
# ABOVE OR BELOW A SET THRESHOLD
#
# %%


import rasterio
import numpy as np
import os

def process_raster(input_raster, output_raster, threshold):
    with rasterio.open(input_raster) as src:
        data = src.read(1)
        meta = src.meta.copy()

        # Remove values above threshold
        data[data > threshold] = np.nan

        # Update metadata to reflect changes
        meta.update(dtype='float32', nodata=np.nan)

        with rasterio.open(output_raster, "w", **meta) as dst:
            dst.write(data.astype("float32"), 1)
    print(f"Raster processed and saved to {output_raster}")

if __name__ == "__main__":
    input_raster = r"E:\Thesis\data\CHM\merged_chm.tif"
    output_raster = r"E:\Thesis\data\CHM\chm_under2.tif"
    threshold = 2   # Set your threshold value here
    os.makedirs(os.path.dirname(output_raster), exist_ok=True)

    process_raster(input_raster, output_raster, threshold)
