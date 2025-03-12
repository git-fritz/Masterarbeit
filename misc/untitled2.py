# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:03:09 2025

@author: Felix
"""

import rasterio
import numpy as np

# 📌 Define Input and Output Paths
input_raster = r"E:\Thesis\shapes\cluster_mounds.tif"
output_raster = r"E:\Thesis\shapes\cluster_mounds_filter.tif"

# 📌 Open the Raster
with rasterio.open(input_raster) as src:
    data = src.read(1)  # Read the first band
    profile = src.profile  # Copy metadata
    nodata_value = src.nodata  # Get existing NoData value

    # If NoData is not set, define one
    if nodata_value is None:
        nodata_value = -9999  # Set a custom NoData value

    # 📌 Replace all non-zero values with NoData
    data[data != 0] = nodata_value

    # 📌 Update the metadata
    profile.update(dtype=rasterio.float32, nodata=nodata_value)

    # 📌 Write the modified raster to disk
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)

print(f"✅ Filtered raster saved as {output_raster}")
