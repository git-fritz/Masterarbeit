# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:21:58 2025

@author: Felix
"""

import numpy as np
import rasterio

# 📌 Define Paths
dtm_path = r"E:\Thesis\data\DEM\nDTM_clip.tif"  # Input DTM
output_filtered = r"E:\Thesis\testing\dtm_top_quantile8.tif"  # Output raster

# 📌 Set the Quantile Threshold (e.g., top 10%)
quantile_threshold = 0.8 # Keep only the top 10% of elevations

# 📌 Load the DTM
with rasterio.open(dtm_path) as src:
    dtm = src.read(1)  # Read elevation values
    profile = src.profile  # Get metadata
    nodata_value = src.nodata  # Get NoData value

    # If NoData is not set, define one
    if nodata_value is None:
        nodata_value = -9999

    # 📌 Compute Quantile Value (Threshold for High Elevations)
    valid_pixels = dtm[dtm != nodata_value]  # Exclude NoData
    quantile_value = np.quantile(valid_pixels, quantile_threshold)

    # 📌 Create a Mask: Keep Only the Top Elevations
    top_elevations = np.where(dtm >= quantile_value, dtm, nodata_value)

    # 📌 Update Metadata for Output
    profile.update(dtype=rasterio.float32, nodata=nodata_value)

    # 📌 Save the Filtered Raster
    with rasterio.open(output_filtered, "w", **profile) as dst:
        dst.write(top_elevations.astype(np.float32), 1)

print(f"✅ Filtered DTM saved: {output_filtered}")
print(f"Top {100 - (quantile_threshold * 100)}% of elevations removed, keeping values ≥ {quantile_value:.2f}m")
# %%
# same script but BINARY output
import numpy as np
import rasterio

# 📌 Define Paths
dtm_path = r"E:\Thesis\data\DEM\nDTM_clip.tif"  # Input DTM
output_mask = r"E:\Thesis\testing\dtm_top_quantile85bin.tif"   # Output binary mask

# 📌 Set the Quantile Threshold (e.g., top 10%)
quantile_threshold = 0.85  # Keep only the top 10% of elevations

# 📌 Load the DTM
with rasterio.open(dtm_path) as src:
    dtm = src.read(1)  # Read elevation values
    profile = src.profile  # Get metadata
    nodata_value = src.nodata  # Get NoData value

    # If NoData is not set, define one
    if nodata_value is None:
        nodata_value = -9999

    # 📌 Compute Quantile Value (Threshold for High Elevations)
    valid_pixels = dtm[dtm != nodata_value]  # Exclude NoData
    quantile_value = np.quantile(valid_pixels, quantile_threshold)

    # 📌 Create a Binary Mask: 1 = High Elevation, 0 = Not High
    binary_mask = np.where(dtm >= quantile_value, 1, 0)

    # 📌 Update Metadata for Output
    profile.update(dtype=rasterio.uint8, nodata=0)  # Binary output (0/1)

    # 📌 Save the Binary Mask Raster
    with rasterio.open(output_mask, "w", **profile) as dst:
        dst.write(binary_mask.astype(np.uint8), 1)

print(f"✅ Binary mask saved: {output_mask}")
print(f"Top {100 - (quantile_threshold * 100)}% of elevations removed, keeping values ≥ {quantile_value:.2f}m")

# %%
# script to only keep the lowest parts = hollows

import numpy as np
import rasterio

# 📌 Define Paths
dtm_path = r"E:\Thesis\data\DEM\nDTM_clip.tif"  # Input DTM
output_mask = r"E:\Thesis\testing\dtm_low_quantile_mask15.tif"  # Output binary mask for hollows

# 📌 Set the Quantile Threshold (e.g., bottom 10%)
quantile_threshold = 0.15   # Keep only the lowest 10% of elevations

# 📌 Load the DTM
with rasterio.open(dtm_path) as src:
    dtm = src.read(1)  # Read elevation values
    profile = src.profile  # Get metadata
    nodata_value = src.nodata  # Get NoData value

    # If NoData is not set, define one
    if nodata_value is None:
        nodata_value = -9999

    # 📌 Compute Quantile Value (Threshold for Low Elevations)
    valid_pixels = dtm[dtm != nodata_value]  # Exclude NoData
    quantile_value = np.quantile(valid_pixels, quantile_threshold)  # Get bottom 10% threshold

    # 📌 Create a Binary Mask: 1 = Hollow (Low Elevation), 0 = Not a Hollow
    binary_mask = np.where(dtm <= quantile_value, 1, 0)

    # 📌 Update Metadata for Output
    profile.update(dtype=rasterio.uint8, nodata=0)  # Binary output (0/1)

    # 📌 Save the Binary Mask Raster
    with rasterio.open(output_mask, "w", **profile) as dst:
        dst.write(binary_mask.astype(np.uint8), 1)

print(f"✅ Binary mask for hollows saved: {output_mask}")
print(f"Top {quantile_threshold * 100}% of lowest elevations kept, removing values above {quantile_value:.2f}m")
