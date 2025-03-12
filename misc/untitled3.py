# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:21:54 2025

@author: Felix
"""

import rasterio
import numpy as np
from scipy.ndimage import uniform_filter

# 📌 Load DTM
dtm_path = r"E:\Thesis\shapes\ndtm_east.tif"
output_mounds = r"E:\Thesis\testing\mounds_new\mounds_prominence.tif"

with rasterio.open(dtm_path) as src:
    dtm = src.read(1)
    profile = src.profile

# 📌 Compute Smoothed DTM (5m moving window)
window_size = 5  # Adjust for different mound sizes
smoothed_dtm = uniform_filter(dtm, size=window_size)

# 📌 Compute Relative Elevation (Prominence)
relative_elevation = dtm - smoothed_dtm

# 📌 Define Mounds: Areas that are > 0.2m above surroundings
mounds = np.where(relative_elevation > 0.1, 1, 0)  # 1 = Mound, 0 = Not a Mound

# 📌 Save Output Raster
profile.update(dtype=rasterio.uint8, nodata=0)
with rasterio.open(output_mounds, "w", **profile) as dst:
    dst.write(mounds.astype(np.uint8), 1)

print(f"✅ Mound extraction completed. Saved to {output_mounds}")
# %%

import richdem as rd

# 📌 Load DTM
dtm = rd.LoadGDAL(r"E:\Thesis\shapes\ndtm_east.tif")

# 📌 Compute Curvature
curvature = rd.TerrainAttribute(dtm, attrib="curvature")

# 📌 Define Mounds as Convex Areas (Curvature > 0.1)
mounds = (curvature > 0.1).astype(np.uint8)

# 📌 Save Output Raster
rd.SaveGDAL(r"E:\Thesis\testing\mounds_new\mounds_curvature.tif", mounds)
# %%

from scipy.ndimage import label

# 📌 Detect Mound Candidates
mound_candidates = (relative_elevation > 0.2).astype(np.uint8)

# 📌 Group Neighboring Mound Pixels
labeled_mounds, num_mounds = label(mound_candidates)

# 📌 Remove Small Mounds (< 5 pixels)
for i in range(1, num_mounds + 1):
    if np.sum(labeled_mounds == i) < 5:  # 5 pixels = 2m² at 50cm resolution
        labeled_mounds[labeled_mounds == i] = 0

# 📌 Save the refined mounds
with rasterio.open(r"E:\Thesis\testing\mounds_new\refined_mounds.tif", "w", **profile) as dst:
    dst.write(labeled_mounds.astype(np.uint8), 1)
