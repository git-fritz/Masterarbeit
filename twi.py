# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 20:33:28 2025

@author: Felix
"""
# %%
 
gdalinfo "E:\Thesis\data\DEM\merged_raster_clean9999.tif"
Pixel Size = (X_RESOLUTION, Y_RESOLUTION)

# %%


import richdem as rd
import rasterio
import numpy as np

# File paths
dem_path = r"E:\Thesis\data\DEM\resampled_dem.tif"
output_twi = r"E:\Thesis\data\TWI\twi.tif"

# Step 1: Load the DEM
dem = rd.LoadGDAL(dem_path)

# Step 2: Compute Slope (tan β)
slope = rd.TerrainAttribute(dem, attrib='slope_radians')

# Step 3: Compute Flow Accumulation (D8 method)
flow_accum = rd.FlowAccumulation(dem, method='D8')

# Step 4: Compute TWI
# Avoid division by zero errors
slope[slope == 0] = np.nan  # Replace zero slopes with NaN
flow_accum[flow_accum == 0] = np.nan  # Avoid log(0)

# Compute TWI safely
twi = np.log(flow_accum / np.tan(slope))

# Replace NaN values with 0 (or another meaningful value)
twi[np.isnan(twi)] = 0


# Replace infinite values with NaN
twi[np.isinf(twi)] = np.nan

# Step 5: Save TWI as a GeoTIFF
with rasterio.open(dem_path) as src:
    meta = src.meta.copy()
    meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)

    with rasterio.open(output_twi, 'w', **meta) as dst:
        dst.write(twi.astype(np.float32), 1)

print(f"✅ TWI raster saved as: {output_twi}")
# %%

import richdem as rd
import rasterio
import numpy as np

# File paths
dem_path = r"E:\Thesis\data\DEM\resampled_dem.tif"  # Ensure resampled DEM
output_twi = r"E:\Thesis\data\TWI\twi_richdem.tif"

# Step 1: Load the DEM
dem = rd.LoadGDAL(dem_path)

# Step 2: Compute Slope (tan β)
slope = rd.TerrainAttribute(dem, attrib='slope_radians')

# Step 3: Compute Flow Accumulation (D8 method)
flow_accum = rd.FlowAccumulation(dem, method='D8')

# Step 4: Compute TWI (handling NaN and division by zero)
slope[slope == 0] = np.nan  # Prevent division by zero
flow_accum[flow_accum == 0] = np.nan  # Prevent log(0)

twi = np.log(flow_accum / np.tan(slope))
twi[np.isnan(twi)] = 0  # Replace NaN values with 0

# Step 5: Save TWI as a GeoTIFF
with rasterio.open(dem_path) as src:
    meta = src.meta.copy()
    meta.update(dtype=rasterio.float32, count=1, nodata=0)

    with rasterio.open(output_twi, 'w', **meta) as dst:
        dst.write(twi.astype(np.float32), 1)

print(f"✅ Fixed TWI raster saved as: {output_twi}")

