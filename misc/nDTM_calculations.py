import os
import cupy as cp
from cupyx.scipy import ndimage as ndi
import rasterio
import numpy as np
from skimage.measure import block_reduce
from rasterio.enums import Resampling
import rasterio.warp

# Define input and output folders.
input_folder = r"E:\Thesis\data\DEM\nDTM_test"
output_folder = r"E:\Thesis\data\DEM\output"
os.makedirs(output_folder, exist_ok=True)

# Loop over all .tif files in the input folder.
for filename in os.listdir(input_folder):
    if not filename.endswith('.tif'):
        continue
    input_tif = os.path.join(input_folder, filename)
    output_tif = os.path.join(output_folder, filename.replace('.tif', '_nDTM.tif'))
    print(f"Processing: {input_tif}")

    # 1. Read the original DTM raster.
    with rasterio.open(input_tif) as src:
        dtm = src.read(1)  # Assumes single-band raster.
        profile = src.profile.copy()
        # Original resolution is assumed to be 10 cm.

    # 2. Downscale the DTM to 50 cm resolution using block_reduce (5x5 blocks).
    dtm_down = block_reduce(dtm, block_size=(5, 5), func=np.median)
    # dtm_down now has 1/5 the resolution (i.e. 50 cm).

    # Transfer the downscaled array to GPU.
    dtm_down_cp = cp.asarray(dtm_down)

    # 3. Compute the median surface over a sliding window of 6 m.
    #    At 50 cm resolution, a 6 m square is 6 / 0.5 = 12 pixels.
    median_surface_down_cp = ndi.median_filter(dtm_down_cp, size=12)

    # 4. Upsample the median surface back to the original 10 cm resolution.
    #    Upsampling factor is 5.
    median_surface_up_cp = ndi.zoom(median_surface_down_cp, zoom=5, order=1)  # bilinear interpolation
    # Ensure the upsampled surface has the same shape as the original DTM.
    if median_surface_up_cp.shape != dtm.shape:
        print("Warning: Upsampled median surface shape does not match the original raster shape. Cropping...")
        median_surface_up_cp = median_surface_up_cp[:dtm.shape[0], :dtm.shape[1]]

    # Convert the upsampled median surface to a NumPy array.
    median_surface = cp.asnumpy(median_surface_up_cp)

    # 5. Compute nDTM: difference between the original DTM and the median surface.
    nDTM = dtm - median_surface

    # 6. Save the nDTM raster.
    # Update the profile for a single-band float32 raster.
    profile.update({
        'dtype': 'float32',
        'count': 1
    })
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(nDTM.astype(np.float32), 1)

    print(f"✅ nDTM raster saved to: {output_tif}")

# %%


import os
import cupy as cp
from cupyx.scipy import ndimage as ndi
import rasterio
import numpy as np
from skimage.measure import block_reduce
from rasterio.enums import Resampling
import rasterio.warp

# Define input and output folders.
input_folder = r"E:\Thesis\data\DEM\nDTM_test"
output_folder = r"E:\Thesis\data\DEM\output"
os.makedirs(output_folder, exist_ok=True)

# Loop over all .tif files in the input folder.
for filename in os.listdir(input_folder):
    if not filename.endswith('.tif'):
        continue
    input_tif = os.path.join(input_folder, filename)
    output_tif = os.path.join(output_folder, filename.replace('.tif', '_nDTM.tif'))
    print(f"Processing: {input_tif}")

    # 1. Read the original DTM raster.
    with rasterio.open(input_tif) as src:
        dtm = src.read(1).astype(np.float32)  # Assumes single-band raster, ensures float dtype
        profile = src.profile.copy()
        nodata_value = src.nodata  # Get the original NoData value

        # Replace -9999 (or the detected nodata value) with NaN
        if nodata_value is not None:
            dtm[dtm == nodata_value] = np.nan
        else:
            dtm[dtm == -9999] = np.nan  # Default case

    # 2. Downscale the DTM to 50 cm resolution using block_reduce (5x5 blocks).
    dtm_down = block_reduce(dtm, block_size=(5, 5), func=np.nanmedian)  # Use nan-aware median

    # Transfer the downscaled array to GPU.
    dtm_down_cp = cp.asarray(dtm_down)

    # 3. Compute the median surface over a sliding window of 6 m.
    #    At 50 cm resolution, a 6 m square is 6 / 0.5 = 12 pixels.
    median_surface_down_cp = ndi.median_filter(dtm_down_cp, size=12)

    # 4. Upsample the median surface back to the original 10 cm resolution.
    median_surface_up_cp = ndi.zoom(median_surface_down_cp, zoom=5, order=1)  # bilinear interpolation

    # Ensure the upsampled surface has the same shape as the original DTM.
    if median_surface_up_cp.shape != dtm.shape:
        print("Warning: Upsampled median surface shape does not match the original raster shape. Cropping...")
        median_surface_up_cp = median_surface_up_cp[:dtm.shape[0], :dtm.shape[1]]

    # Convert the upsampled median surface back to a NumPy array.
    median_surface = cp.asnumpy(median_surface_up_cp)

    # 5. Compute nDTM: difference between the original DTM and the median surface.
    nDTM = dtm - median_surface

    # Preserve NaNs in the final output (optional: set a new nodata value)
    output_nodata_value = -9999
    nDTM[np.isnan(nDTM)] = output_nodata_value

    # 6. Save the nDTM raster.
    # Update the profile for a single-band float32 raster.
    profile.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': output_nodata_value  # Ensure nodata value is correctly stored
    })

    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(nDTM.astype(np.float32), 1)

    print(f"✅ nDTM raster saved to: {output_tif}")
