# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 10:45:33 2025

@author: Felix
"""

import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.cluster import MiniBatchKMeans  # More memory-efficient than KMeans
from sklearn.preprocessing import StandardScaler

# 📌 Step 1: Load nDTM and Slope Raster
ndtm_path = r"E:\Thesis\shapes\ndtm_east.tif"  # Normalized DTM
slope_path = r"E:\Thesis\shapes\slope_ndtm_east.tif"  # Slope raster
output_mound_map = r"E:\Thesis\shapes\cluster_mounds.tif"

with rasterio.open(ndtm_path) as src:
    ndtm = src.read(1)  # Read normalized elevation
    profile = src.profile  # Get metadata

with rasterio.open(slope_path) as src:
    slope = src.read(1)  # Read slope values

# 📌 Step 2: Prepare Data for Clustering
ndtm_flat = ndtm.flatten()
slope_flat = slope.flatten()

# Remove NoData values and define a valid range for nDTM
valid_mask = (ndtm_flat > -5) & (ndtm_flat < 5) & (slope_flat >= 0)  # Adjust as needed
features = np.column_stack((ndtm_flat[valid_mask], slope_flat[valid_mask]))

# 📌 Step 3: Standardize Features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 📌 Step 4: Apply K-Means Clustering
num_clusters = 3  # 3 Classes: Mounds, Hollows, Flat
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = np.full(ndtm_flat.shape, -9999)  # Default NoData value
clusters[valid_mask] = kmeans.fit_predict(features_scaled)

# 📌 Step 5: Reshape and Save the Mound Classification Raster
classified_mounds = clusters.reshape(ndtm.shape)

# Update metadata to store integer values
profile.update(dtype=rasterio.int32, nodata=-9999)

with rasterio.open(output_mound_map, "w", **profile) as dst:
    dst.write(classified_mounds.astype(np.int32), 1)

# 📌 Step 6: Visualize the Result
plt.figure(figsize=(10, 6))
plt.imshow(classified_mounds, cmap="terrain")
plt.colorbar(label="Cluster ID (0=Flat, 1=Mounds, 2=Hollows)")
plt.title("Mound Classification Using K-Means on nDTM")
plt.show()

print(f"✅ Mound classification saved to {output_mound_map}")

# %%
import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.ndimage import label

# 📌 Step 1: Load nDTM and Slope Raster
ndtm_path = r"E:\Thesis\shapes\ndtm_east.tif"  # Normalized DTM
slope_path = r"E:\Thesis\shapes\slope_ndtm_east.tif"  # Slope raster
output_mound_map = r"E:\Thesis\shapes\cluster_mounds_filter.tif"

with rasterio.open(ndtm_path) as src:
    ndtm = src.read(1)  # Read normalized elevation
    profile = src.profile  # Get metadata

with rasterio.open(slope_path) as src:
    slope = src.read(1)  # Read slope values

# 📌 Step 2: Prepare Data for Clustering
ndtm_flat = ndtm.flatten()
slope_flat = slope.flatten()

# Remove NoData values and define a valid range for nDTM
valid_mask = (ndtm_flat > -2) & (ndtm_flat < 2) & (slope_flat >= 0)  # Adjust as needed
features = np.column_stack((ndtm_flat[valid_mask], slope_flat[valid_mask]))

# 📌 Step 3: Standardize Features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 📌 Step 4: Apply K-Means Clustering with More Strict Classification
num_clusters = 4  # Increase to detect small vs large mounds separately
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = np.full(ndtm_flat.shape, -9999)  # Default NoData value
clusters[valid_mask] = kmeans.fit_predict(features_scaled)

# 📌 Step 5: Reshape the Classified Data
classified_mounds = clusters.reshape(ndtm.shape)

# 📌 Step 6: Post-Processing to Remove Large Mounds and Non-Mounds

# Define mound cluster ID (find cluster with highest mean nDTM)
mound_cluster_id = np.argmax([np.mean(ndtm_flat[clusters == i]) for i in range(num_clusters)])

# Create a binary mound mask
mound_mask = (classified_mounds == mound_cluster_id).astype(np.uint8)

# 📌 Step 7: Remove Mounds That Are Too Large
labeled_mounds, num_features = label(mound_mask)  # Connected component labeling

# Set minimum and maximum mound sizes (in pixels)
min_mound_size = 10  # Adjust based on resolution
max_mound_size = 1000  # Prevent overly large mounds

for i in range(1, num_features + 1):
    size = np.sum(labeled_mounds == i)
    if size < min_mound_size or size > max_mound_size:
        labeled_mounds[labeled_mounds == i] = 0  # Remove small & large mounds

# 📌 Step 8: Remove Mounds on Steep Slopes (> 15°)
mound_mask_final = (labeled_mounds > 0) & (slope < 15)  # Keep only low-slope mounds

# 📌 Step 9: Save the Filtered Mound Map
profile.update(dtype=rasterio.uint8, nodata=0)

with rasterio.open(output_mound_map, "w", **profile) as dst:
    dst.write(mound_mask_final.astype(np.uint8), 1)

# 📌 Step 10: Visualize the Final Result
plt.figure(figsize=(10, 6))
plt.imshow(mound_mask_final, cmap="terrain")
plt.colorbar(label="Mound Detection (1=Yes, 0=No)")
plt.title("Filtered Mound Classification Using K-Means & Post-Processing")
plt.show()

print(f"✅ Stricter mound classification saved to {output_mound_map}")

# %%



# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 10:45:33 2025

@author: Felix
"""

import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.cluster import MiniBatchKMeans  # More memory-efficient than KMeans
from sklearn.preprocessing import StandardScaler

# 📌 Step 1: Load nDTM and Slope Raster
ndtm_path = r"E:\Thesis\data\DEM\nDTM_clip.tif"  # Normalized DTM
slope_path = r"E:\Thesis\testing\metrics\slope_ndtm.tif"  # Slope raster
output_mound_map = r"E:\Thesis\data\cluster_mounds.tif"


# 📌 Define Block Size (Adjust Based on RAM Availability)
block_size = 1024  # Process raster in 1024x1024 chunks

# 📌 Open Input Rasters
with rasterio.open(ndtm_path) as ndtm_src, rasterio.open(slope_path) as slope_src:
    profile = ndtm_src.profile.copy()
    
    # Update profile for integer classification output
    profile.update(dtype=rasterio.int32, nodata=-9999, count=1)

    # 📌 Open Output Raster for Writing
    with rasterio.open(output_mound_map, "w", **profile) as dst:
        for j in range(0, ndtm_src.height, block_size):
            for i in range(0, ndtm_src.width, block_size):
                print(f"Processing block: X={i}, Y={j}")

                # 📌 Define Block Window
                window = Window(i, j, min(block_size, ndtm_src.width - i), min(block_size, ndtm_src.height - j))

                # 📌 Read Block Data
                ndtm_block = ndtm_src.read(1, window=window)
                slope_block = slope_src.read(1, window=window)

                # 📌 Mask NoData values
                valid_mask = (ndtm_block > -5) & (ndtm_block < 5) & (slope_block >= 0)
                valid_pixels = np.column_stack((ndtm_block[valid_mask], slope_block[valid_mask]))

                if valid_pixels.shape[0] < 10:
                    print(f"Skipping empty block at X={i}, Y={j}")
                    continue

                # 📌 Standardize Features
                scaler = StandardScaler()
                valid_pixels_scaled = scaler.fit_transform(valid_pixels)

                # 📌 Apply MiniBatchKMeans (Better for Large Data)
                num_clusters = 3
                kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000, random_state=42)
                cluster_labels = np.full(ndtm_block.shape, -9999, dtype=np.int32)  # Default NoData
                cluster_labels[valid_mask] = kmeans.fit_predict(valid_pixels_scaled)

                # 📌 Write Processed Block to Output Raster
                dst.write(cluster_labels, 1, window=window)

print(f"✅ Mound classification saved to {output_mound_map}")