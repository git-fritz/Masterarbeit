# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 18:40:58 2025

@author: Felix
"""
# %%
# THIS CODE TAKES THE FINAL MODEL INPUT GPKG AND RANDOMLY SPLITS IT INTO A 80/20
# SPLIT: THE 80% ARE USED TO TRAIN THE MODEL, THE 20% ARE SET ASIDE TO BE USED
# LATER AS UNSEEN/-TRAINED VALIDATION DATA
#
# %%



#create subset of data
import geopandas as gpd
import numpy as np
import math

# File paths
input_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics100_v8.gpkg"
output_subset_gpkg = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics100_v6_20.gpkg"
output_remaining_gpkg = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics100_v6_80.gpkg"

# Load the GPKG
gdf = gpd.read_file(input_gpkg)

# Compute 20% of the dataset (rounding up)
subset_size = math.ceil(len(gdf) * 0.2)

# Select a random subset of the data
subset_gdf = gdf.sample(n=subset_size, random_state=42)  # Set seed for reproducibility

# Select the remaining 80% of the data
remaining_gdf = gdf.drop(subset_gdf.index)

# Save the 20% subset
subset_gdf.to_file(output_subset_gpkg, driver="GPKG")

# Save the 80% remaining data
remaining_gdf.to_file(output_remaining_gpkg, driver="GPKG")

print(f"✅ Subset saved with {subset_size} rows: {output_subset_gpkg}")
print(f"✅ Remaining dataset saved with {len(remaining_gdf)} rows: {output_remaining_gpkg}")
