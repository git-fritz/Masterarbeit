# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 16:58:37 2025

@author: Felix
"""

import geopandas as gpd

# Load both GeoPackages
gdf_A = gpd.read_file(r"E:\Thesis\FLM_shrink\felix_shrunk_footprint_ID_segments100m2.gpkg")  # Larger shapes
gdf_B = gpd.read_file(r"E:\Thesis\FLM_shrink\felix_shrunk_footprint_plots20m2.gpkg")  # Smaller shapes with "side" column


# Ensure both datasets have the same CRS
if gdf_A.crs != gdf_B.crs:
    gdf_B = gdf_B.to_crs(gdf_A.crs)

# 🔹 Reset Index to Avoid Reindexing Issues
gdf_A = gdf_A.reset_index()
gdf_B = gdf_B.reset_index()

# Perform spatial join to find overlaps
joined = gpd.sjoin(gdf_B, gdf_A, how="left", predicate="intersects")  # Use `how="left"` to keep all of A

# 🔹 Merge geometries before intersection calculation
joined = joined.merge(gdf_A[["index", "geometry"]], left_on="index_right", right_on="index")

# Compute intersection areas correctly
joined["overlap_area"] = joined.geometry_x.intersection(joined.geometry_y).area

# Keep only the row with the largest overlap per shape in A
best_overlap = joined.loc[joined.groupby("index_right")["overlap_area"].idxmax()]

# 🔹 Merge the "side" column from B to A
gdf_A = gdf_A.merge(best_overlap[["index_right", "side"]], left_on="index", right_on="index_right", how="left")

# 🔹 If no overlap was found, set "side" = 1
gdf_A["side"].fillna(1, inplace=True)

# Save the updated GeoPackage
gdf_A.to_file(r"E:\Thesis\FLM_shrink\felix_shrunk_footprint_ID_segments100m2_side.gpkg", driver="GPKG")

print("✅ 'side' column successfully added to A based on the largest overlap!")