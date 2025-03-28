# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:53:30 2025

@author: Felix
"""
# %%
# This script processes a GeoPackage (.gpkg) file containing polygon geometries
# and calculates the area of each polygon, ensuring the correct coordinate 
# reference system (CRS) for accurate calculations. Then, it saves the updated dataset.
#
# ADDS AREA_M2 COLUMN TO TPI GPKG TO THEN USE LATER ON FOR MODEL INPUTS
#
# %%


import geopandas as gpd

# ✅ Update paths
input_gpkg = r"E:\Thesis\data\mounds_percentile\mounds_percentile.gpkg" # Replace with your input GPKG file
output_gpkg = r"E:\Thesis\data\mounds_percentile\mounds_percentile_area.gpkg"  # Where to save the new GPKG

# ✅ Load the dataset
gdf = gpd.read_file(input_gpkg)

# ✅ Ensure CRS is projected (for accurate area calculations)
if gdf.crs is None:
    print("Warning: No CRS found! Assigning EPSG:2956 (NAD83 UTM Zone 12N).")
    gdf.set_crs("EPSG:2956", inplace=True)

elif not gdf.crs.is_projected:
    print(f"Reprojecting from {gdf.crs} to EPSG:2956 for accurate area calculation.")
    gdf = gdf.to_crs("EPSG:2956")

# ✅ Calculate the area of each polygon
gdf["Area_m2"] = gdf.geometry.area  # Area in square meters

# ✅ Save the updated dataset
gdf.to_file(output_gpkg, driver="GPKG")

print(f"Updated dataset with area column saved to: {output_gpkg}")
