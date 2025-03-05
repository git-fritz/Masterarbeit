# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:19:30 2025

@author: Felix
"""

import geopandas as gpd

# Load the GeoPackage
gpkg_path = r"E:\Thesis\data\DEM_TPI\tpi55\tpi55_polygon_area01.gpkg"  # Replace with your actual file path
layer_name = None  # Replace with your actual layer name

# Read the GeoPackage
gdf = gpd.read_file(gpkg_path, layer=layer_name)

# Ensure area column exists; otherwise, calculate it
if "area" not in gdf.columns:
    gdf["area"] = gdf.geometry.area

# Calculate perimeter
gdf["perimeter"] = gdf.geometry.length

# Compute complexity
gdf["complexity"] = gdf["perimeter"] / gdf["area"]

# Save the updated GeoPackage
output_gpkg = r"E:\Thesis\data\DEM_TPI\tpi55\tpi55_polygon_area01_complx.gpkg"
gdf.to_file(output_gpkg, layer=layer_name, driver="GPKG")

# Display the first few rows to verify
print(gdf.head())

# If you want to save as CSV for quick inspection:
gdf.to_csv("E:/Thesis/models and code/complexity_output.csv", index=False)
