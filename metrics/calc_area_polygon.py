# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:53:30 2025

@author: Felix
"""

import geopandas as gpd

# ✅ Update paths
input_gpkg = r"E:\Thesis\data\DEM_TPI\tpi55_filtered_polygon.gpkg" # Replace with your input GPKG file
output_gpkg = r"E:\Thesis\data\DEM_TPI\merged_raster_clean9999_tpi55_filter002_polygons_area.gpkg"  # Where to save the new GPKG

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
