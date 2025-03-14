# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:40:53 2025

@author: Felix
"""

# %%
# THIS CODE IS USED TO THRESHOLD THE POLYGONIZED TPI LAYERS BY ADDING AN "AREA"
# COLUMN AND THEN SETTING A MINIMUM AREA THRESHOLD TO CLEAR UP NOISY TPI DATA
# %%



import geopandas as gpd
import fiona

def add_area_column(gpkg_path, output_path=None):
    # Get all layer names
    layers = fiona.listlayers(gpkg_path)
    
    for layer in layers:
        # Load each layer in the GeoPackage
        gdf = gpd.read_file(gpkg_path, layer=layer)
        
        # Calculate area and add as a new column
        gdf['area'] = gdf.geometry.area
        
        # Save the modified file (overwrite or create a new one)
        if output_path:
            gdf.to_file(output_path, layer=layer, driver="GPKG")
        else:
            gdf.to_file(gpkg_path, layer=layer, driver="GPKG")
    
    print(f"Updated GPKG saved to: {output_path or gpkg_path}")
    return layers

def remove_small_areas(gpkg_path, area_threshold, output_path=None):
    # Get all layer names
    layers = fiona.listlayers(gpkg_path)
    
    for layer in layers:
        # Load each layer in the GeoPackage
        gdf = gpd.read_file(gpkg_path, layer=layer)
        
        # Filter out features with area less than the threshold
        gdf = gdf[gdf['area'] >= area_threshold]
        
        # Save the modified file (overwrite or create a new one)
        if output_path:
            gdf.to_file(output_path, layer=layer, driver="GPKG")
        else:
            gdf.to_file(gpkg_path, layer=layer, driver="GPKG")
    
    print(f"Filtered GPKG saved to: {output_path or gpkg_path}")
    return layers

def get_layer_names(gpkg_path):
    # Get all layer names from the GeoPackage
    layers = fiona.listlayers(gpkg_path)
    print(f"Layers in {gpkg_path}: {layers}")
    return layers

# Example usage
gpkg_file = r"E:\Thesis\data\mounds_percentile\mounds_percentile_area.gpkg"
output_gpkg = r"E:\Thesis\data\mounds_percentile\mounds_percentile_area05.gpkg" # Optional, set to None to overwrite
area_threshold = 0.5  # Set the area threshold

layer_names = get_layer_names(gpkg_file)
updated_layers = add_area_column(gpkg_file, output_gpkg)
filtered_layers = remove_small_areas(output_gpkg or gpkg_file, area_threshold, output_gpkg)
