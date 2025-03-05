# -*- coding: utf-8 -*-
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the GeoPackage file
def load_gpkg(file_path, layer_name=None):
    gdf = gpd.read_file(file_path, layer=layer_name) if layer_name else gpd.read_file(file_path)
    return gdf

# Generate boxplot and histogram
def plot_area_distribution(gdf, column_name="Area_m2", max_value=10):
    if column_name not in gdf.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")
    
    # Filter values to include only those ≤ max_value
    area_values = gdf[column_name].dropna()
    area_values = area_values[area_values <= max_value]  # Remove values greater than max_value
    
    if area_values.empty:
        print(f"No values are ≤ {max_value}. Nothing to plot.")
        return
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Boxplot
    axes[0].boxplot(area_values, vert=True, patch_artist=True)
    axes[0].set_title("Boxplot of Area Values (≤ 10)")
    axes[0].set_ylabel("Area")
    
    # Histogram
    axes[1].hist(area_values, bins=400, edgecolor='black', alpha=0.7)
    axes[1].set_title("Histogram of Area Values (≤ 10)")
    axes[1].set_xlabel("Area")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# Example usage
file_path = r"E:\Thesis\data\DEM_TPI\merged_raster_clean9999_tpi25_filter002_polygons_area.gpkg" # Replace with your file path
layer_name = None  # Specify if your GPKG has multiple layers

gdf = load_gpkg(file_path, layer_name)
plot_area_distribution(gdf, max_value=0.2)

