# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 19:49:33 2025

@author: Felix
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import box

# File paths
area_shp_path = r"E:\Thesis\müll\area.shp" # Update path if necessary
seismic_lines_gpkg_path = r"E:\Thesis\FLM_shrink\all_group_copy_ID.gpkg"  # Update path if necessary

# Load the study area shapefile
study_area = gpd.read_file(area_shp_path)

# Load the seismic lines
seismic_lines = gpd.read_file(seismic_lines_gpkg_path)

# Ensure both datasets are in the same coordinate reference system (CRS)
study_area = study_area.to_crs(epsg=3857)  # Web Mercator for basemap compatibility
seismic_lines = seismic_lines.to_crs(epsg=3857)

# Create the main map
fig, ax = plt.subplots(figsize=(10, 10))

# Plot study area outline in red
study_area.boundary.plot(ax=ax, edgecolor='red', linewidth=2, label="Study Area")

# Plot seismic lines in orange
seismic_lines.plot(ax=ax, color='orange', linewidth=1, label="Seismic Lines")

# Add basemap (satellite imagery)
cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, alpha=0.7)

# Customize and label
ax.set_title("Study Area and Seismic Lines in Alberta", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
ax.legend()

# Inset Map (Overview of Alberta)
alberta = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).query("name == 'Canada'")
alberta = alberta.to_crs(epsg=3857)  # Convert to Web Mercator

fig, ax2 = plt.subplots(figsize=(4, 4))
alberta.boundary.plot(ax=ax2, edgecolor='black')
study_area.boundary.plot(ax=ax2, edgecolor='red', linewidth=2, label="Study Area")

ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Overview Map - Alberta")

# Show both maps
plt.show()
# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import box

# File paths
area_shp_path ="E:\Thesis\lideafootprint.shp"  # Update path if necessary
seismic_lines_gpkg_path =  r"E:\Thesis\FLM_shrink\all_group_copy_ID.gpkg"  # Update path if necessary

# Load the study area shapefile
study_area = gpd.read_file(area_shp_path)

# Load the seismic lines
seismic_lines = gpd.read_file(seismic_lines_gpkg_path)

# Ensure both datasets are in the same coordinate reference system (CRS)
study_area = study_area.to_crs(epsg=3857)  # Web Mercator for basemap compatibility
seismic_lines = seismic_lines.to_crs(epsg=3857)

# Create the main map
fig, ax = plt.subplots(figsize=(10, 10))

# Plot study area outline in red
study_area.boundary.plot(ax=ax, edgecolor='red', linewidth=2, label="Study Area")

# Plot seismic lines in orange
seismic_lines.plot(ax=ax, color='orange', linewidth=1, label="Seismic Lines")

# Add basemap (satellite imagery)
cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, attribution='')  # Removes attribution subtitle

# Customize and label
ax.set_title("Study Area and Seismic Lines in Alberta", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
ax.legend()

# ========== Inset Map: Overview of Alberta ==========
# Load Canada country boundaries and filter Alberta
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
canada = world[world.name == "Canada"].to_crs(epsg=3857)

# Create bounding box around the study area
minx, miny, maxx, maxy = study_area.total_bounds
bbox = box(minx, miny, maxx, maxy)
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:3857")

# Create inset map
fig, ax2 = plt.subplots(figsize=(4, 4))
canada.boundary.plot(ax=ax2, edgecolor='black', linewidth=1)
bbox_gdf.boundary.plot(ax=ax2, edgecolor='red', linewidth=2, label="Study Area")

ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Overview Map - Alberta")
ax2.legend()

# Show both maps
plt.show()
# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import box
from matplotlib.patches import Rectangle
import os

# File paths
area_shp_path ="E:\Thesis\lideafootprint.shp"  # Update path if necessary
seismic_lines_gpkg_path =  r"E:\Thesis\FLM_shrink\all_group_copy_ID.gpkg"  # Update path if necessary
 # Update path if necessary
output_geotiff_path = r"E:\Thesis\map_output.tif"  # Update path for exported GeoTIFF

# Load the study area shapefile
study_area = gpd.read_file(area_shp_path)

# Load the seismic lines
seismic_lines = gpd.read_file(seismic_lines_gpkg_path)

# Ensure both datasets are in the same coordinate reference system (CRS)
study_area = study_area.to_crs(epsg=3857)  # Web Mercator for basemap compatibility
seismic_lines = seismic_lines.to_crs(epsg=3857)

# Create a high-resolution figure
fig, ax = plt.subplots(figsize=(15, 15), dpi=300)  # Higher DPI for better quality

# Plot study area outline in red
study_area.boundary.plot(ax=ax, edgecolor='red', linewidth=2, label="Study Area")

# Plot seismic lines in orange
seismic_lines.plot(ax=ax, color='orange', linewidth=1, label="Seismic Lines")

# Add basemap (satellite imagery)
cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, attribution='')

# Customize and label
ax.set_title("Study Area and Seismic Lines in Alberta", fontsize=18)
ax.set_xticks([])
ax.set_yticks([])
ax.legend()

# ========== Add North Star in Bottom Right ==========
def add_north_star(ax, x=0.9, y=0.1, size=0.07):
    """ Adds a North Star (N arrow) at the specified location. """
    ax.annotate('N', xy=(x, y), xytext=(x, y - size),
                arrowprops=dict(facecolor='black', edgecolor='black', width=3, headwidth=10),
                ha='center', va='center', fontsize=16, xycoords='axes fraction')

add_north_star(ax)

# ========== Custom Scale Bar (500m Steps) ==========
def add_scale_bar(ax, length=500, location=(0.1, 0.05), linewidth=3):
    """ Adds a simple scale bar to the plot. """
    x_start, y_start = location
    ax.add_patch(Rectangle((x_start, y_start), length, 10, linewidth=linewidth, edgecolor='black', facecolor='black', transform=ax.transAxes))
    ax.text(x_start + length / 2, y_start + 0.02, f"{length} m", fontsize=12, ha='center', transform=ax.transAxes)

add_scale_bar(ax, length=0.15, location=(0.1, 0.05))  # Adjusted length for relative positioning

# ========== Save Map as GeoTIFF ==========
extent = ax.get_xlim() + ax.get_ylim()
fig.savefig(output_geotiff_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"Map saved as GeoTIFF: {output_geotiff_path}")

# Show the map
plt.show()
