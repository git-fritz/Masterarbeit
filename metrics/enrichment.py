# %%
# THIS CODE TAKES ALL PREVIOUSLY CALCULATED METRICS AND "ENRICHES" THE MODEL INPUT
# GPKG WITH THE VALUES FOR EACH PLOT
# %%
# this step adds dem_diff, dem_SD, rough_avg, rough_SD and all TPI Metrics
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

# File paths (update these with your actual file paths)
gpkg_path = r"E:\Thesis\data\shrink_metrics\shrinkmetrics.gpkg"
dem_path = r"E:\Thesis\data\DEM\nDTM_clip.tif"
roughness_path = r"E:\Thesis\data\DEM\roughness_ndtm.tif"
# tpi_paths = {
#     25: r"E:\Thesis\data\DEM_TPI\tpi25\merged_raster_clean9999_tpi25_filter002.tif",
#     35: r"E:\Thesis\data\DEM_TPI\tpi35\merged_raster_clean9999_tpi35_filter002.tif",
#     55: r"E:\Thesis\data\DEM_TPI\tpi55\merged_raster_clean9999_tpi55_filter002.tif"
# }

# Load the GPKG file with plots
gdf = gpd.read_file(gpkg_path)
print(f"✅ Loaded {len(gdf)} plots from {gpkg_path}")

# Function to safely extract values from zonal stats
def safe_stat(stats, key):
    return [s[key] if s and s[key] is not None else None for s in stats]

# Check if GPKG geometries overlap with the DEM raster
with rasterio.open(dem_path) as src:
    raster_bounds = src.bounds
    print(f"📍 DEM Bounds: {raster_bounds}")
    print(f"📍 GPKG Bounds: {gdf.total_bounds}")
    
    # Check for intersection
    if (
        gdf.total_bounds[0] > raster_bounds.right or
        gdf.total_bounds[2] < raster_bounds.left or
        gdf.total_bounds[1] > raster_bounds.top or
        gdf.total_bounds[3] < raster_bounds.bottom
    ):
        print("❌ Warning: The GPKG plots do not overlap with the DEM raster!")
    else:
        print("✅ GPKG plots overlap with the DEM raster.")

# Handle DEM NoData values
with rasterio.open(dem_path) as src:
    nodata_value = src.nodata
    print(f"🔍 DEM NoData value: {nodata_value}")

# Calculate DEM metrics
dem_stats = zonal_stats(gdf, dem_path, stats=["min", "max", "std"], nodata=nodata_value)
print("📊 DEM Stats Calculated!")

gdf["dem_diff"] = [s["max"] - s["min"] if s["max"] is not None and s["min"] is not None else None for s in dem_stats]
gdf["dem_SD"] = safe_stat(dem_stats, "std")

# Calculate Roughness metrics
roughness_stats = zonal_stats(gdf, roughness_path, stats=["mean", "std"])
print("📊 Roughness Stats Calculated!")

gdf["rough_avg"] = safe_stat(roughness_stats, "mean")
gdf["rough_SD"] = safe_stat(roughness_stats, "std")

# # Calculate TPI metrics for each TPI raster
# for tpi_size, tpi_path in tpi_paths.items():
#     print(f"🔄 Processing TPI {tpi_size}...")
#     tpi_stats = zonal_stats(gdf, tpi_path, stats=["mean", "max", "std"])
    
#     gdf[f"tpi{tpi_size}_mean"] = safe_stat(tpi_stats, "mean")
#     gdf[f"tpi{tpi_size}_max"] = safe_stat(tpi_stats, "max")
#     gdf[f"tpi{tpi_size}_SD"] = safe_stat(tpi_stats, "std")

print("✅ All TPI metrics processed!")

# Save the updated GPKG
output_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v1.gpkg"
gdf.to_file(output_gpkg, driver="GPKG")
print(f"✅ Updated GPKG saved as: {output_gpkg}")

#%%
# this step calculates mound_area and mound_count inside a plot
import geopandas as gpd

# File paths (update these)
plots_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v1.gpkg"
mounds_gpkg = r"E:\Thesis\data\mounds_percentile\mounds_percentile_area05.gpkg"

# Load GeoPackages
plots = gpd.read_file(plots_gpkg)
mounds = gpd.read_file(mounds_gpkg)

# Ensure same CRS
if plots.crs != mounds.crs:
    mounds = mounds.to_crs(plots.crs)

# 🔹 Ensure 'plot_id' exists
if "plot_id" not in plots.columns:
    raise ValueError("❌ 'plot_id' column missing in plots!")

# 🔹 Ensure 'mound_id' exists
if "mound_id" not in mounds.columns:
    raise ValueError("❌ 'mound_id' column missing in mounds!")

# 🔹 Perform spatial intersection (clip mounds to plot boundaries)
intersections = gpd.overlay(mounds, plots, how="intersection", keep_geom_type=False)

# 🔍 Debugging print statements
print(intersections.head())  
print("Columns in intersections:", intersections.columns)
print("plot_id counts:\n", intersections["plot_id"].value_counts())

# 🔹 Ensure 'plot_id' is in intersections
if "plot_id" not in intersections.columns:
    raise ValueError("❌ 'plot_id' missing after overlay!")

# 🔹 Count total number of mounds per plot (each mound is counted separately in each plot)
mound_counts = intersections.groupby("plot_id")["mound_id"].nunique().reset_index(name="mound_count_15percentile")

# 🔹 Calculate total mound area inside each plot
intersections["mound_area_15percentile"] = intersections.area
mound_areas = intersections.groupby("plot_id")["mound_area_15percentile"].sum().reset_index()

# 🔹 Merge results back into plots
plots = plots.merge(mound_counts, on="plot_id", how="left").fillna({"mound_count_15percentile": 0})
plots = plots.merge(mound_areas, on="plot_id", how="left").fillna({"mound_area_15percentile": 0})

# Save updated plots with new attributes
output_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v2.gpkg"
plots.to_file(output_gpkg, driver="GPKG")

print(f"✅ Updated GPKG saved as: {output_gpkg}")

#%%
# this step calculates mound_density and mound_coverage for every tpi value

# import geopandas as gpd

# # File path (update this with the actual file path)
# gpkg_path = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v2.gpkg"

# # Load the GPKG
# gdf = gpd.read_file(gpkg_path)

# # 🔹 Ensure required columns exist
# required_columns = ["plot_area"] + [f"tpi{size}_mound_count" for size in [25, 35, 55]] + [f"tpi{size}_mound_area" for size in [25, 35, 55]]
# missing_columns = [col for col in required_columns if col not in gdf.columns]

# if missing_columns:
#     raise ValueError(f"❌ Missing required columns in GPKG: {missing_columns}")

# # 🔹 Calculate Mound Density (mounds per square meter) for TPI25, TPI35, TPI55
# for size in [25, 35, 55]:
#     gdf[f"tpi{size}_mound_density"] = gdf[f"tpi{size}_mound_count"] / gdf["plot_area"]

# # 🔹 Calculate Mound Coverage (% of plot covered by mounds) for TPI25, TPI35, TPI55
# for size in [25, 35, 55]:
#     gdf[f"tpi{size}_mound_coverage"] = (gdf[f"tpi{size}_mound_area"] / gdf["plot_area"]) * 100

# # Save the updated GPKG
# output_gpkg = r"E:\Thesis\data\metrics_dataset_v1.4.gpkg"
# gdf.to_file(output_gpkg, driver="GPKG")

# print(f"✅ Updated GPKG saved as: {output_gpkg}")

import geopandas as gpd

# File path (update this with the actual file path)
gpkg_path =  r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v2.gpkg"

# Load the GPKG
gdf = gpd.read_file(gpkg_path)

# 🔹 Ensure required columns exist
required_columns = ["plot_area", "mound_count_15percentile", "mound_area_15percentile"]
missing_columns = [col for col in required_columns if col not in gdf.columns]

if missing_columns:
    raise ValueError(f"❌ Missing required columns in GPKG: {missing_columns}")

# 🔹 Calculate Mound Density (mounds per square meter)
gdf["mound_density_15percentile"] = gdf["mound_count_15percentile"] / gdf["plot_area"]

# 🔹 Calculate Mound Coverage (% of plot covered by mounds)
gdf["mound_coverage_15percentile"] = (gdf["mound_area_15percentile"] / gdf["plot_area"]) * 100

# Save the updated GPKG
output_gpkg =  r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v3.gpkg"
gdf.to_file(output_gpkg, driver="GPKG")

print(f"✅ Updated GPKG saved as: {output_gpkg}")

#%%
# updates all tpi values and adds mound_area as column (replaces calc_area_TPI_polygon.py)

# import geopandas as gpd

# # File paths for TPI GPKGs
# tpi_gpkgs = {
#    25: r"E:\Thesis\data\DEM_TPI\tpi25\tpi25_polygon_area01_complx.gpkg",
#    35: r"E:\Thesis\data\DEM_TPI\tpi35\tpi35_polygon_area01_complx.gpkg",
#    55: r"E:\Thesis\data\DEM_TPI\tpi55\tpi55_polygon_area01_complx.gpkg"
# }

# # Loop through each TPI dataset and add "mound_area" column
# for size, tpi_path in tpi_gpkgs.items():
#     # Load the TPI GPKG
#     mounds = gpd.read_file(tpi_path)

#     # Ensure the "area" column exists
#     if "area" not in mounds.columns:
#         raise ValueError(f"❌ 'area' column missing in {tpi_path}!")

#     # Copy "area" values into a new column "mound_area"
#     mounds["mound_area"] = mounds["area"]

#     # Save updated TPI GPKG
#     mounds.to_file(tpi_path, driver="GPKG")
#     print(f"✅ Added 'mound_area' to {tpi_path}")

# print("🚀 All TPI GPKGs have been updated with 'mound_area'!")

#%%
# this step adds max_mound_size and avg_mound_size

# import geopandas as gpd

# # File paths
# plots_gpkg = r"E:\Thesis\data\metrics_dataset_v1.4.gpkg"

# mounds_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v2.gpkg"


# # Load the plots GPKG
# plots = gpd.read_file(plots_gpkg)

# # Ensure 'plot_id' exists
# if "plot_id" not in plots.columns:
#     raise ValueError("❌ 'plot_id' column missing in plots!")

# # Loop through each TPI dataset
# for size, mound_path in mounds_gpkg.items():
#     # Load the mound GPKG
#     mounds = gpd.read_file(mound_path)

#     # Ensure same CRS
#     if plots.crs != mounds.crs:
#         mounds = mounds.to_crs(plots.crs)

#     # Ensure required columns exist
#     if "mound_id" not in mounds.columns:
#         raise ValueError(f"❌ 'mound_id' column missing in {mound_path}!")
#     if "mound_area" not in mounds.columns:
#         raise ValueError(f"❌ 'mound_area' column missing in {mound_path}! Ensure you added it.")

#     # 🔹 Perform spatial intersection (clip mounds to plot boundaries)
#     intersections = gpd.overlay(mounds, plots, how="intersection", keep_geom_type=False)

#     # 🔍 Debugging print statements
#     print(f"🔹 Intersections for TPI{size}:")
#     print(intersections.head())  
#     print("Columns in intersections:", intersections.columns)
#     print("plot_id counts:\n", intersections["plot_id"].value_counts())

#     # 🔹 Ensure 'plot_id' is in intersections
#     if "plot_id" not in intersections.columns:
#         raise ValueError(f"❌ 'plot_id' missing after overlay for TPI{size}!")

#     # 🔹 Calculate Maximum Mound Size per Plot (Using Only Clipped Area)
#     intersections["clipped_mound_area"] = intersections.area  # Only use the clipped area
#     max_mound_size = intersections.groupby("plot_id")["clipped_mound_area"].max().reset_index()
#     max_mound_size.rename(columns={"clipped_mound_area": f"tpi{size}_max_mound_size"}, inplace=True)

#     # 🔹 Calculate Average Mound Size per Plot (Using Only Clipped Area)
#     avg_mound_size = intersections.groupby("plot_id")["clipped_mound_area"].mean().reset_index()
#     avg_mound_size.rename(columns={"clipped_mound_area": f"tpi{size}_avg_mound_size"}, inplace=True)

#     # 🔹 Merge results back into plots
#     plots = plots.merge(max_mound_size, on="plot_id", how="left").fillna({f"tpi{size}_max_mound_size": 0})
#     plots = plots.merge(avg_mound_size, on="plot_id", how="left").fillna({f"tpi{size}_avg_mound_size": 0})

# # Save the updated GPKG
# output_gpkg = r"E:\Thesis\data\metrics_dataset_v1.5.gpkg"
# plots.to_file(output_gpkg, driver="GPKG")

# print(f"✅ Updated GPKG saved as: {output_gpkg}")


import geopandas as gpd

# File path (update if needed)
gpkg_path = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v3.gpkg"

# Load the GPKG
gdf = gpd.read_file(gpkg_path)

# 🔹 Ensure required columns exist
required_columns = ["plot_id", "plot_area", "mound_area_15percentile"]
missing_columns = [col for col in required_columns if col not in gdf.columns]

if missing_columns:
    raise ValueError(f"❌ Missing required columns in GPKG: {missing_columns}")

# 🔹 Calculate Maximum Mound Size per Plot
gdf["max_mound_size_15percentile"] = gdf.groupby("plot_id")["mound_area_15percentile"].transform("max")

# 🔹 Calculate Average Mound Size per Plot
gdf["avg_mound_size_15percentile"] = gdf.groupby("plot_id")["mound_area_15percentile"].transform("mean")

# Save the updated GPKG
output_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v4.gpkg"
gdf.to_file(output_gpkg, driver="GPKG")

print(f"✅ Updated GPKG saved as: {output_gpkg}")

#%% 
# this step adds mean CHM value per plot (uses normal and thresholded chm)
# ADD MEDIAN TO THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# import geopandas as gpd
# import rasterio
# import rasterstats
# from rasterstats import zonal_stats

# # Define input file paths
# gpkg_path =  r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v4.gpkg"  # Replace with actual GPKG file path
# chm_raster_path = r"E:\Thesis\data\CHM\chm_under5.tif"  # Replace with actual CHM raster file path

# # Load the GPKG as a GeoDataFrame
# gdf = gpd.read_file(gpkg_path)

# # Ensure the GPKG and CHM raster have the same CRS
# with rasterio.open(chm_raster_path) as src:
#     chm_crs = src.crs

# if gdf.crs != chm_crs:
#     gdf = gdf.to_crs(chm_crs)

# # Compute zonal statistics (average CHM for each plot)
# stats = zonal_stats(gdf, chm_raster_path, stats=["mean"])

# # Extract mean CHM values and add them to the GeoDataFrame
# gdf["mean_chm_under5"] = [stat["mean"] if stat["mean"] is not None else 0 for stat in stats]

# # Save the updated GeoDataFrame with CHM statistics
# output_gpkg_path = r"E:\Thesis\data\metrics_dataset_v1.8_plot.gpkg"  # Replace with desired output path
# gdf.to_file(output_gpkg_path, driver="GPKG")

# print(f"Updated GPKG saved to: {output_gpkg_path}")
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

# Define input file paths
gpkg_path = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v4.gpkg"  # Replace with actual GPKG file path
chm_rasters = {
    "chm": r"E:\Thesis\data\CHM\merged_chm.tif",               # Full CHM
    "chm_under5": r"E:\Thesis\data\CHM\chm_under5.tif",  # CHM under 5m
    "chm_under2": r"E:\Thesis\data\CHM\chm_under2.tif"   # CHM under 2m
}

# Load the GPKG as a GeoDataFrame
gdf = gpd.read_file(gpkg_path)

# Ensure the GPKG and CHM rasters have the same CRS
with rasterio.open(list(chm_rasters.values())[0]) as src:
    chm_crs = src.crs

if gdf.crs != chm_crs:
    gdf = gdf.to_crs(chm_crs)

# Compute zonal statistics for each CHM raster
for chm_name, chm_path in chm_rasters.items():
    print(f"Processing {chm_name}...")

    # Compute mean and median CHM per plot
    stats = zonal_stats(gdf, chm_path, stats=["mean", "median"])

    # Extract values and add to GeoDataFrame
    gdf[f"mean_{chm_name}"] = [stat["mean"] if stat["mean"] is not None else 0 for stat in stats]
    gdf[f"median_{chm_name}"] = [stat["median"] if stat["median"] is not None else 0 for stat in stats]

# Save the updated GeoDataFrame with CHM statistics
output_gpkg_path = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v5.gpkg"   # Replace with desired output path
gdf.to_file(output_gpkg_path, driver="GPKG")

print(f"✅ Updated GPKG saved to: {output_gpkg_path}")


#%% 
# this step adds plot_shrub_veg_%cover (vegetation below 60cm)

# import geopandas as gpd
# import rasterio
# import numpy as np
# from rasterio.mask import mask

# # File paths
# plots_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v5.gpkg"
# chm_raster = r"E:\Thesis\data\CHM\merged_chm.tif"

# # Load the plots GPKG
# plots = gpd.read_file(plots_gpkg)

# # Ensure 'plot_id' exists
# if "plot_id" not in plots.columns:
#     raise ValueError("❌ 'plot_id' column missing in plots!")

# # Open the CHM raster
# with rasterio.open(chm_raster) as src:
#     chm_crs = src.crs  # Get CRS of CHM raster

#     # Ensure CRS matches between CHM and plots
#     if plots.crs != chm_crs:
#         plots = plots.to_crs(chm_crs)

#     # Initialize a list to store shrub vegetation coverage for each plot
#     shrub_veg_cover = []

#     for _, row in plots.iterrows():
#         geom = [row["geometry"]]  # Convert plot geometry into list format for masking

#         try:
#             # Mask CHM raster using plot geometry
#             out_image, out_transform = mask(src, geom, crop=True, nodata=np.nan)

#             # Flatten the masked array and remove NaN values
#             chm_values = out_image[0].flatten()
#             chm_values = chm_values[~np.isnan(chm_values)]

#             # Compute total number of valid pixels (non-NaN)
#             total_pixels = len(chm_values)
#             if total_pixels == 0:
#                 shrub_veg_cover.append(0)  # No valid data for this plot
#                 continue

#             # Compute number of pixels with vegetation height between 0 and 0.6m
#             shrub_pixels = np.sum((chm_values > 0) & (chm_values <= 0.6))

#             # Calculate % shrub vegetation cover
#             shrub_cover_percent = (shrub_pixels / total_pixels) * 100
#             shrub_veg_cover.append(shrub_cover_percent)

#         except Exception as e:
#             print(f"⚠️ Error processing plot {row['plot_id']}: {e}")
#             shrub_veg_cover.append(0)

# # Add results to the plots GPKG
# plots["plot_shrub_veg_%cover"] = shrub_veg_cover

# # Save updated GPKG
# output_gpkg = r"E:\Thesis\data\metrics_dataset_v1.10_plot.gpkg"
# plots.to_file(output_gpkg, driver="GPKG")

# print(f"✅ Updated GPKG saved as: {output_gpkg}")

#%% 
#this step adds max_, min_ and avg_twi 

# import geopandas as gpd
# import rasterio
# import numpy as np
# from rasterio.mask import mask

# # File paths
# plots_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v5.gpkg"
# twi_raster = r"E:\Thesis\data\TWI\twi_richdem.tif"
# # Load the plots GPKG
# plots = gpd.read_file(plots_gpkg)

# # Ensure 'plot_id' exists
# if "plot_id" not in plots.columns:
#     raise ValueError("❌ 'plot_id' column missing in plots!")

# # Open the TWI raster
# with rasterio.open(twi_raster) as src:
#     twi_crs = src.crs  # Get CRS of TWI raster

#     # Ensure CRS matches between TWI and plots
#     if plots.crs != twi_crs:
#         plots = plots.to_crs(twi_crs)

#     # Initialize lists to store TWI statistics
#     max_twi_values = []
#     min_twi_values = []
#     avg_twi_values = []

#     for _, row in plots.iterrows():
#         geom = [row["geometry"]]  # Convert plot geometry into list format for masking

#         try:
#             # Mask TWI raster using plot geometry
#             out_image, out_transform = mask(src, geom, crop=True, nodata=np.nan)

#             # Flatten the masked array and remove NaN values
#             twi_values = out_image[0].flatten()
#             twi_values = twi_values[~np.isnan(twi_values)]

#             # Check if there are valid TWI values
#             if len(twi_values) == 0:
#                 max_twi_values.append(np.nan)
#                 min_twi_values.append(np.nan)
#                 avg_twi_values.append(np.nan)
#             else:
#                 max_twi_values.append(np.max(twi_values))
#                 min_twi_values.append(np.min(twi_values))
#                 avg_twi_values.append(np.mean(twi_values))

#         except Exception as e:
#             print(f"⚠️ Error processing plot {row['plot_id']}: {e}")
#             max_twi_values.append(np.nan)
#             min_twi_values.append(np.nan)
#             avg_twi_values.append(np.nan)

# # Add results to the plots GPKG
# plots["plot_max_twi"] = max_twi_values
# plots["plot_min_twi"] = min_twi_values
# plots["plot_avg_twi"] = avg_twi_values

# # Save updated GPKG
# output_gpkg = r"E:\Thesis\data\metrics_dataset_v1.10_plot.gpkg"
# plots.to_file(output_gpkg, driver="GPKG")

# print(f"✅ Updated GPKG saved as: {output_gpkg}")

#%% 
# this step adds aspect and slope avg

import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask

# File paths
plots_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v5.gpkg"
slope_raster = r"E:\Thesis\data\DEM\slope_ndtm.tif"
aspect_raster = r"E:\Thesis\data\DEM\aspect_ndtm.tif"

# Load the plots GPKG
plots = gpd.read_file(plots_gpkg)

# Ensure 'plot_id' exists
if "plot_id" not in plots.columns:
    raise ValueError("❌ 'plot_id' column missing in plots!")

# Open the slope raster
with rasterio.open(slope_raster) as src:
    slope_crs = src.crs  # Get CRS of slope raster

    # Ensure CRS matches between slope and plots
    if plots.crs != slope_crs:
        plots = plots.to_crs(slope_crs)

    # Initialize lists to store slope statistics
    avg_slope_values = []

    for _, row in plots.iterrows():
        geom = [row["geometry"]]  # Convert plot geometry into list format for masking

        try:
            # Mask slope raster using plot geometry
            out_image, out_transform = mask(src, geom, crop=True, nodata=np.nan)

            # Flatten the masked array and remove NaN values
            slope_values = out_image[0].flatten()
            slope_values = slope_values[~np.isnan(slope_values)]

            # Check if there are valid slope values
            if len(slope_values) == 0:
                avg_slope_values.append(np.nan)
            else:
                avg_slope_values.append(np.mean(slope_values))

        except Exception as e:
            print(f"⚠️ Error processing plot {row['plot_id']}: {e}")
            avg_slope_values.append(np.nan)

# Open the aspect raster
with rasterio.open(aspect_raster) as src:
    aspect_crs = src.crs  # Get CRS of aspect raster

    # Ensure CRS matches between aspect and plots
    if plots.crs != aspect_crs:
        plots = plots.to_crs(aspect_crs)

    # Initialize lists to store aspect statistics
    avg_aspect_values = []

    for _, row in plots.iterrows():
        geom = [row["geometry"]]  # Convert plot geometry into list format for masking

        try:
            # Mask aspect raster using plot geometry
            out_image, out_transform = mask(src, geom, crop=True, nodata=np.nan)

            # Flatten the masked array and remove NaN values
            aspect_values = out_image[0].flatten()
            aspect_values = aspect_values[~np.isnan(aspect_values)]

            # Check if there are valid aspect values
            if len(aspect_values) == 0:
                avg_aspect_values.append(np.nan)
            else:
                avg_aspect_values.append(np.mean(aspect_values))

        except Exception as e:
            print(f"⚠️ Error processing plot {row['plot_id']}: {e}")
            avg_aspect_values.append(np.nan)

# Add results to the plots GPKG
plots["plot_avg_slope"] = avg_slope_values
plots["plot_avg_aspect"] = avg_aspect_values

# Save updated GPKG
output_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v6.gpkg"
plots.to_file(output_gpkg, driver="GPKG")

print(f"✅ Updated GPKG saved as: {output_gpkg}")
#%%
# This step calculates the number of trees inside a plot, tree density, and trees per hectare
import geopandas as gpd

# File paths
plots_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v6.gpkg"
trees_gpkg = r"E:\Thesis\seedlings_max.gpkg"

# Load the data
plots = gpd.read_file(plots_gpkg)
trees = gpd.read_file(trees_gpkg)

# Ensure same CRS
if plots.crs != trees.crs:
    trees = trees.to_crs(plots.crs)

# Compute tree centroids
trees["centroid"] = trees.geometry.centroid

# Convert centroids to a GeoDataFrame
tree_centroids = gpd.GeoDataFrame(trees, geometry="centroid", crs=trees.crs)

# Perform spatial join using centroids to check if they fall within plots
tree_intersections = gpd.sjoin(tree_centroids, plots, how="inner", predicate="within")

# Debugging: Check if the join worked
if tree_intersections.empty:
    raise ValueError("❌ Spatial join found no trees inside any plots!")

# Ensure `plot_id` exists in tree_intersections
if "plot_id" not in tree_intersections.columns:
    raise ValueError("❌ 'plot_id' is missing from tree_intersections after the join!")

# Count trees per plot
tree_counts = tree_intersections.groupby("plot_id").size().reset_index(name="tree_count")

# Debugging: Check if tree_counts has valid values
print("🟢 Tree counts per plot preview:\n", tree_counts.head())

# Ensure `plot_id` exists in `plots`
if "plot_id" not in plots.columns:
    raise ValueError("❌ 'plot_id' is missing from plots!")

# Merge results back into plots
plots = plots.merge(tree_counts, on="plot_id", how="left").fillna({"tree_count": 0})

# Debugging: Ensure `tree_count` was added correctly
if "tree_count" not in plots.columns:
    raise ValueError("❌ 'tree_count' column is missing after merge!")

# Calculate Tree Density (trees per square meter)
plots["tree_density"] = plots["tree_count"] / plots["plot_area"]

# Calculate Trees per Hectare (extrapolated from plot basis)
plots["trees_per_ha"] = plots["tree_density"] * 10000

# Save the updated GPKG
output_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v7.gpkg"
plots.to_file(output_gpkg, driver="GPKG")

print(f"✅ Updated GPKG saved as: {output_gpkg}")


import geopandas as gpd
import rasterio
import numpy as np
from rasterstats import zonal_stats
import tempfile
import os

import geopandas as gpd
import rasterio
import numpy as np
from rasterstats import zonal_stats

import geopandas as gpd
import rasterio
import numpy as np
from rasterstats import zonal_stats
import tempfile
import os

#%%
# This step calculates percent cover of vegetation above 60cm and binary recovery classification

# File paths
plots_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v7.gpkg"
chm_raster = r"E:\Thesis\data\CHM\merged_chm.tif"

# Load the plots GPKG
plots = gpd.read_file(plots_gpkg)

# Ensure 'plot_id' exists
if "plot_id" not in plots.columns:
    raise ValueError("❌ Missing 'plot_id' column in plots!")

# Open CHM raster
with rasterio.open(chm_raster) as src:
    chm_array = src.read(1).astype(np.float32)  # Convert to float32 to handle NoData
    chm_meta = src.meta  # Store metadata

    # Mask NoData values
    nodata_value = src.nodata if src.nodata is not None else np.nan
    chm_array = np.where(chm_array == nodata_value, np.nan, chm_array)  # Replace NoData with NaN

    # Create a binary mask where CHM > 60cm (0.6m)
    veg_mask = np.where(chm_array > 0.6, 1, 0).astype(np.uint8)

# Create a temporary raster for the binary mask
with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_raster:
    temp_raster_path = temp_raster.name

with rasterio.open(
    temp_raster_path, "w", driver="GTiff",
    height=veg_mask.shape[0], width=veg_mask.shape[1],
    count=1, dtype=np.uint8, crs=src.crs,
    transform=src.transform, nodata=0
) as dst:
    dst.write(veg_mask, 1)

# Compute zonal statistics to get the total number of pixels inside each plot
plot_pixel_stats = zonal_stats(plots, chm_raster, stats=["count"], affine=src.transform, nodata=nodata_value)

# Extract the total number of pixels inside each plot
plots["plot_pixels"] = [stat["count"] if stat and "count" in stat else 0 for stat in plot_pixel_stats]

# Compute zonal statistics using the binary mask raster
chm_stats = zonal_stats(plots, temp_raster_path, stats=["sum"], affine=src.transform, nodata=0)

# Extract the number of pixels with vegetation above 60cm
plots["veg_pixels_above_60cm"] = [stat["sum"] if stat and "sum" in stat else 0 for stat in chm_stats]

# Compute percent cover of vegetation above 60cm based on total pixels inside plot
plots["veg_cover_percent_above_60cm"] = (plots["veg_pixels_above_60cm"] / plots["plot_pixels"]) * 100

# Create a binary mask to classify recovery (1 if vegetation cover > 20%, else 0)
plots["binary_recovery"] = (plots["veg_cover_percent_above_60cm"] > 20).astype(int)

# Save the updated GPKG
output_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v8.gpkg"
plots.to_file(output_gpkg, driver="GPKG")

# Remove temporary raster file after processing
os.remove(temp_raster_path)

print(f"✅ Updated GPKG saved as: {output_gpkg}")

# %%

# %%

#%%
# this step calculates hollow_area and hollow_count inside a plot
import geopandas as gpd

# File paths (update these)
plots_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v8.gpkg"
mounds_gpkg = r"E:\Thesis\testing\hollows_low15_05.gpkg"

# Load GeoPackages
plots = gpd.read_file(plots_gpkg)
mounds = gpd.read_file(mounds_gpkg)

# Ensure same CRS
if plots.crs != mounds.crs:
    mounds = mounds.to_crs(plots.crs)

# 🔹 Ensure 'plot_id' exists
if "plot_id" not in plots.columns:
    raise ValueError("❌ 'plot_id' column missing in plots!")

# 🔹 Ensure 'hollow_id' exists
if "hollow_id" not in mounds.columns:
    raise ValueError("❌ 'hollow_id' column missing in mounds!")

# 🔹 Perform spatial intersection (clip mounds to plot boundaries)
intersections = gpd.overlay(mounds, plots, how="intersection", keep_geom_type=False)

# 🔍 Debugging print statements
print(intersections.head())  
print("Columns in intersections:", intersections.columns)
print("plot_id counts:\n", intersections["plot_id"].value_counts())

# 🔹 Ensure 'plot_id' is in intersections
if "plot_id" not in intersections.columns:
    raise ValueError("❌ 'plot_id' missing after overlay!")

# 🔹 Count total number of mounds per plot (each mound is counted separately in each plot)
mound_counts = intersections.groupby("plot_id")["hollow_id"].nunique().reset_index(name="hollow_count_15percentile")

# 🔹 Calculate total mound area inside each plot
intersections["hollow_area_low15percentile"] = intersections.area
mound_areas = intersections.groupby("plot_id")["hollow_area_low15percentile"].sum().reset_index()

# 🔹 Merge results back into plots
plots = plots.merge(mound_counts, on="plot_id", how="left").fillna({"hollow_count_15percentile": 0})
plots = plots.merge(mound_areas, on="plot_id", how="left").fillna({"hollow_area_low15percentile": 0})

# Save updated plots with new attributes
output_gpkg = r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v9.gpkg"
plots.to_file(output_gpkg, driver="GPKG")

print(f"✅ Updated GPKG saved as: {output_gpkg}")

# %%

import geopandas as gpd

# File path (update this with the actual file path)
gpkg_path =  r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v9.gpkg"

# Load the GPKG
gdf = gpd.read_file(gpkg_path)

# 🔹 Ensure required columns exist
required_columns = ["plot_area", "hollow_count_15percentile", "hollow_area_low15percentile"]
missing_columns = [col for col in required_columns if col not in gdf.columns]

if missing_columns:
    raise ValueError(f"❌ Missing required columns in GPKG: {missing_columns}")

# 🔹 Calculate Mound Density (mounds per square meter)
gdf["hollow_density_low15percentile"] = gdf["hollow_count_15percentile"] / gdf["plot_area"]

# 🔹 Calculate Mound Coverage (% of plot covered by mounds)
gdf["hollow_coverage_low15percentile"] = (gdf["hollow_area_low15percentile"] / gdf["plot_area"]) * 100

# Save the updated GPKG
output_gpkg =  r"E:\Thesis\data\shrink_metrics\shrinkmetrics_v10.gpkg"
gdf.to_file(output_gpkg, driver="GPKG")

print(f"✅ Updated GPKG saved as: {output_gpkg}")
