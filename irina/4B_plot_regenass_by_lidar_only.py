import os
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
from tqdm import tqdm
import geopandas as gpd
from multiprocessing import Pool, cpu_count
from rasterio.features import geometry_mask
from rasterio.coords import disjoint_bounds


from rasterio.features import geometry_mask

def read_clipped_data(src, geom, block_size=1024):
    """
    Reads and clips raster data to the exact geometry with block-level validation.
    Ensures only inner pixels are considered using a mask.
    """
    geom_mapping = [geom.__geo_interface__]

    # Check if geometry overlaps with raster bounds
    if disjoint_bounds(src.bounds, geom.bounds):
        print("Geometry does not overlap with raster. Skipping...")
        return None, None

    try:
        # Mask and crop the data to the geometry
        clipped_data, clipped_transform = mask(src, geom_mapping, crop=True, nodata=255)

        # Create a binary mask for the geometry
        binary_mask = geometry_mask([geom_mapping[0]], out_shape=clipped_data.shape[1:], transform=clipped_transform, invert=True)

        # Apply the binary mask to keep only inner pixels
        inner_data = np.where(binary_mask, clipped_data[0], 255)  # Set outer pixels to nodata (255)

        return inner_data, binary_mask  # Return masked data and the binary mask
    except rasterio.errors.RasterioIOError as e:
        print(f"Error clipping data: {e}")
        return None, None


def calculate_metrics(data, mask, resolution, segment_area):
    """
    Calculate metrics such as area, coverage, and statistics for a raster dataset.
    """
    # Extract valid data
    valid_data = data[mask]
    valid_data = valid_data[valid_data < 5]
    # print('Valid data minimum: {} and maximum: {}'.format(valid_data.min(), valid_data.max()))

    if np.sum(valid_data) == 0:
        return {"iqr": None,
            "hummock_coverage": 0,
            "hollow_coverage": 0
        }

    # Calculate statistics
    q1, median, q3 = np.percentile(valid_data, [10, 50, 90])
    iqr = q3 - q1

    # Calculate hummock and hollow coverage
    hummock_pixels = np.sum(valid_data > 0.15)
    hummock_area = np.sum(hummock_pixels) * (resolution ** 2)  # Total area in square meters

    hollow_pixels = np.sum(valid_data < -0.15)
    hollow_area = np.sum(hollow_pixels) * (resolution ** 2)  # Total area in square meters

    return {
        "iqr": iqr,
        "hummock_coverage": 100 * hummock_area / segment_area  if segment_area > 0 else 0,
        "hollow_coverage": 100 * hollow_area / segment_area  if segment_area > 0 else 0
    }


def calculate_adjacency_coverage(buffer_data, mask, height_threshold_cm):
    """
    Calculate adjacency coverage for shrub/tree heights using a binary mask.
    Args:
        buffer_data (ndarray): CHM or height data array.
        mask (ndarray): Binary mask where `True` indicates valid pixels.
        height_threshold_cm (float): Height threshold in centimeters.

    Returns:
        float: Percentage of pixels exceeding the height threshold within the mask.
    """
    height_threshold_m = height_threshold_cm / 100.0  # Convert to meters

    # Apply the mask to filter valid pixels
    valid_pixels = buffer_data[mask]

    # Count total valid pixels
    total_pixels = len(valid_pixels)

    if total_pixels == 0:
        return 0  # Avoid division by zero if no valid pixels exist

    # Calculate pixels exceeding the height threshold
    pixels_above_threshold = np.sum(valid_pixels > height_threshold_m)

    # Calculate adjacency coverage as a percentage
    adjacency_coverage = (pixels_above_threshold / total_pixels) * 100

    return adjacency_coverage

def calculate_height_bin_metrics(valid_pixels, resolution, area, height_bins):
    """
    Calculate vegetation metrics (area and coverage) for specified height bins.

    Args:
        valid_pixels (ndarray): Array of valid CHM pixels within the mask.
        resolution (float): Spatial resolution of the raster (in meters).
        area (float): Area of the polygon (in square meters).
        height_bins (dict): Dictionary with height bin ranges as keys (tuples) and descriptive names as values.

    Returns:
        dict: Dictionary with coverage percentage for each height bin.
    """
    metrics = {}

    for bin_range, bin_name in height_bins.items():
        min_height, max_height = bin_range

        # Select pixels within the height bin
        if max_height is None:  # Handle the case for ">max_height" (e.g., >5.0)
            bin_pixels = valid_pixels > min_height
        else:
            bin_pixels = (valid_pixels > min_height) & (valid_pixels <= max_height)

        # Calculate area and coverage
        bin_area = np.sum(bin_pixels) * (resolution ** 2)
        bin_coverage = (bin_area / area) * 100 if area > 0 else 0

        # Add coverage to metrics
        metrics[f"{bin_name}_%cover"] = bin_coverage

    return metrics

def process_segment(unique_group, chm_src, ndtm_src, resolution, buffer_size=20):
    """
    Process a single segment to compute adjacency metrics (per segment),
    vegetation metrics (per side), and plot-level metrics.
    """
    results = []
    segment_groups = unique_group.groupby("SegmentID")

    for segment_id, segment_group in segment_groups:
        # print(f"Processing SegmentID: {segment_id}")

        # Combine geometries for the full segment
        combined_polygon = segment_group.geometry.union_all()
        segment_area = combined_polygon.area

        if combined_polygon.is_empty or combined_polygon is None:
            print(f"Warning: Combined polygon is empty for SegmentID {segment_id}.")
            continue

        # Calculate buffer for adjacency metrics
        centroid = combined_polygon.centroid
        buffer_polygon = centroid.buffer(buffer_size).difference(combined_polygon.buffer(2))

        # Clip CHM data for adjacency metrics
        buffer_data, buffer_mask = read_clipped_data(chm_src, buffer_polygon)

        # Calculate adjacency metrics for the full segment
        adjacency_tree13m_coverage = calculate_adjacency_coverage(buffer_data, buffer_mask, height_threshold_cm=1300)
        adjacency_tree8m_coverage = calculate_adjacency_coverage(buffer_data, buffer_mask, height_threshold_cm=800)
        adjacency_tree5m_coverage = calculate_adjacency_coverage(buffer_data, buffer_mask, height_threshold_cm=500)
        adjacency_tree3m_coverage = calculate_adjacency_coverage(buffer_data, buffer_mask, height_threshold_cm=300)
        adjacency_tree1m_coverage = calculate_adjacency_coverage(buffer_data, buffer_mask, height_threshold_cm=150)

        # Clip nDTM data for the full segment
        ndtm_clipped, mask = read_clipped_data(ndtm_src, combined_polygon)
        ndtm_metrics = calculate_metrics(ndtm_clipped, mask, resolution, segment_area)

        # Define descriptive names for the bins
        height_bins = {
            (0.6, 1.0): "short_veg",
            (1.0, 3.0): "medium_veg",
            (3.0, 5.0): "tall_veg",
            (5.0, None): "forest"
        }

        chm_clipped, mask = read_clipped_data(chm_src, combined_polygon)
        valid_pixels = chm_clipped[mask]  # Pixels within mask
        veg_metrics_side = calculate_height_bin_metrics(valid_pixels, resolution, segment_area, height_bins)
        segment_veg_metrics = {f"segment_{k}": v for k, v in veg_metrics_side.items()}

        # Process each side within the segment
        side_groups = segment_group.groupby("side")
        for side, side_group in side_groups:
            # print(f"  Processing Side {side} of SegmentID {segment_id}")

            # Combine geometries for this side
            side_polygon = side_group.geometry.union_all()
            side_area = side_polygon.area

            if side_polygon.is_empty or side_polygon is None:
                print(f"  Warning: Polygon is empty for SegmentID {segment_id}, Side {side}.")
                continue

            # Clip CHM data for this side
            chm_clipped, mask = read_clipped_data(chm_src, side_polygon)

            if chm_clipped is None:
                print(f"  Skipping SegmentID {segment_id}, Side {side} due to invalid CHM data.")
                continue

            valid_pixels = chm_clipped[mask]  # Pixels within mask
            veg_metrics_side = calculate_height_bin_metrics(valid_pixels, resolution, side_area, height_bins)
            side_veg_metrics = {f"side_{k}": v for k, v in veg_metrics_side.items()}

            # Process plots within the side
            plot_groups = side_group.groupby("plot_id")
            for plot_id, plot_group in plot_groups:
                # print(f"    Processing Plot {plot_id} of Side {side}, SegmentID {segment_id}")

                # Combine geometries for this plot
                plot_polygon = plot_group.geometry.union_all()
                plot_area = plot_polygon.area

                if plot_polygon.is_empty or plot_polygon is None:
                    print(f"    Warning: Plot is empty for PlotID {plot_id}. Skipping...")
                    continue

                # Clip CHM data for this plot
                plot_chm_clipped, plot_mask = read_clipped_data(chm_src, plot_polygon)

                if plot_chm_clipped is None:
                    print(f"    Skipping PlotID {plot_id} due to invalid CHM data.")
                    continue

                valid_pixels = plot_chm_clipped[plot_mask]  # Pixels within mask

                # Calculate vegetation metrics for height bins
                veg_metrics = calculate_height_bin_metrics(valid_pixels, resolution, plot_area, height_bins)
                plot_veg_metrics = {f"plot_{k}": v for k, v in veg_metrics.items()}

                # Collect attributes for this plot
                original_attributes = plot_group.iloc[0].to_dict()
                original_attributes.pop("geometry", None)

                # Append results for this plot
                results.append({
                    **original_attributes,
                    "geometry": plot_polygon,  # Use plot-specific geometry
                    "segment_id": segment_id,
                    "side": side,
                    "plot_id": plot_id,
                    "segment_area": segment_area,
                    "side_area": side_area,
                    "plot_area": plot_area,
                    **plot_veg_metrics,
                    **side_veg_metrics,
                    **segment_veg_metrics,
                    "adjacency_tree13m_coverage": adjacency_tree13m_coverage,
                    "adjacency_tree8m_coverage": adjacency_tree8m_coverage,
                    "adjacency_tree5m_coverage": adjacency_tree5m_coverage,
                    "adjacency_tree3m_coverage": adjacency_tree3m_coverage,
                    "adjacency_tree1.5m_coverage": adjacency_tree1m_coverage,
                    "hummock_coverage": ndtm_metrics["hummock_coverage"],
                    "hollow_coverage": ndtm_metrics["hollow_coverage"],
                    "ndtm_iqr": ndtm_metrics["iqr"],
                })

    return results



def save_results(results, output_path, crs):
    """
    Save results to a GeoPackage.
    """
    gdf = gpd.GeoDataFrame(results, crs=crs)
    gdf.to_file(output_path, driver="GPKG")
    print(f"Results saved to {output_path}")


def process_chunk(chunk_name, polygons_gdf, chm_path, ndtm_path):
    """
    Process a single chunk.
    """
    chunk_data = polygons_gdf[polygons_gdf["chunk"] == chunk_name]
    results = []

    for _, unique_group in chunk_data.groupby("UniqueID"):
        uniqueid_geom = unique_group.union_all()

        # Process segments
        with rasterio.open(chm_path) as chm_src:
            chm_meta = chm_src.meta
            with rasterio.open(ndtm_path) as ndtm_src:
                segment_results = process_segment(
                    unique_group, chm_src, ndtm_src, chm_meta["transform"][0]
            )
                results.extend(segment_results)

    return results


def process_chunks_with_seedlings(footprint_path, chm_path, ndtm_path, output_path):
    """
    Process all chunks in parallel and calculate metrics.
    """
    polygons_gdf = gpd.read_file(footprint_path)
    polygons_gdf = polygons_gdf[polygons_gdf.geometry.notnull()]
    polygons_gdf.set_geometry("geometry", inplace=True)

    polygons_gdf["geometry"] = polygons_gdf["geometry"].apply(
        lambda geom: geom.buffer(0) if not geom.is_valid else geom)

    chunk_names = polygons_gdf['chunk'].unique()
    # chunk_names = chunk_names[:2]

    # print(f"Processing {len(chunk_names)} chunks...")

    all_results = []
    with tqdm(total=len(chunk_names), desc="Processing Chunks") as pbar:
        with Pool(24) as pool:
            chunk_results = pool.starmap(
                process_chunk,
                [(chunk, polygons_gdf, chm_path, ndtm_path) for chunk in chunk_names]
            )
            for result in chunk_results:
                all_results.extend(result)
                pbar.update()

    save_results(all_results, output_path, polygons_gdf.crs)


def main():
    footprint_path = r"E:\try_linux_man\test\fix_ID_segments100m2.gpkg"
    chm_path = r"E:\Thesis\merge_chm\merged_chm.tif"
    ndtm_path = r"E:\Thesis\merge\merged_raster.tif"
    output_path = footprint_path.replace('.gpkg', '_extra_ass_v3.gpkg')

    process_chunks_with_seedlings(footprint_path, chm_path, ndtm_path, output_path)


if __name__ == "__main__":
    main()
