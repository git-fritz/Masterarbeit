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

def process_segment(unique_group, chm_src, ndtm_src, resolution, buffer_size=20):
    """
    Process a single segment to compute adjacency metrics (per segment)
    and vegetation metrics (per side).
    """
    results = []
    segment_groups = unique_group.groupby("SegmentID")

    for segment_id, segment_group in segment_groups:
        print(f"Processing SegmentID: {segment_id}")

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
        adjacency_tree3m_coverage = calculate_adjacency_coverage(buffer_data, buffer_mask, height_threshold_cm=300)
        adjacency_tree1m_coverage = calculate_adjacency_coverage(buffer_data, buffer_mask, height_threshold_cm=100)

        # Clip nDTM data for the full segment
        ndtm_clipped, mask = read_clipped_data(ndtm_src, combined_polygon)
        ndtm_metrics = calculate_metrics(ndtm_clipped, mask, resolution, segment_area)
        # print(f"  nDTM Metrics: {ndtm_metrics}")

        # Process each side within the segment
        side_groups = segment_group.groupby("side")
        for side, side_group in side_groups:
            # print(f"  Processing Side: {side} of SegmentID: {segment_id}")

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

            valid_pixels = chm_clipped[mask]  # Select only pixels within the inner mask

            # Calculate CHM metrics for new height bins
            veg_60_100cm = (valid_pixels > 0.6) & (valid_pixels <= 1.0)
            veg_100_200cm = (valid_pixels > 1.0) & (valid_pixels <= 2.0)
            veg_200_300cm = (valid_pixels > 2.0) & (valid_pixels <= 3.0)
            veg_300_500cm = (valid_pixels > 3.0) & (valid_pixels <= 5.0)
            veg_above_500cm = valid_pixels > 5.0

            veg_60_100cm_area = np.sum(veg_60_100cm) * resolution ** 2
            veg_100_200cm_area = np.sum(veg_100_200cm) * resolution ** 2
            veg_200_300cm_area = np.sum(veg_200_300cm) * resolution ** 2
            veg_300_500cm_area = np.sum(veg_300_500cm) * resolution ** 2
            veg_above_500cm_area = np.sum(veg_above_500cm) * resolution ** 2

            veg_60_100cm_coverage = (veg_60_100cm_area / side_area) * 100 if side_area > 0 else 0
            veg_100_200cm_coverage = (veg_100_200cm_area / side_area) * 100 if side_area > 0 else 0
            veg_200_300cm_coverage = (veg_200_300cm_area / side_area) * 100 if side_area > 0 else 0
            veg_300_500cm_coverage = (veg_300_500cm_area / side_area) * 100 if side_area > 0 else 0
            veg_above_500cm_coverage = (veg_above_500cm_area / side_area) * 100 if side_area > 0 else 0

            # Collect attributes for this side
            original_attributes = side_group.iloc[0].to_dict()
            original_attributes.pop("geometry", None)

            # Append results for this side
            results.append({
                **original_attributes,
                "geometry": side_polygon,  # Use side-specific geometry
                "segment_id": segment_id,
                "segment_area": segment_area,
                "side_area": side_area,
                "side": side,
                "60_100cm_veg_coverage": veg_60_100cm_coverage,
                "100_200cm_veg_coverage": veg_100_200cm_coverage,
                "200_300cm_veg_coverage": veg_200_300cm_coverage,
                "300_500cm_veg_coverage": veg_300_500cm_coverage,
                ">500cm_veg_coverage": veg_above_500cm_coverage,
                "adjacency_tree13m_coverage": adjacency_tree13m_coverage,  # Same for all sides
                "adjacency_tree8m_coverage": adjacency_tree8m_coverage,
                "adjacency_tree3m_coverage": adjacency_tree3m_coverage,
                "adjacency_tree1m_coverage": adjacency_tree1m_coverage,
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


def process_chunks_with_seedlings(footprint_path, seedling_folder, chm_path, ndtm_path, output_path):
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

    print(f"Processing {len(chunk_names)} chunks...")

    all_results = []
    with tqdm(total=len(chunk_names), desc="Processing Chunks") as pbar:
        with Pool(16) as pool:
            chunk_results = pool.starmap(
                process_chunk,
                [(chunk, polygons_gdf, chm_path, ndtm_path) for chunk in chunk_names]
            )
            for result in chunk_results:
                all_results.extend(result)
                pbar.update()

    save_results(all_results, output_path, polygons_gdf.crs)


def main():
    seedling_folder = '/media/irina/My Book/Recovery/DATA/vector_data/Seedlings_ens_14jan24_IoU0.5_plots_v5.3/'
    footprint_path = "/media/irina/My Book/Recovery/DATA/vector_data/FLM_2024/Assessments/LiDea_2024_v5.4_plots100m2.gpkg"
    chm_path = '/media/irina/My Book/Recovery/DATA/raw/CHM/LiDea_CHM30cm_2024.tif'
    ndtm_path = '/media/irina/My Book/Recovery/DATA/raw/nDTM/LiDea_nDTM30cm_2024.tif'
    output_path = footprint_path.replace('.gpkg', '_extra_ass_v2.gpkg')

    process_chunks_with_seedlings(footprint_path, seedling_folder, chm_path, ndtm_path, output_path)


if __name__ == "__main__":
    main()
