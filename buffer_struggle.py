
import geopandas as gpd
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import split

# File Paths
buffer_gpkg = r"E:\Thesis\data\buffer_cut.gpkg"  # Input buffer (single object)
output_gpkg = r"E:\Thesis\data\buffer_100cut.gpkg" 
# Load the buffer dataset
gdf_buffer = gpd.read_file(buffer_gpkg)

# Ensure all geometries are single polygons
gdf_buffer = gdf_buffer.explode(ignore_index=True)

# Define segment size
buffer_width = 5  # Buffer width is 5m
segment_length = 20  # Length of each segment (5m x 20m = 100m²)

# Function to segment polygons into 100m² sections
def segment_polygon(polygon, segment_length):
    """
    Splits a polygon into smaller 100m² segments along its longer axis.
    """
    if polygon.geom_type not in ["Polygon", "MultiPolygon"]:
        return []  # Skip invalid geometries
    
    if polygon.geom_type == "MultiPolygon":
        polygons = list(polygon.geoms)
    else:
        polygons = [polygon]

    segmented_parts = []

    for poly in polygons:
        minx, miny, maxx, maxy = poly.bounds
        length = maxx - minx if (maxx - minx) > (maxy - miny) else maxy - miny

        # Create cutting lines at 20m intervals along the longer axis
        num_segments = int(length // segment_length)
        cutting_lines = [
            LineString([(minx + i * segment_length, miny), (minx + i * segment_length, maxy)])
            if (maxx - minx) > (maxy - miny)  # Horizontal or vertical splitting
            else LineString([(minx, miny + i * segment_length), (maxx, miny + i * segment_length)])
            for i in range(1, num_segments + 1)
        ]

        # Perform the split operation
        for line in cutting_lines:
            try:
                poly = split(poly, line)
            except Exception:
                continue
        
        if isinstance(poly, Polygon):
            segmented_parts.append(poly)
        else:
            segmented_parts.extend(list(poly.geoms))

    return segmented_parts

# Apply segmentation to each polygon in the dataset
segmented_polygons = []
for _, row in gdf_buffer.iterrows():
    segmented_polygons.extend(segment_polygon(row.geometry, segment_length))

# Create a new GeoDataFrame with the segmented polygons
gdf_segmented = gpd.GeoDataFrame(geometry=segmented_polygons, crs=gdf_buffer.crs)

# Save segmented buffer to GeoPackage
gdf_segmented.to_file(output_gpkg, driver="GPKG")

print(f"Segmented buffer saved as: {output_gpkg}")
