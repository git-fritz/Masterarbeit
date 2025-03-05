# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:16:00 2025

@author: Felix
"""

from osgeo import gdal, ogr

# Input raster (TPI Raster) - Update this path
input_raster = "path/to/your_tpi_raster.tif"

# Output vector file (GeoPackage or Shapefile)
output_vector = "path/to/output_tpi_polygons.gpkg"

# Open raster
src_ds = gdal.Open(input_raster)
src_band = src_ds.GetRasterBand(1)  # Assuming single-band raster

# Create output vector dataset
drv = ogr.GetDriverByName("GPKG")  # Change to "ESRI Shapefile" if needed
out_ds = drv.CreateDataSource(output_vector)
out_layer = out_ds.CreateLayer("TPI_Polygons", srs=None, geom_type=ogr.wkbPolygon)

# Add field to store TPI values
field_defn = ogr.FieldDefn("TPI", ogr.OFTReal)
out_layer.CreateField(field_defn)

# Polygonize (convert raster to vector)
gdal.Polygonize(src_band, None, out_layer, 0, [], callback=None)

# Cleanup
src_ds, out_ds = None, None
print(f"Polygonized TPI raster saved to {output_vector}")
