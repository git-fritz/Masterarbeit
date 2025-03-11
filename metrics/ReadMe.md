# LiDAR Terrain and Vegetation Metrics Workflow

## Overview
This repository contains a collection of **Python scripts** for computing various **terrain and vegetation metrics** from **Digital Terrain Models (DTM), Canopy Height Models (CHM), and other LiDAR-derived rasters**. The scripts process spatial datasets and generate metrics that can be used for environmental analysis, ecological modeling, and landscape classification.

## Features
✅ **Topographic Position Index (TPI)** calculation at multiple scales (e.g., 25m, 35m, 55m)  
✅ **Topographic Wetness Index (TWI)** using slope and flow accumulation  
✅ **Aspect and Slope** extraction from DTM  
✅ **Enrichment of plots** with computed metrics (zonal statistics)  
✅ **Automated data processing with raster clipping, resampling, and aggregation**  

## Requirements
### **1. Software Dependencies**
Ensure the following software is installed:
- Python 3.x
- [GDAL](https://gdal.org/) (for raster operations)
- [LASTools](https://rapidlasso.com/lastools/) (if using additional LiDAR pre-processing)

### **2. Required Python Packages**
The scripts use several Python packages, which must be installed before running:
```bash
pip install numpy rasterio richdem geopandas rasterstats scipy
```

## Scripts & Functionality
### **1. `tpi_working.py` - Topographic Position Index (TPI) Calculation**
- Computes **TPI at different scales (25, 35, 55, etc.)** using a moving window approach.
- Excludes NoData values to ensure robust calculations.
- Output: **TPI GeoTIFF raster**.

Usage:
```python
tpi_working.calculate_tpi_exclude_nodata(input_path, output_path, window_size=55)
```

---

### **2. `twi.py` - Topographic Wetness Index (TWI) Calculation**
- Uses **RichDEM** to compute **Slope and Flow Accumulation**.
- Computes **TWI** using the formula:
  ```python
  twi = log(flow_accum / tan(slope))
  ```
- Handles **division by zero** and **NoData values**.
- Output: **TWI GeoTIFF raster**.

Usage:
```python
twi.calculate_twi(input_dem, output_twi)
```

---

### **3. `aspect.py` - Aspect Calculation**
- Uses **RichDEM** to compute terrain **aspect** from DTM.
- Output: **Aspect GeoTIFF raster**.

Usage:
```python
aspect.calculate_aspect(input_dtm, output_aspect)
```

---

### **4. `slope.py` - Slope Calculation**
- Uses **RichDEM** to compute terrain **slope in degrees**.
- Output: **Slope GeoTIFF raster**.

Usage:
```python
slope.calculate_slope(input_dtm, output_slope)
```

---

### **5. `enrichment.py` - Spatial Data Enrichment (Zonal Statistics)**
- Extracts **DEM statistics (min, max, std)** per plot.
- Computes **Roughness metrics** and **TPI mean/max/std** at different scales.
- Integrates mound density and vegetation cover per plot.
- Merges all computed metrics into a **GeoPackage (GPKG)** for further analysis.

Usage:
```python
enrichment.process_metrics(gpkg_input, output_gpkg)
```

---

## Workflow Breakdown
### **Step 1: Compute Terrain Metrics**
- **TPI** (`tpi_working.py`)
- **TWI** (`twi.py`)
- **Slope & Aspect** (`slope.py`, `aspect.py`)

### **Step 2: Enrich Plots with Metrics**
- Compute **zonal statistics** (`enrichment.py`)
- Integrate TPI, TWI, slope, aspect, and vegetation cover into plot data.

### **Step 3: Generate Outputs**
- Save **GeoTIFF rasters** for each computed metric.
- Export **GeoPackage (GPKG)** with enriched data.

## Example Outputs
- **TPI Raster Example:**
  ![TPI Example](https://example.com/tpi_example.png)
- **Slope Raster Example:**
  ![Slope Example](https://example.com/slope_example.png)

## Troubleshooting
### **Common Issues & Fixes**
❌ **Raster size mismatch** → Ensure input rasters have the same resolution and CRS.  
❌ **GDAL or rasterio errors** → Check your GDAL installation (`gdalinfo --version`).  
❌ **Memory issues with large datasets** → Process in smaller tiles or increase RAM allocation.  

## License
This project is released under the **MIT License**.

## Contributors
- **Felix Wiemer** (Primary Author)
- **Felix Wiemer** (Adaptation & GitHub Documentation)


