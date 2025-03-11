# Complete LiDAR Processing Workflow (LASTools + lidR)

## Overview
This repository contains an **R-based LiDAR processing workflow** that automates the preprocessing and classification of airborne LiDAR data using **LASTools** and **lidR**. It follows the methodology developed by **Jasper Koch and Marlis Hegels** to generate high-quality **Digital Terrain Models (DTM)**, **Digital Surface Models (DSM)**, and **Canopy Height Models (CHM)**.

The script `Complete_Workflow_LASTools.r` is a **fully integrated pipeline**, combining multiple steps such as **flightline cleaning, noise removal, ground classification, and height normalization** into a structured processing sequence.

## Features
✅ **LiDAR Preprocessing:** Removes noise, duplicates, and cleans flightlines.  
✅ **Ground Classification:** Identifies and processes ground points.  
✅ **Height Normalization:** Computes heights above ground to create CHM.  
✅ **Raster Generation:** Produces **DTM, DSM, and CHM** at high resolution.  
✅ **LASTools + lidR Integration:** Combines command-line tools with R-based processing.  

## Requirements
### **1. Software Dependencies**
Ensure the following software is installed:
- [LASTools](https://rapidlasso.com/lastools/) (must be added to system path)
- [R](https://www.r-project.org/)
- [RStudio](https://posit.co/downloads/)

### **2. Required R Packages**
The script uses several R packages, which must be installed before running:
```r
install.packages(c("lidR", "terra", "geometry", "gstat", "future", "comprehenr"))
```

## Usage Instructions
### **1. Setup Paths**
Modify the paths in the script to match your LiDAR dataset locations:
```r
LAStoolsDir = "D:\\LAStools\\bin\\"
cloudPath  = "E:\\YourData\\Lidar\\Input.las"
outDir = "E:\\YourData\\Lidar\\ProcessedData"
```

### **2. Run the Script**
Execute the script in RStudio:
```r
source("Complete_Workflow_LASTools.r")
```

## Workflow Breakdown
### **Step 1: Metadata Extraction**
- Validates and indexes LAS files.
- Computes metadata (e.g., scan angles, density).

### **Step 2: Flightline Cleansing**
- Clips flightlines.
- Removes overlap between flightlines.
- Generates point density rasters.

### **Step 3: Data Cleansing**
- Splits large files into tiles.
- Removes duplicate points.
- Identifies and classifies noise.

### **Step 4: Ground Classification & Height Processing**
- Detects ground points and removes non-ground features.
- Computes point height above ground.
- Classifies vegetation height categories.

### **Step 5: Raster Generation**
- Generates DTM (Digital Terrain Model) at **15 cm resolution**.
- Generates CHM (Canopy Height Model) using **bilinear resampling**.
- Combines DTM + CHM to create DSM (Digital Surface Model).

### **Step 6: Saving Outputs**
- Saves `dtm.tif`, `chm.tif`, and `dsm.tif` to the `ProcessedData` folder.
- Creates a spatial index for faster access.

## Example Outputs
- **DTM (Digital Terrain Model):**
  ![DTM Example](https://example.com/dtm_example.png)
- **CHM (Canopy Height Model):**
  ![CHM Example](https://example.com/chm_example.png)

## Troubleshooting
### **Common Issues**
❌ **LASTools command not found** → Ensure LASTools is installed and added to the system path.  
❌ **CRS mismatch** → Ensure `.las` and `.shp` files use the same projection (UTM).  
❌ **Memory issues** → Process large datasets in smaller tiles or increase available memory.  

## License
This project is released under the **MIT License**.

## Contributors
- **Jasper Koch** (Workflow Author)
- **Marlis Hegels** (Workflow Author)
- **Xue Yan Chan** (Workflow Author)
- **Felix Wiemer** (Adaptation & GitHub Documentation)
