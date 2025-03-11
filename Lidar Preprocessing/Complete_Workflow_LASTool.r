# I'm going to be implementing Jasper and Marlis' workflow but in R instead of cmd prompt. 

# this script is preprocessing_loop_areas_part2.bat COMBINED WITH
# the R script lidr_thin.R

# ---------------------------------------------------- #

# Batch script for preprocessing LiDAR data to prepare the creation of DEM, DSM and CHM
# 0. Metadata
# - lasvalidate	Verify validity of dataset
# - lasindex		Create a spacial index
# - lasinfo 		Metadata
# - lasgrid		Generate point density raster
# 1. Flightline cleansing
# - lasclip		Clip flightlines to flightboundaries so flightlines can be seperated
# - lassplit		Divide file into flightlines based on GPS-time gaps
# - lasoverlap		Generate raster respectively with amount of flightlines and height difference
# - lasoverage		Merge flightlines together and delete the overlap
# 2. Data cleansing
# - lastile		Split into tiles
# - lasduplicate	Delete duplicates
# - lasnoise 		Classify noise
# Author: Jasper Koch, Marlis Hegels 
# Munich, February 2023


# trying to call LASTools from R. 
# from this link: 
# https://groups.google.com/g/lastools/c/Eje6fiFprzc


# Define the path for lastools executables
# need to include the "\\" at the end of the below path for this to work. 
# LAStoolsDir = "C:\\Program Files (x86)\\LAStools\\bin\\"
LAStoolsDir = "D:\\LAStools\\bin\\"
# Define the R function that calls lastools executables
LAStool = function(tool, inputFile, ...){
  cmd = paste(paste(LAStoolsDir, tool, sep=''), '-i', inputFile , ...)
  cat(cmd)
  system(cmd)
  return(cmd)
}

# Define the directory for the ALS project
cloudPath  = "E:\\BERA\\BERA_Aufenthalt\\LiDEA_Pilot\\0000_Results\\LideaPilot_Lidar_August_opticloud\\lidars\\terra_las\LideaPilot-L1-4302_SEP.las""E:\\BERA\\BERA_Aufenthalt\\LiDEA_Pilot\\0000_Results\\LideaPilot_Lidar_August_opticloud\\lidars\\terra_las\\LideaPilot-L1-4302-SEP.las"
#cloudPath  = "E:\\BERA\\BERA_Aufenthalt\\LiDEA_Pilot\\0000_Results\\LideaPilot_Lidar_June\\June_Roadwest\\lidars\\terra_las\\LideaPilot-L1-ROADWEST_June.las"
# Define las/laz files to be processed
inFiles  = list.files(cloudPath, '.las', full.names = T)
inFiles

# inDir needs to be one .las file. 
inDir = cloudPath

# set path to flight areas folder
flightarea = "E:\\BERA\\BERA_Aufenthalt\\LiDEA_Pilot\\Footprints_SHP_august\\LideaPilot-L1-4302.shp
#flightarea = "E:\\BERA\\BERA_Aufenthalt\\LiDEA_Pilot\\Footprints_SHP_august\\LideaPilot-L1-4392.shp"
#flightarea = "C:\\Users\\Felix\\Documents\\LideaPilot-L1-ROADEAST-3.shp"

# Define output directory (many subfolders will be created within it)
outDir =  "E:\\BERA\\BERA_Aufenthalt\\LiDEA_Pilot\\0000_Results\\Test"
#outDir = "D:\\Test\\LideaPilot-ROADEAST-1"
#outDir = "E:\\BERA\\BERA_Aufenthalt\\LiDEA_Pilot\\0000_Results\\XY_Results\\Jun\\ROADWEST-JUNE"
#outDir2 = "D:\\Test\\LideaPilot-ROADEAST-1"
#inDir3 =  "D:\\Test\\LideaPilot-ROADEAST-1\\14_TilesNoOverlap"

outDir2 = "E:\\BERA\\BERA_Aufenthalt\\LiDEA_Pilot\\0000_Results\\Test"
inDir3 =  "E:\\BERA\\BERA_Aufenthalt\\LiDEA_Pilot\\0000_Results\\Test\\14_TilesNoOverlap"
#outDir2 ="E:\\BERA\\BERA_Aufenthalt\\LiDEA_Pilot\\0000_Results\\XY_Results\\Aug\\LideaPilot-L1-4352_SEP"
#inDir3 = "E:\\BERA\\BERA_Aufenthalt\\LiDEA_Pilot\\0000_Results\\XY_Results\\Aug\\LideaPilot-L1-4352_SEP\\14_TilesNoOverlap"



# load libraries for thinning LiDAR tiles to a certain point 
# density based on pulses
library(lidR)
library(future)
library(comprehenr)
substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}

# ---------------------------------------- # 

# parameters 
cores = 5

# utm zone
# NOTE: this script assumes both the .las and flight area .shp
# are projected in a NAD83 UTM zone, 
# and also that they are BOTH projected in the same one. 
utm = 12

# lasgrid
STEP_PD_GRID = 1 # lasgrid: grid size for point density raster in meter

# lassplit
TIME_GAP = 0.01 # lassplit: GPS-time gap to seperate flightlines

# lastile
TILE_SIZE = 60 # lastile: tile_size
TILE_BUFFER = 5 # lastile: buffer

# lasnoise
STEP_XY = 4 # lasnoise: step_xy
STEP_Z = 1 # lasnoise: step_z
POINTS = 5 # lasnoise: isolated
SA = 32 # lasnoise: maximum scan angle



# end of inputs

# ------------------------------------------- # 

# create subfolders

dir.create(file.path(outDir, '00_Info'), showWarnings = F)
dir.create(file.path(outDir, '01_Clip'), showWarnings = F)
dir.create(file.path(outDir, '02_Flightlines'), showWarnings = F)
dir.create(file.path(outDir, '03_CleanedFlightlines'), showWarnings = F)
dir.create(file.path(outDir, '04_Tiles'), showWarnings = F)
dir.create(file.path(outDir, '05_NoDups'), showWarnings = F)
dir.create(file.path(outDir, '06_Noise'), showWarnings = F)
dir.create(file.path(outDir, '07_ThinnedPD'), showWarnings = F)

# ------------------------------------------ #

# Start computing

# 0. Metadata
# 0.1 Verify validity
LAStool('lasvalidate', inDir,
        '-o', file.path(outDir, '00_Info','validate.xml'))

# 0.2 Spatial index of input las-file
LAStool('lasindex', inDir)

# 0.3 Metadata txt-file
LAStool('lasinfo', inDir,
        '-cores', cores,
        '-cd',
        '-histo scan_angle 1',
        '-odir', file.path(outDir,'00_Info'),
        '-odix _INFO',
        '-otxt')

# 0.4 Generate point density raster
LAStool('lasgrid', inDir,
        '-last_only',
        '-density',
        '-step', STEP_PD_GRID,
        paste0('-use_bb -nad83 -utm ', utm, 'U'),
        '-odir',file.path(outDir,'00_Info'),
        paste0('-odix _PointDensity',STEP_PD_GRID,'m'),
        '-otif')

# 1. Flightline cleansing
# 1.1 Clip LAS file with flight line shape file so flightiness can be separated

LAStool('lasclip', inDir,
        '-poly', flightarea,
        '-odir',file.path(outDir,'01_Clip'),
        '-olaz')

# 1.2 Divide the flightlines based on GPS time
LAStool('lassplit', list.files(file.path(outDir, '01_Clip'), '.laz', full.names = T),
        '-recover_flightlines_interval', TIME_GAP,
        '-odir', file.path(outDir,'02_Flightlines'),
        '-olaz')

# 1.3 Generate raster with amount of flightlines and height difference
LAStool('lasoverlap', paste0(file.path(outDir, '02_Flightlines'), '/*.laz'),
        '-merged -faf', 
        '-step', STEP_PD_GRID,
        '-values -elevation -lowest',
        paste0('-nad83 -utm ', utm, 'U'),
        '-odir', file.path(outDir,'03_CleanedFlightlines'),
        '-otif')

# 1.4 Delete overlapping points
LAStool('lasoverage', paste0(file.path(outDir, '02_Flightlines'), '/*.laz'),
        '-faf -remove_overage -merged',
        '-odir',file.path(outDir,'03_CleanedFlightlines'),
        '-olaz')

# 1.5 Generate point density raster
LAStool('lasgrid', paste0(file.path(outDir, '03_CleanedFlightlines'), '/*.laz'),
        '-last_only',
        '-density',
        '-step', STEP_PD_GRID,
        paste0('-use_bb -nad83 -utm ', utm, 'U'),
        '-odir',file.path(outDir,'03_CleanedFlightlines'),
        paste0('-odix CleanedPointDensity',STEP_PD_GRID,'m'),
        '-otif')


# 2. Data cleansing
# 2.1 Make data manageable by creating files that are easier to compute
LAStool('lastile', paste0(file.path(outDir, '03_CleanedFlightlines'), '/*.laz'),
        '-tile_size',TILE_SIZE,
        '-buffer', TILE_BUFFER,
        '-flag_as_synthetic', # should they be flagged as withheld or synthetic?
        '-odir',file.path(outDir,'04_Tiles'),
        '-olaz')

# 2.2 Delete duplicates
LAStool('lasduplicate', paste0(file.path(outDir, '04_Tiles'), '/*.laz'),
        '-unique_xyz',
        '-cores',cores,
        '-odir',file.path(outDir,'05_NoDups'),
        '-olaz')

# 2.3 Classify noise
LAStool('lasnoise', paste0(file.path(outDir, '05_NoDups'), '/*.laz'),
        '-step_xy', STEP_XY,
        '-step_z', STEP_Z,
        '-isolated', POINTS,
        '-keep_scan_angle',paste0('-',SA), SA,
        '-cores',cores,
        '-odir',file.path(outDir,'06_Noise'),
        '-olaz')

# original script removes intermediates, but I am going
# to keep them for now. 

# ------------------------------------- #

# this next part thins the point cloud based on pulses.
# rationale: used to remove artificially high density 
# sections that are caused by the matrice 300 rtk calibration dance

pd=85 #achieved point density
res_thinning = 1 # pixel size used to filter the points

tiles = list.files(file.path(outDir, '06_Noise'), '/*.laz',
                   full.names = TRUE)

# create a folder to hold density plots
dir.create(file.path(outDir, '07_ThinnedPD','density_plots'))

tile = tiles[2]
for(tile in tiles) {
  las = readLAS(tile)
  las = retrieve_pulses(las)
  thinned = decimate_points(las, homogenize(pd, res_thinning, use_pulse = T))
  density = rasterize_density(thinned, res=1)
  
  # export density plot
  png(filename = file.path(outDir,'07_ThinnedPD','density_plots', paste0(tools::file_path_sans_ext(basename(tile)), '.png')),
      width = 900, height = 600, res = 100)
  plot(density)
  title(basename(tile))
  dev.off()
  
  #pod = density[[1]] # point_density
  #pud = density[[2]] # pulse density
  #summary(pod)
  #summary(pud)
  
  writeLAS(thinned, file.path(outDir,'07_ThinnedPD', basename(tile)))
}



# I'm going to be implementing Jasper and Marlis' workflow but in R instead of cmd prompt. 

# this script is preprocessing_loop_areas_part2.bat

# ---------------------------------------------------- #

# Batch script for preprocessing LiDAR data to prepare the creation of DEM, DSM and CHM
# 3. Classifying
# - lasground 		Classify ground
# - lasthin 		Thin ground points
# - lasclassify 	Reclassify thinned points to ground points
# - lasheight 		Calculate point height over the ground
# - lasclassify 	Classify points >1m over the ground as high vegetation
# 4. Merging
# - lasmerge 		Merge files
# 5. Metadata preprocessed output
# - lasinfo		Metadata
# - lasindex		Generate spatial index
# Author: Jasper Koch, Marlis Hegels 
# Munich, February 2023

# trying to call LASTools from R. 
# from this link: 
# https://groups.google.com/g/lastools/c/Eje6fiFprzc

# Define the path for lastools executables
# need to include the "\\" at the end of the below path for this to work. 
LAStoolsDir = "D:\\LAStools\\bin\\"

# Define the R function that calls lastools executables
LAStool = function(tool, inputFile, ...){
  cmd = paste(paste(LAStoolsDir, tool, sep=''), '-i', inputFile , ...)
  cat(cmd)
  system(cmd)
  return(cmd)
}


# set directory for where the outputs from previous script are stored. 
# this is also where new output subfolders will be created. 

# parameters
CORES=5

# lasthin
PD=85 # lasthin: max. point density
STEP_T=0.15 # lasthin: step

# lasground
STEP_G=3 # lasground: step
OFFSET_G=0.1 # lasground: offset

# lasclassify
OFFSET_C=1.0 # lasclassify: ground_offset

# lastile
TILE_SIZE = 60 # lastile: tile_size
TILE_BUFFER = 0 # lastile: buffer

# lasgrid
STEP_PD_GRID = 1 # lasgrid: grid size for point density raster in meter

# utm zone
# NOTE: this script assumes both the .las and flight area .shp
# are projected in a NAD83 UTM zone, 
# and also that they are BOTH projected in the same one. 
utm = 12

# end of inputs 

# ------------------------------------ #

# create subfolders

dir.create(file.path(outDir2, '08_Ground'), showWarnings = F)
dir.create(file.path(outDir2, '09_GroundThinned'), showWarnings = F)
dir.create(file.path(outDir2, '10_Classified'), showWarnings = F)
dir.create(file.path(outDir2, '11_Height'), showWarnings = F)
dir.create(file.path(outDir2, '12_Reclassified'), showWarnings = F)
dir.create(file.path(outDir2, '13_Merged'), showWarnings = F)
dir.create(file.path(outDir, '14_TilesNoOverlap'), showWarnings = F)

# ------------------------------------------ #

# Start computing

# 3. Classifying
# 3.1 Classify ground
LAStool('lasground', paste0(file.path(outDir2, '07_ThinnedPD'), '/*.laz'),
        '-compute_height -ignore_class 7',
        '-step', STEP_G,
        '-offset', OFFSET_G,
        '-cores', CORES,
        '-odir', file.path(outDir2,'08_Ground'),
        '-olaz')

### TESTING NEW VALUES BELOW
# LAStool('lasground', paste0(file.path(outDir2, '07_ThinnedPD'), '/*.laz'),
#         '-compute_height -ignore_class 7',
#         '-not_airborne',
#         '-cores', CORES,
#         '-odir', file.path(outDir,'08_Ground'),
#         '-olaz')

# 3.2 Thin ground
LAStool('lasthin', paste0(file.path(outDir2, '08_Ground'), '/*.laz'),
        '-ignore_class 1 -classify_as 14',
        '-lowest -step',STEP_T,
        '-cores',CORES,
        '-odir',file.path(outDir2,'09_GroundThinned'),
        '-olaz')

# 3.3 Reclassify thinned points to ground points
LAStool('lasclassify', paste0(file.path(outDir2, '09_GroundThinned'), '/*.laz'),
        '-change_classification_from_to 2 13',
        '-change_classification_from_to 14 2',
        '-cores', CORES,
        '-odir',file.path(outDir2,'10_Classified'),
        '-olaz')

# 3.4 Recalculate point height above ground
LAStool('lasheight', paste0(file.path(outDir2, '10_Classified'), '/*.laz'),
        '-cores',CORES,
        '-odir',file.path(outDir2,'11_Height'),
        '-olaz')

# 3.5 Classify points with 1 metre  above the ground as high vegetation
LAStool('lasclassify', paste0(file.path(outDir2, '11_Height'), '/*.laz'),
        '-ground_offset',OFFSET_C, 
        '-small_trees',
        '-cores',CORES,
        '-odir',file.path(outDir2,'12_Reclassified'),
        '-olaz')

# 4. Merging the file

# first grab the filename 
merge_name <- gsub('.{4}$', '', list.files(file.path(outDir2, '01_Clip'), '.laz'))

LAStool('lasmerge', paste0(file.path(outDir2, '12_Reclassified'), '/*.laz'),
        '-drop_synthetic',
        '-o',file.path(outDir2,'13_Merged',paste0(merge_name,'_PD',PD,'.las')))

# 4.1 Spatial index of output las-file
LAStool('lasindex', paste0(file.path(outDir2, '13_Merged'), '/*.las'))

# 5. Metadata preprocessed output
# 5.1 Compute Metadata
LAStool('lasinfo', paste0(file.path(outDir2, '13_Merged'), '/*.las'),
        '-cd -histo scan_angle 1',
        '-odir',file.path(outDir2,'13_Merged'),
        '-odix _INFO -otxt')

# 6. Generate point density raster
LAStool('lasgrid', paste0(file.path(outDir2, '13_Merged'), '/*.las'),
        '-last_only',
        '-density',
        '-step', STEP_PD_GRID,
        paste0('-use_bb -nad83 -utm ', utm, 'U'),
        '-odir',file.path(outDir2,'13_Merged'),
        paste0('-odix CleanedPointDensity',STEP_PD_GRID,'m'),
        '-otif')

# 7. Create tiles with no buffer, for use with lascatalogue in lidR
LAStool('lastile', paste0(file.path(outDir2, '13_Merged'), '/*.las'),
        '-tile_size',TILE_SIZE,
        '-buffer', TILE_BUFFER, # buffer here is 0
        '-odir',file.path(outDir2,'14_TilesNoOverlap'),
        '-olaz')


# 8. Create spatial index for tiles. 
LAStool('lasindex', paste0(file.path(outDir2,'14_TilesNoOverlap'), '/*.laz'))


# Jasper and Marlis' script deletes the intermediates, but I'm 
# going to keep them all for now. 

#install.packages("lidR")
library("lidR")
library("geometry")
library("gstat")
library("terra")


# ---------------------------------------------------------------------- #

# folder where input tiles are stored
#create a lascatalog to work with a collection of las/laz file
ctg <- readLAScatalog(inDir3)
print(ctg)

las_check(ctg)

#temp folder
dir.create(file.path(dirname(inDir3), '15_dtmtemp'), showWarnings = F)
opt_output_files(ctg) <-paste0(file.path(dirname(inDir3), '15_dtmtemp'), "/{*}_dtm")

#You would have to normalize the las catalog to get meaningful CHM
#First, generate DTM in 15cm resolution. DTM will be saved in the temp folder
dtm <- rasterize_terrain(ctg, 0.15, knnidw())
plot(dtm, col = gray(1:50/50))


#try hybrid method for chm
#see documentation page: https://r-lidar.github.io/lidRbook/norm.html#norm-dtm
ctg2 <- normalize_height(ctg, tin(), dtm = dtm)
#To see historgram, uncomment the line below
#hist(filter_ground(nlas)$Z, breaks = seq(-0.6, 0.6, 0.01), main = "", xlab = "Elevation")

#output location for CHM
dir.create(file.path(dirname(inDir3), '16_chmtemp'), showWarnings = F)

opt_output_files(ctg2) <- paste0(file.path(dirname(inDir3), '16_chmtemp'), "/CHM_{XRIGHT}_{YBOTTOM}")

# rasterize the chm
chm <- rasterize_canopy(ctg2, res = 0.15, p2r(0.15, na.fill =  knnidw()))
chm

# for the chm, force all negative values to be 0. 
m <- c(-50, 0, 0)

rclmat <- matrix(m, ncol = 3, byrow = TRUE) 
chm <- classify(chm, rclmat)

# now resample so that chm matches the dtm. 
chm <- resample(chm, dtm, method = 'bilinear') 

# create dsm by adding the dtm and chm. 
dsm <- dtm + chm

plot(dsm, col = gray(1:50/50))

# output 
# create folder to output things 
dir.create(file.path(dirname(inDir3), '17_Raster'), showWarnings = F)

writeRaster(dtm, 
            file.path(dirname(inDir3), '17_Raster','dtm.tif'),
            overwrite= TRUE)

writeRaster(chm, 
            file.path(dirname(inDir3), '17_Raster','chm.tif'),
            overwrite= TRUE)

writeRaster(dsm, 
            file.path(dirname(inDir3), '17_Raster','dsm.tif'),
            overwrite= TRUE)


#To be improved script, use at your own risk

#seeking solution of filter outliers
#possible solution for outlier removal: https://gis.stackexchange.com/questions/371774/way-to-filter-outliers-from-point-cloud-in-lidr
#possible solution for outlier removal: https://cran.r-project.org/web/packages/lidR/vignettes/lidR-catalog-apply-examples.html
