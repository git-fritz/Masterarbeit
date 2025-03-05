import numpy as np
import rasterio

def convert_nan_to_nodata(input_path, output_path):
    """
    Convert all NaN values in a GeoTIFF to NoData.

    Parameters:
        input_path (str): Path to the input GeoTIFF file.
        output_path (str): Path to save the modified GeoTIFF file.
    """
    with rasterio.open(input_path) as src:
        # Read the data and metadata
        data = src.read(1)  # Read the first band
        meta = src.meta.copy()

        # Replace NaN values with a NoData value
        nodata_value = -9999  # Define your desired NoData value
        data = np.where(np.isnan(data), nodata_value, data)

        # Update metadata to reflect the NoData value
        meta.update({"nodata": nodata_value})

        # Save the modified data to a new GeoTIFF
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(data, 1)

if __name__ == "__main__":
    # Example usage
    input_file = "E:\Thesis\merge\merged_east.tif"
    output_file = "E:\Thesis\merge\merged_east9999.tif"

    convert_nan_to_nodata(input_file, output_file)
    print(f"Converted NaN values to NoData and saved to {output_file}")

#%%

import numpy as np
import rasterio
import matplotlib.pyplot as plt

def visualize_nan_locations(dem_path):
    """
    Visualize where NaN values are located in a DEM.

    Parameters:
        dem_path (str): Path to the input DEM GeoTIFF file.
    """
    with rasterio.open(dem_path) as src:
        data = src.read(1)
        
        # Create a binary mask for NaN values
        nan_mask = np.isnan(data)
        
        # Plot the mask with reversed colors
        plt.figure(figsize=(10, 8))
        plt.imshow(nan_mask, cmap='gray_r', interpolation='none')
        plt.colorbar(label='NaN Mask (1=NaN, 0=Valid)')
        plt.title("NaN Locations in DEM (Reversed Colors)")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.show()

# Example usage
dem_file = r"E:\Thesis\testing\output_metrics\roadwest2\nodatatest9999.tif"
visualize_nan_locations(dem_file)

import numpy as np
import rasterio

def count_nan_values(input_path):
    """
    Count the number of NaN values in a DTM GeoTIFF.

    Parameters:
        input_path (str): Path to the input DTM GeoTIFF file.

    Returns:
        int: The number of NaN values in the data.
    """
    with rasterio.open(input_path) as src:
        data = src.read(1)  # Read the first band
        nan_count = np.isnan(data).sum()
    return nan_count

if __name__ == "__main__":
    input_file = r"E:\Thesis\testing\output_metrics\roadwest2\nodatatest9999.tif"

    nan_count = count_nan_values(input_file)
    print(f"Number of NaN values in the file: {nan_count}")