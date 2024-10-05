import os
import zipfile
import xarray as xr
import numpy as np
from PIL import Image

# Step 1: Unzip the NetCDF file
def unzip_netcdf(zip_filepath, extract_to='.'):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    # Assuming the NetCDF file is now extracted
    extracted_files = [f for f in os.listdir(extract_to) if f.endswith('.nc')]
    return extracted_files

# Step 2: Load the NetCDF file
def load_netcdf(filepath):
    ds = xr.open_dataset(filepath)
    return ds

# Step 3: Generate heatmap image from CO2 data
def generate_co2_heatmap(ds, output_png, image_size=(10000, 10000)):
    # Extract CO2 data
    co2_data = ds['co2'].values  # Adjust this based on the actual variable name
    # Assuming co2_data is 2D: [latitude, longitude]
    
    # Normalize data for image
    co2_min, co2_max = np.min(co2_data), np.max(co2_data)
    co2_normalized = (co2_data - co2_min) / (co2_max - co2_min)  # Normalize to [0, 1]

    # Scale to [0, 255] for image
    co2_scaled = (co2_normalized * 255).astype(np.uint8)

    # Create a PIL Image from the scaled data
    heatmap_image = Image.fromarray(co2_scaled, mode='L')
    
    # Resize the image to the desired size
    heatmap_image = heatmap_image.resize(image_size, Image.LANCZOS)
    
    # Save the heatmap image
    heatmap_image.save(output_png)
    print(f"CO2 heatmap saved as {output_png}")

# Example usage
zip_filepath = 'co2_data.zip'
output_dir = './extracted_files'
output_png = 'co2_heatmap_10kx10k.png'

# Step 1: Unzip the file
extracted_files = unzip_netcdf(zip_filepath, extract_to=output_dir)

# Step 2: Load the first extracted NetCDF file
netcdf_data = load_netcdf(os.path.join(output_dir, extracted_files[0]))

# Step 3: Generate CO2 heatmap
generate_co2_heatmap(netcdf_data, output_png)
