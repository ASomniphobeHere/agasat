import cfgrib
import numpy as np
from PIL import Image

# Load GRIB file
grib_file = 'co2_data.grib'
ds = cfgrib.open_datasets(grib_file)[0]  # Open the GRIB file

# Assuming the data you want is in a specific variable, e.g., 't'
# Adjust to the name of the variable you're interested in
data_array = ds['co2'].values

# Normalize data to fit within the range [0, 255]
data_min = np.min(data_array)
data_max = np.max(data_array)
normalized_data = 255 * (data_array - data_min) / (data_max - data_min)

# Convert to uint8 type
image_data = normalized_data.astype(np.uint8)

# Resize to 10,000 x 10,000
image_resized = Image.fromarray(image_data).resize((10000, 10000))

# Save the image as PNG
output_image = 'dataset_api/europe_co2.png'
image_resized.save(output_image)

print(f"Image saved as {output_image}")
