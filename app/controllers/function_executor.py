import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.controllers.image_processor import ImageProcessor
import cv2
import numpy as np
from loguru import logger

class FunctionExecutor:
    def __init__(self, datasets):
        """
        Initialize the FunctionExecutor with the available datasets.
        
        Parameters:
        - datasets: A dictionary containing dataset names and their corresponding images.
        """
        self.datasets = datasets
        self.processor = ImageProcessor  # Reference to your ImageProcessor class

    def execute_functions(self, output_json):
        """
        Execute the function order from the given output JSON and return the final result image.

        Parameters:
        - output_json: A JSON object containing datasets, function calls, and the final result variable.

        Returns:
        - final_image: The resulting image after executing all functions.
        """
        images = {}

        print("Output JSON")
        print(output_json)

        # Load initial datasets into the images dictionary
        for dataset in output_json['datasets']:
            images[dataset] = ImageProcessor.normalize_power(self.datasets[dataset], power=0.2)
            images[dataset] = cv2.cvtColor(images[dataset], cv2.COLOR_BGR2GRAY)
            # ImageProcessor.show_image(images[dataset], "Initial Dataset Image Normalized")

        # Execute each function in the specified order and store the result using 'output_image'
        for function in output_json['functions']:
            func_name = function['func']
            params = function['params']
            output_image_name = function['output_image']
            logger.info(f"Function: {func_name}, {params}")

            # Map function names to actual methods in the ImageProcessor class
            if func_name == "multiply_masks":
                images[output_image_name] = self.processor.multiply_masks(images[params[0]], images[params[1]])
            elif func_name == "invert_bw_image":
                images[output_image_name] = self.processor.invert_bw_image(images[params[0]])
            elif func_name == "weighted_average":
                image_list = [images[param] for param in params[0]]
                weight_list = params[1]
                images[output_image_name] = self.processor.weighted_average(image_list, weight_list)
            elif func_name == "edge_detection":
                images[output_image_name] = self.processor.edge_detection(images[params[0]], min_val=params[1], max_val=params[2])
            elif func_name == "gaussian_blur":
                images[output_image_name] = self.processor.gaussian_blur(images[params[0]], kernel_size=params[1], sigma=params[2])

            # ImageProcessor.show_image(images[output_image_name], "Intermediate Result")
            logger.info(f"Image dimensions: {images[output_image_name].shape}")

        # Return the final image specified in the JSON output under 'final_result'
        final_image = images[output_json['final_result']]
        return final_image

def generate_base_image(shape=(500, 500), noise_level=50):
    """
    Generate a base image with noise to simulate general urban density.

    Parameters:
    - shape: The shape of the image (height, width).
    - noise_level: The maximum intensity for the noise.

    Returns:
    - base_image: A synthetic base image representing urban density.
    """
    # Create a base image with noise
    base_image = np.random.randint(0, noise_level, shape, dtype=np.uint8)
    
    # Add larger bright spots to simulate dense urban areas
    num_dense_areas = 10
    for _ in range(num_dense_areas):
        x = np.random.randint(50, shape[1] - 50)
        y = np.random.randint(50, shape[0] - 50)
        intensity = np.random.randint(100, 256)
        cv2.circle(base_image, (x, y), np.random.randint(30, 60), intensity, -1)

    return base_image

def generate_light_pollution_image(base_image):
    """
    Generate a synthetic light pollution image from the base image.

    Parameters:
    - base_image: The base image used to derive light pollution data.

    Returns:
    - light_image: A synthetic light pollution image.
    """
    # Create a light pollution image by increasing intensity in certain areas
    light_image = base_image.copy()

    # Increase intensity in random locations to simulate light pollution
    for _ in range(30):  # Simulate 30 light sources
        x = np.random.randint(0, light_image.shape[1])
        y = np.random.randint(0, light_image.shape[0])
        brightness = np.random.randint(150, 256)
        cv2.circle(light_image, (x, y), np.random.randint(5, 15), brightness, -1)

    return light_image

def generate_co2_emission_image(base_image):
    """
    Generate a synthetic CO2 emissions image from the base image.

    Parameters:
    - base_image: The base image used to derive CO2 emissions data.

    Returns:
    - co2_image: A synthetic CO2 emissions image.
    """
    # Create a CO2 emissions image by modifying the base image
    co2_image = base_image.copy()

    # Simulate varying CO2 intensity based on the base image's intensity
    for i in range(co2_image.shape[0]):
        for j in range(co2_image.shape[1]):
            if co2_image[i, j] > 0:  # Only modify where there is light
                # Decrease the intensity to simulate emissions (lower values = more emissions)
                co2_image[i, j] = max(0, co2_image[i, j] - np.random.randint(0, 50))
    
    return co2_image

# Example usage
if __name__ == "__main__":
    # Load datasets as numpy arrays (for example purposes, replace with actual loading code)
    base_image = generate_base_image(shape=(500, 500), noise_level=100)

    light_pollution_image = generate_light_pollution_image(base_image)
    co2_emission_image = generate_co2_emission_image(base_image)

    datasets = {
        "light": light_pollution_image,  # Replace with actual light data
        "co2": co2_emission_image    # Replace with actual CO2 data
        # Add other datasets as needed
    }

    
    # Example output JSON for the question
    output_json = {
        "datasets": ["light", "co2"],
        "functions": [
            {
                "func": "multiply_masks",
                "params": ["light", "co2"],
                "output_image": "combined_light_co2"
            }
        ],
        "final_result": "combined_light_co2"
    }
    
    executor = FunctionExecutor(datasets)
    final_image = executor.execute_functions(output_json)
    # executor.processor.show_image(base_image)
    # executor.processor.show_image(light_pollution_image)
    # executor.processor.show_image(co2_emission_image)
    
    # Display the resulting image
    cv2.imshow("Final Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
