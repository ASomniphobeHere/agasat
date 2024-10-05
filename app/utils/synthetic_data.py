import numpy as np
import cv2
import matplotlib.pyplot as plt

class SyntheticDataGenerator:
    def __init__(self):
        pass
    @staticmethod
    def generate_light_pollution_image(shape=(500, 500), num_sources=10):
        """
        Generate a synthetic light pollution image with bright regions simulating light pollution.

        Parameters:
        - shape: The dimensions of the image (height, width).
        - num_sources: Number of bright sources to simulate.

        Returns:
        - light_pollution: A synthetic grayscale image with bright regions.
        """
        # Create a black canvas for the image
        light_pollution = np.zeros(shape, dtype=np.float32)

        # Randomly place bright sources in the image
        for _ in range(num_sources):
            # Random center for the bright source
            center_x, center_y = np.random.randint(0, shape[1]), np.random.randint(0, shape[0])
            # Random intensity and radius for each source
            intensity = np.random.uniform(0.6, 1.0)  # High intensity for bright spots
            radius = np.random.randint(30, 70)       # Radius of the light pollution spot

            # Draw a filled circle to simulate a bright region
            cv2.circle(light_pollution, (center_x, center_y), radius, intensity, -1)

        # Apply Gaussian blur to spread out the intensity of the light sources
        light_pollution = cv2.GaussianBlur(light_pollution, (61, 61), 0)

        # Normalize to the range [0, 255] for proper visualization
        light_pollution = (light_pollution * 255).astype(np.uint8)

        return light_pollution

    @staticmethod
    def generate_co2_emission_image(shape=(500, 500), num_sources=15):
        """
        Generate a synthetic CO2 emissions image with diffused, cloudy regions.

        Parameters:
        - shape: The dimensions of the image (height, width).
        - num_sources: Number of CO2 emission sources to simulate.

        Returns:
        - co2_emissions: A synthetic grayscale image with diffused emission regions.
        """
        # Create a black canvas for the image
        co2_emissions = np.zeros(shape, dtype=np.float32)

        # Randomly place diffused sources in the image
        for _ in range(num_sources):
            # Random center for the emission source
            center_x, center_y = np.random.randint(0, shape[1]), np.random.randint(0, shape[0])
            # Random intensity and spread for each source
            intensity = np.random.uniform(0.8, 1)  # Lower intensity for emissions
            radius = np.random.randint(50, 100)      # Larger radius for more spread

            # Draw a filled circle to simulate a CO2 emission region
            cv2.circle(co2_emissions, (center_x, center_y), radius, intensity, -1)

        # Apply Gaussian blur to smooth out the emission regions
        co2_emissions = cv2.GaussianBlur(co2_emissions, (101, 101), 0)

        # Normalize to the range [0, 255] for proper visualization
        co2_emissions = (co2_emissions * 255).astype(np.uint8)

        return co2_emissions