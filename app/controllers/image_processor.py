import cv2
import numpy as np
from imutils import contours
import base64
import imutils
import matplotlib.pyplot as plt
import os
import sys
from loguru import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.utils.synthetic_data import SyntheticDataGenerator

class ImageProcessor:

    @staticmethod
    def detect_bright_spots(image, blur_radius=(11, 11), min_area=100, max_spots=5, initial_brightness_factor=1.5, step_factor=0.1):
        """
        Detect and return the centers of the top bright spots in the given image, sorted by area.

        Parameters:
        - image: A pre-loaded OpenCV image object (numpy array).
        - blur_radius: Kernel size for the Gaussian blur.
        - min_area: Minimum area (in pixels) for a region to be considered a bright spot.
        - max_spots: Maximum number of bright spots to return, sorted by area.
        - initial_brightness_factor: Initial multiplier for setting the brightness threshold based on image statistics.
        - step_factor: Factor by which to decrease the threshold in each iteration if not enough bright spots are found.

        Returns:
        - centers: A list of tuples representing the (x, y) coordinates of the centers of the bright spots.
        """

        try:
            # Step 1: Validate input image
            if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                logger.error("Invalid image: Input image is either None, not a numpy array, or empty.")
                return []

            # Step 2: Ensure the image is in single-channel grayscale format
            if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is in BGR format
                logger.info("Converting BGR image to grayscale.")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) != 2:  # If it's not single-channel or not a standard grayscale image
                logger.error("Invalid image format: Expected a single-channel grayscale image.")
                return []
            

            # Step 3: Apply Gaussian blur to the grayscale image to smooth out noise
            blurred = cv2.GaussianBlur(image, blur_radius, 0)
            blurred = ImageProcessor.normalize(blurred)
            # ImageProcessor.show_image(blurred)

            # Step 4: Calculate image statistics (mean and standard deviation)
            mean, std_dev = cv2.meanStdDev(blurred)
            mean = mean[0][0]
            std_dev = std_dev[0][0]

            # Step 5: Set the initial brightness threshold (aggressive thresholding)
            min_brightness = 220  # More aggressive to highlight strong bright spots
            logger.info(f"Min brightness: {min_brightness}")

            # Step 6: Apply aggressive threshold to get an initial bright spot mask
            _, thresh = cv2.threshold(blurred, min_brightness, 255, cv2.THRESH_BINARY)
            # ImageProcessor.show_image(thresh, "Initial Threshold")

            # Step 7: Perform iterative erosion until we have at least `max_spots` distinct components
            erosion_history = []
            kernel_size = 11
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

            logger.info(f"num labels {num_labels}")
            best_thresh = thresh.copy()
            max_comps = num_labels - 1

            while num_labels > 1 and kernel_size < thresh.shape[0]-10:  # Continue until we have at least `max_spots` distinct components
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                eroded = cv2.erode(thresh, kernel, iterations=1)
                # ImageProcessor.show_image(eroded, f"Eroded with Kernel Size: {kernel_size}")
                
                # Count the number of distinct connected components
                num_labels, _, _, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
                logger.info(f"Number of components with kernel size {kernel_size}: {num_labels - 1}")  # Exclude the background
                
                if num_labels - 1 > max_comps:  # If we have at least `max_spots` bright spots
                    max_comps = num_labels - 1
                    best_thresh = eroded

                kernel_size += 8  # Increase kernel size aggressively

            # Step 10: Use OpenCV's connected components to label the thresholded image
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(best_thresh, connectivity=8, ltype=cv2.CV_32S)

            height, width = best_thresh.shape[:2]  # Get image dimensions
            total_area = height * width             # Calculate the total area of the image

            # Calculate the threshold for 1/4 of the total area
            quarter_area_threshold = total_area / 4
            # Step 11: Filter components based on area and collect component centers
            components = []
            for i in range(1, num_labels):  # Start from 1 to skip the background
                area = stats[i, cv2.CC_STAT_AREA]
                logger.info(f"Area : {area}")
                if area >= min_area and area < quarter_area_threshold:  # Filter by area
                    # Calculate the center of the bright spot
                    cX = int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2)
                    cY = int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2)

                    # Add the component's area and center coordinates to the list
                    components.append((area, (cX, cY)))
            logger.info(f"List len {len(components)}")
            # Step 12: Sort the components by area in descending order and select the top `max_spots`
            components = sorted(components, key=lambda x: x[0], reverse=True)[:max_spots]

        except Exception as e:
            logger.error(f"Error in bright spot detection: {e}")
            return []  # Return an empty list in case of an error

        # Step 13: Extract the center coordinates of the top components
        centers = [center for _, center in components]

        return centers
    
    def map_grayscale_to_custom_colormap(grayscale_image, custom_colors=None, transparency_range=(0, 255)):
        """
        Maps the grayscale image values to a custom color map with transparency.

        Args:
        grayscale_image (np.array): Input grayscale image.
        custom_colors (np.array): A list of colors defining the colormap.
        transparency_range (tuple): Min and max transparency values for green colors.

        Returns:
        np.array: Transformed image with the custom colormap and transparency applied.
        """
        if custom_colors is None:
            # Define a custom red-yellow-green color spectrum (colormap) with an RGBA format.
            custom_colors = np.array([
                [255, 0, 0, 0],        # Green (fully transparent)
                [255, 0, 128, 64],    # Light Green (semi-transparent)
                [255, 0, 255, 64],    # Yellow (opaque)
                [128, 0, 255, 64],    # Orange (opaque)
                [0, 0, 255, 64],      # Red (opaque)
            ], dtype=np.uint8)
        
        # Normalize grayscale image values to the range of [0, len(custom_colors)-1]
        norm_image = cv2.normalize(grayscale_image, None, 0, len(custom_colors) - 1, cv2.NORM_MINMAX).astype(np.uint8)

        # Create an RGBA image where each pixel has 4 channels: [B, G, R, A]
        colored_image = np.zeros((*grayscale_image.shape, 4), dtype=np.uint8)

        # Map the normalized values to the custom RGBA colormap
        for i in range(len(custom_colors)):
            colored_image[norm_image == i] = custom_colors[i]

        return colored_image
    
    @staticmethod
    def multiply_masks(image1, image2):
        """
        Multiply two MatLike images (numpy arrays) after normalizing to [0, 1].

        Parameters:
        - image1: The first image/mask (numpy array in MatLike format).
        - image2: The second image/mask (numpy array in MatLike format).

        Returns:
        - multiplied_mask: A new mask with pixel values between 0 and 1.
        """
        # Ensure that both images have the same dimensions
        if image1.shape != image2.shape:
            raise ValueError("The input images must have the same dimensions for multiplication.")

        # Convert images to float32 for scaling and normalization
        image1_normalized = image1.astype(np.float32) / 255.0
        image2_normalized = image2.astype(np.float32) / 255.0

        # Element-wise multiplication of the two normalized images
        multiplied_mask = np.sqrt(image1_normalized * image2_normalized)

        # Convert the result back to the range [0, 255] for visualization if needed
        multiplied_mask_visual = (multiplied_mask * 255).astype(np.uint8)

        return multiplied_mask_visual
    
    @staticmethod
    def invert_bw_image(image):
        """
        Invert a black-and-white image.

        Parameters:
        - image: The input image (numpy array).

        Returns:
        - inverted_image: The inverted image.
        """
        return cv2.bitwise_not(image)
    
    @staticmethod
    def weighted_average(image_list, weight_list):
        """
        Compute the weighted average of a list of images.
        
        Parameters:
        - image_list: A list of input images (numpy arrays).
        - weight_list: A list of weights for each image. The weights should sum to 1.
        
        Returns:
        - weighted_avg: The weighted average image.
        """
        
        # Initialize the weighted average image
        weighted_avg = np.zeros_like(image_list[0], dtype=np.float32)
        
        # Compute the weighted sum of images
        for img, weight in zip(image_list, weight_list):
            weighted_avg += img * weight
        
        return weighted_avg
    
    @staticmethod
    def pixel_threshold(image, threshold_value=128):
        """
        Apply a pixel-wise threshold to the input image.

        Parameters:
        - image: The input image (numpy array).
        - threshold_value: The threshold value for pixel intensity (default: 128).

        Returns:
        - thresholded_image: The thresholded image.
        """
        _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        return thresholded_image

    @staticmethod
    def edge_detection(image, min_val=100, max_val=200):
        """
        Apply Canny edge detection to the input image.

        Parameters:
        - image: The input image (numpy array).
        - min_val: Minimum intensity gradient value (default: 100).
        - max_val: Maximum intensity gradient value (default: 200).

        Returns:
        - edges: The image with detected edges.
        """
        edges = cv2.Canny(image, min_val, max_val)
        return edges
    
    @staticmethod
    def gaussian_blur(image, kernel_size=5, sigma=0):
        """
        Apply Gaussian blur to the input image.

        Parameters:
        - image: The input image (numpy array).
        - kernel_size: Size of the Gaussian kernel (default: 5x5).
        - sigma: Standard deviation of the Gaussian kernel (default: 0).

        Returns:
        - blurred_image: The image after applying Gaussian blur.
        """
        kernel = (kernel_size, kernel_size)
        blurred_image = cv2.GaussianBlur(image, kernel, sigma)
        return blurred_image
    
    @staticmethod
    def normalize(image):
        # Ensure the input is a numpy array
        image = np.array(image, dtype=np.float32)

        # Get minimum and maximum pixel values
        min_val = np.min(image)
        max_val = np.max(image)

        # Apply linear scaling
        scaled_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return scaled_image
    
    @staticmethod
    def normalize_power(image, power=0.5):
        """
        Normalize the input image using square root transformation.

        Parameters:
        - image: Input image as a numpy array (any dtype).

        Returns:
        - Transformed image with pixel values scaled to 0-255 using sqrt transformation.
        """

        # Ensure the input is a numpy array and convert to float32 for calculations
        image = np.array(image, dtype=np.float32)

        # Get minimum and maximum pixel values for normalization
        min_val = np.min(image)
        max_val = np.max(image)

        # Normalize image to the range [0, 1]
        normalized_image = (image - min_val) / (max_val - min_val + 1e-5)  # Adding a small value to prevent division by zero

        # Apply square root transformation
        sqrt_image = np.power(normalized_image, power)

        # Scale back to [0, 255]
        scaled_sqrt_image = (sqrt_image * 255).astype(np.uint8)

        return scaled_sqrt_image
    
    @staticmethod
    def visualize_centers(image, centers, marker_size=10, color=(0, 0, 255), thickness=2):
        """
        Visualize the centers of detected bright spots on the image.

        Parameters:
        - image: The original input image (numpy array in BGR format).
        - centers: A list of tuples representing the (x, y) coordinates of the centers.
        - marker_size: Size of the cross marker to draw (default: 10).
        - color: Color of the marker in BGR format (default: red).
        - thickness: Thickness of the marker lines (default: 2).

        Returns:
        - image_with_centers: A new image with the centers marked.
        """
        # Create a copy of the image to avoid modifying the original
        image_with_centers = image.copy()

        # Loop over each center and draw a cross marker
        for (x, y) in centers:
            # Draw a horizontal line of the cross
            cv2.line(image_with_centers, (x - marker_size, y - marker_size), (x + marker_size, y + marker_size), color, thickness)
            # Draw a vertical line of the cross
            cv2.line(image_with_centers, (x + marker_size, y - marker_size), (x - marker_size, y + marker_size), color, thickness)

        return image_with_centers



    @staticmethod
    def show_image(image, window_name="Image", wait_time=0):
        """
        Display the given image in a window.

        Parameters:
        - image: The image to display.
        - window_name: Name of the display window.
        - wait_time: Time to display the image (0 = wait indefinitely).
        """
        cv2.imshow(window_name, image)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()

    def show_image_with_transparency(image, window_name="Image", wait_time=0, background_color=(255, 255, 255)):
        """
        Displays an RGBA image with transparency simulated using a background color.

        Parameters:
        - image: RGBA image to display.
        - window_name: Name of the display window.
        - wait_time: Time to display the image (0 = wait indefinitely).
        - background_color: Background color in [B, G, R] format.
        """
        # Separate the alpha channel
        bgr_image = image[..., :3]
        alpha_channel = image[..., 3] / 255.0  # Normalize alpha to [0, 1]

        # Create a background image of the same size
        background = np.full_like(bgr_image, background_color, dtype=np.uint8)

        # Blend the image with the background using the alpha channel
        blended_image = cv2.convertScaleAbs(bgr_image * alpha_channel[..., None] + background * (1 - alpha_channel[..., None]))

        # Display the blended image
        cv2.imshow(window_name, blended_image)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()

    @staticmethod
    def save_image(image, output_path):
        """
        Save the image to the given path.

        Parameters:
        - image: The image to save.
        - output_path: Path where the image will be saved.
        """
        cv2.imwrite(output_path, image)
        print(f"Image saved to {output_path}")

    @staticmethod
    def base64_to_image(base64_string):
        image_arr = np.fromstring(base64.b64decode(base64_string), np.uint8)
        image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        return image


# Example usage
if __name__ == "__main__":

    light_pollution_image = SyntheticDataGenerator.generate_light_pollution_image(shape=(500, 500), num_sources=10)
    co2_emission_image = SyntheticDataGenerator.generate_co2_emission_image(shape=(500, 500), num_sources=15)

    ImageProcessor.show_image(light_pollution_image, window_name="Light Pollution Image")
    ImageProcessor.show_image(co2_emission_image, window_name="CO2 Emission Image")

    light_silence_image = ImageProcessor.invert_bw_image(light_pollution_image)

    ImageProcessor.show_image(light_silence_image, window_name="Inverted Light Pollution Image")
    
    good_house_image = ImageProcessor.multiply_masks(light_silence_image, co2_emission_image)

    ImageProcessor.show_image(good_house_image, window_name="Good House Image")

    bright_spots = ImageProcessor.detect_bright_spots(good_house_image, min_area=300, max_spots=5)



    print("Bright spots detected at:", bright_spots)

    vis = ImageProcessor.visualize_centers(good_house_image, bright_spots, marker_size=10, color=(255, 0, 0), thickness=2)

    ImageProcessor.show_image(vis, window_name="Detected Bright Spots")