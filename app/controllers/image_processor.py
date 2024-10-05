import cv2
import numpy as np
from imutils import contours
import imutils
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.utils.synthetic_data import SyntheticDataGenerator

class ImageProcessor:
    @staticmethod
    def detect_bright_spots(image, blur_radius=(11, 11), min_area=300, max_spots=5):
        """
        Detect and return the centers of the top bright spots in the given image, sorted by area.

        Parameters:
        - image: A pre-loaded OpenCV image object (numpy array).
        - blur_radius: Kernel size for the Gaussian blur.
        - min_brightness: Minimum brightness threshold to consider a region as a bright spot.
        - min_area: Minimum area (in pixels) for a region to be considered a bright spot.
        - max_spots: Maximum number of bright spots to return, sorted by area.

        Returns:
        - centers: A list of tuples representing the (x, y) coordinates of the centers of the bright spots.
        """

        # Step 1: Apply Gaussian blur to the grayscale image
        blurred = cv2.GaussianBlur(image, blur_radius, 0)

        # 85% of the maximum image pixel value
        min_brightness = 0.85 * cv2.minMaxLoc(blurred)[1]

        # Step 2: Threshold the blurred image to reveal light regions
        _, thresh = cv2.threshold(blurred, min_brightness, 255, cv2.THRESH_BINARY)

        # Step 3: Perform a series of erosions and dilations to remove small blobs of noise
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        # Step 4: Use OpenCV's connected components to label the thresholded image
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=cv2.CV_32S)

        # Step 5: Create a mask for large components only and collect component areas and centers
        components = []  # List to hold (area, (center_x, center_y)) tuples
        mask = np.zeros(thresh.shape, dtype="uint8")

        for i in range(1, num_labels):  # Start from 1 to skip the background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                component_mask = (labels == i).astype("uint8") * 255
                mask = cv2.add(mask, component_mask)

                # Calculate the center of the bright spot
                cX = int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2)
                cY = int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2)

                # Add the component's area and center coordinates to the list
                components.append((area, (cX, cY)))

        # Step 6: Sort the components by area in descending order and select the top `max_spots`
        components = sorted(components, key=lambda x: x[0], reverse=True)[:max_spots]

        # Step 7: Extract the center coordinates of the top components
        centers = [center for _, center in components]

        return centers

    
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
    def gaussian_blur(image, kernel_size=(5, 5), sigma=0):
        """
        Apply Gaussian blur to the input image.

        Parameters:
        - image: The input image (numpy array).
        - kernel_size: Size of the Gaussian kernel (default: 5x5).
        - sigma: Standard deviation of the Gaussian kernel (default: 0).

        Returns:
        - blurred_image: The image after applying Gaussian blur.
        """
        blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
        return blurred_image
    
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