from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.controllers.llm_controller import LLMController
from app.controllers.image_processor import ImageProcessor
from app.controllers.function_executor import FunctionExecutor
from fastapi.responses import FileResponse
from typing import List, Tuple
from PIL import Image
import os
import uuid
import random
import numpy as np
import cv2

# Set up the main app and router
app = FastAPI()
router = APIRouter(prefix="/prompt", tags=["GIS Magic"])

# Create a directory for storing generated images
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# Create a Pydantic schema for the input text and coordinates
class TextInput(BaseModel):
    text: str
    coordinates: list[float] = Field(..., min_items=4, max_items=4)  # Ensure exactly 4 float coordinates

# Initialize the LLM controller
llm_controller = LLMController()

def get_datasets(datasets_list: List[str]) -> dict:
    """
    Simulates dataset retrieval using random blobs. Replace this with actual loading logic.
    
    Parameters:
    - datasets_list: List of requested datasets.
    
    Returns:
    - A dictionary mapping dataset names to synthetic numpy images.
    """
    def generate_base_image(shape=(500, 500), num_blobs=70):
        """
        Generates a base image with random amorphous blobs using random walk and morphological operations.

        Parameters:
        - shape: Shape of the output image (height, width).
        - num_blobs: Number of distinct blob regions to generate.

        Returns:
        - base_image: A synthetic image with amorphous blobs.
        """
        base_image = np.zeros(shape, dtype=np.uint8)

        # Create random blobs using a combination of random walk and morphological transformations
        for _ in range(num_blobs):
            # Create a random starting point
            x, y = np.random.randint(0, shape[1]), np.random.randint(0, shape[0])

            # Generate an empty mask for the random walk
            mask = np.zeros(shape, dtype=np.uint8)
            mask[y, x] = 255

            # Perform a random walk to create an amorphous shape
            for _ in range(1000):  # Number of random walk steps
                x += np.random.randint(-3, 4)
                y += np.random.randint(-3, 4)
                x = np.clip(x, 0, shape[1] - 1)  # Keep within bounds
                y = np.clip(y, 0, shape[0] - 1)
                mask[y, x] = 255

            # Use morphological operations to smooth the shape and create blob-like regions
            mask = cv2.dilate(mask, None, iterations=np.random.randint(5, 10))
            mask = cv2.erode(mask, None, iterations=np.random.randint(3, 7))

            # Add the generated blob to the base image with random intensity
            intensity = np.random.random() * 0.9
            base_image = cv2.add(base_image, (mask > 0).astype(np.uint8) * int(255 * intensity * intensity * intensity))

        # Apply Gaussian blur for a more natural look
        base_image = cv2.GaussianBlur(base_image, (101, 101), sigmaX=10, sigmaY=10)

        base_image = apply_uniform_buckets(base_image, num_buckets=20)

        return base_image
    
    def apply_uniform_buckets(image, num_buckets=12):
        """
        Apply uniform segmentation on the image, dividing pixel values into equal intensity buckets.

        Parameters:
        - image: Input image as a numpy array.
        - num_buckets: Number of uniform buckets to segment the image into.

        Returns:
        - segmented_image: Image with pixel values replaced by bucket values.
        """
        # Calculate the bucket size based on the number of buckets
        bucket_size = 255 // num_buckets
        segmented_image = np.zeros_like(image)

        # Replace pixel values based on the bucket they fall into
        for i in range(num_buckets):
            lower_bound = i * bucket_size
            upper_bound = (i + 1) * bucket_size

            # Use the average value of the bucket as the replacement
            bucket_value = (lower_bound + upper_bound) // 2

            # Assign bucket value to pixels in the range [lower_bound, upper_bound)
            mask = (image >= lower_bound) & (image < upper_bound)
            segmented_image[mask] = bucket_value

        # Handle the final bucket which might include the value 255
        segmented_image[image >= (num_buckets * bucket_size)] = 255

        return segmented_image
    
    # Create synthetic datasets with random blobs for demonstration purposes
    datasets = {}
    for name in datasets_list:
        datasets[name] = generate_base_image()

    return datasets

@router.post("/")
def process_input(input: TextInput):
    """
    Processes the input text and coordinates, creates images, and returns a response with the final results.
    """
    # Step 1: Use location-based template to get relevant datasets and reasoning
    location_response = llm_controller.handle_location_based_question(input.text)

    # Step 2: Use the data analysis template to get processing functions and their order
    data_analysis_response = llm_controller.handle_data_analysis_question(input.text, location_response)
    
    # Parse the data_analysis_response into a JSON-like format (mocking here for example purposes)
    try:
        output_json = eval(data_analysis_response)  # WARNING: Use safer parsing in real-world applications!
    except Exception as e:
        return {"error": f"Failed to parse data analysis response: {str(e)}"}

    # Step 3: Retrieve the required datasets based on the output_json
    datasets = get_datasets(output_json["datasets"])
    for key, value in datasets.items():
        ImageProcessor.show_image(value, title=key)  # Display the dataset images

    # Step 4: Initialize the FunctionExecutor and run the sequence of functions
    executor = FunctionExecutor(datasets)
    try:
        final_image = executor.execute_functions(output_json)
    except Exception as e:
        return {"error": f"Failed to execute functions: {str(e)}"}

    # Step 5: Save the final result image to the server
    final_image_filename = f"{uuid.uuid4()}.png"
    final_image_path = os.path.join(output_dir, final_image_filename)
    cv2.imwrite(final_image_path, final_image)

    # Step 6: Generate highlight points from the final image
    try:
        highlight_points = ImageProcessor.detect_bright_spots(final_image, blur_radius=(11, 11), min_area=300, max_spots=5)
    except Exception as e:
        return {"error": f"Failed to generate highlight points: {str(e)}"}

    # Step 7: Create a template user response
    template_response = f"""
    Based on the provided analysis, the following datasets were found to be relevant: {', '.join(output_json['datasets'])}.
    The resulting image has been saved and contains the specified high-value points.
    """

    # Return the response with relevant details
    return {
        "input": input.text,
        "response": template_response.strip(),
        "coordinates": input.coordinates,
        "highlight_points": highlight_points,
        "final_image_url": f"/prompt/image/{final_image_filename}"
    }

@router.get("/image/{image_filename}")
def get_large_image(image_filename: str):
    """
    Serves the large image file from the server.
    
    Parameters:
    - image_filename: Name of the image file to retrieve.
    
    Returns:
    - FileResponse: The image file.
    """
    image_path = os.path.join(output_dir, image_filename)
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    else:
        return {"error": "Image not found"}

if __name__ == "__main__":
    image_processor = ImageProcessor()
    list_names = ["base_image", "another_image"]
    base_image = get_datasets(list_names)
    for n in list_names:
        image_processor.show_image(base_image[n], window_name=n)