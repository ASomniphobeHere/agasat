from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.controllers.llm_controller import LLMController
from app.controllers.image_processor import ImageProcessor
from app.controllers.function_executor import FunctionExecutor
from data_api.data_get import Data_get
from fastapi.responses import FileResponse
from typing import List, Tuple
from PIL import Image
import os
import uuid
import random
import numpy as np
import cv2
import json

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


def get_datasets(datasets_list: List[str], coords: Tuple[float, float, float, float]) -> dict:
    datasets = {}
    data_get = Data_get(*coords)  # Initialize the Data_get object with the coordinates
    for ds_name in datasets_list:
        ds_base64 = data_get.get(ds_name)  # Retrieve the dataset using the Data_get object
        ds_image = ImageProcessor.base64_to_image(ds_base64)  # Convert the base64 string to an image
        datasets[ds_name] = ds_image  # Store the image in the datasets dictionary

    return datasets

@router.post("/")
def process_input(input: TextInput):
    """
    Processes the input text and coordinates, creates images, and returns a response with the final results.
    """
    # Step 1: Use location-based template to get relevant datasets and reasoning
    location_response = llm_controller.handle_location_based_question(input.text)
    print(location_response)
    # Step 2: Use the data analysis template to get processing functions and their order
    data_analysis_response = llm_controller.handle_data_analysis_question(input.text, location_response)
    print(data_analysis_response)
    # Parse the data_analysis_response into a JSON-like format (mocking here for example purposes)
    try:
        output_json = json.loads(data_analysis_response)  # WARNING: Use safer parsing in real-world applications!
    except Exception as e:
        return {"error": f"Failed to parse data analysis response: {str(e)}"}

    print(output_json)

    # Step 3: Retrieve the required datasets based on the output_json
    datasets = get_datasets(output_json["datasets"], input.coordinates)
    for key, value in datasets.items():
        ImageProcessor.show_image(value, window_name=key)  # Display the dataset images

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
    Found {len(highlight_points)} areas of interest based on your query. Here are the coordinates of these areas: {highlight_points}.
    Relevant datasets: {', '.join(output_json['datasets'])}.
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