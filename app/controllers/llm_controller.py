from openai import OpenAI

# Set your OpenAI API key here or load it from an environment variable
openai_api_key = "***REMOVED***"

client = OpenAI(api_key=openai_api_key)

class LLMController:
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.location_based_template = """You have several black and white satellite images, where black pixels represent low values and white pixels represent high values of a given measurement.

These are the datasets that are available:
{
	"light": Light pollution, low values means low pollution,
	"co2": CO2 emissions, low values means low emissions,
	"precipitation": Rainfall intensity, low values means low rainfall,
	"urban_density": Population density in urban areas, low values means low density,
	"temperature": Average temperature data, low values means low average temperature,
	"wind_speed": Wind speed, low value means low speeds,
	"elevation": Altitude variations, low value means low elevation,
	"vegetation": Forest and green space areas, low value means low green spaces
}

You DON'T HAVE TO use all of the datasets, use only the ones that actually are relevant to the question.
Try to make the model as simple as possible(use the least number of datasets) while also actually taking all the highly relevant datasets.

Give list of Relevant datasets, Reasoning, and Best Location
example
Question: Where is the best location for a new wind farm?
Answer:
""
Relevant Datasets:
Wind speed, urban density, vegetation, elevation.
Reasoning:
Wind speed directly impacts energy generation potential.
Urban density avoids conflicts and reduces noise complaints.
Vegetation helps avoid ecologically sensitive areas.
Elevation influences wind availability and stability.
Best Location: High wind speed, low urban density, low vegetation, and high elevation areas.
""
Give very concise reasoning to what datasets positively or negatively impact the answer to the question, and what specific aspects of them."""

        self.data_analysis_template = """You have several black and white satellite images, where black pixels represent low values and white pixels represent high values of a given measurement.

You can use these function to perform data analysis and create a dataset that closely answers the users question.
    def multiply_masks(image1, image2):
    ""
    Multiply to isolate regions where both images are strong. 
    Behavior: High overlap areas become brighter, places where at least one value is low turn dark.
    ""

def invert_bw_image(image):
    ""
    Reverse values to emphasize regions with initially low values.
    Behavior: Black areas turn white and vice versa, revealing low-value patterns.
    ""

def weighted_average(image_list, weight_list):
    ""
    Combine images to reflect varying dataset importance.
    Behavior: Blends images, emphasizing regions based on weights. Weights should sum to 1.
    ""

def pixel_threshold(image, threshold_value=128):
    ""
    Mark regions that surpass a set intensity.
    Behavior: Pixels above the threshold turn white; others darken.
    ""

def edge_detection(image, min_val=100, max_val=200):
    ""
    Identify sudden changes in pixel values.
    Behavior: Creates thin outlines highlighting boundaries.
    ""

def gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    ""
    Smooth the image to reduce sharp transitions.
    Behavior: Blurs noise and small details to show general trends.
    ""


Output JSON format:
{
	"datasets": ["light", "co2", ...(other dataset names)],
	"functions": [
		{
			"func": "name of the function",
			"params": [ (paramaters of the function), ... ],
			"output_image": "the new image variable name"
		},
		... (other functions, in order)
	],
	"final_result": variable name of the resulting image
}
the output_image can be used as a variable in later functions, after it has been defined
If you pass images as parameters, instead of writing the variables as themselves, write their names as strings


These are the datasets that are available:
{
	"light": Light pollution, low values means low pollution,
	"co2": CO2 emissions, low values means low emissions,
	"precipitation": Rainfall intensity, low values means low rainfall,
	"urban_density": Population density in urban areas, low values means low density,
	"temperature": Average temperature data, low values means low average temperature,
	"wind_speed": Wind speed, low value means low speeds,
	"elevation": Altitude variations, low value means low elevation,
	"vegetation": Forest and green space areas, low value means low green spaces
}

You DON'T HAVE TO use all of the datasets, use only the ones that actually are relevant to the question.

examples of output
Question 1:
"Where are the regions with the highest combination of light pollution and CO2 emissions?"
Answer 1:
{
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

Question 2:
"Which regions have the highest rainfall intensity with a high level of vegetation?"
Answer 2:
{
    "datasets": ["precipitation", "vegetation"],
    "functions": [
        {
            "func": "multiply_masks",
            "params": ["precipitation", "vegetation"],
            "output_image": "high_rainfall_vegetation"
        }
    ],
    "final_result": "high_rainfall_vegetation"
}
Question 3:
"Show areas with significant temperature variations along regions of low altitude."
Answer 3:
{
    "datasets": ["temperature", "elevation"],
    "functions": [
        {
            "func": "invert_bw_image",
            "params": ["elevation"],
            "output_image": "inverted_elevation"
        },
        {
            "func": "multiply_masks",
            "params": ["temperature", "inverted_elevation"],
            "output_image": "temperature_low_altitude"
        }
    ],
    "final_result": "temperature_low_altitude"
}

After applying these functions we will extract n places with the highest pixel value in the final_result image, and assign them to be the answer.

Use all of this information to create a statistics mask that closely answers the user question."""
    
    def get_response(self, assistant_prompt: str, user_prompt: str) -> str:
        """
        Sends the given prompts to GPT-4 and returns the generated response.
        """
        try:
            messages = [
                {"role": "assistant", "content": assistant_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Call the OpenAI API to generate a response
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=500,
                frequency_penalty=0.0
            )
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            return response_text, tokens_used

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    def handle_location_based_question(self, user_question: str) -> str:
        """
        Process and respond to location-based questions using the location-based template.
        """
        user_prompt = f"Question: {user_question}\n"
        response, _ = self.get_response(self.location_based_template, user_prompt)
        return response

    def handle_data_analysis_question(self, user_question: str, reasoning: str) -> str:
        """
        Process and respond to questions that require data analysis using the available functions and datasets.
        """
        # Include reasoning in the user prompt for data analysis questions
        combined_user_prompt = f"Question: {user_question}\n{reasoning}\n"
        response, _ = self.get_response(self.data_analysis_template, combined_user_prompt)
        return response