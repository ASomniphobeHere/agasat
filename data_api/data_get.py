import base64
import cdsapi
from PIL import Image
import cfgrib
import numpy as np
import requests as re
from time import time
import io
import cv2
import json

DATASETS: tuple[str] = ("co2", "light", "health")


class Data_get:
    def __init__(self, tl_lat: float, tl_lon: float, br_lat: float, br_lon: float) -> None:
        self.tl_lon = tl_lon
        self.tl_lat = tl_lat
        self.br_lon = br_lon
        self.br_lat = br_lat
        
    def get(self, dataset: str) -> str:
        match (dataset):
            case "co2":
                return self.__get_co2()
            case "light":
                return self.__get_light()
            case "health":
                return self.__get_health()
            case _:
                raise ValueError("Invalid dataset name")
    
    def __file_to_b64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def __get_co2(self) -> str:
        dataset = "cams-global-ghg-reanalysis-egg4"
        request = {
            "pressure_level": ["950"],
            "model_level": ["56"],
            "date": ["2020-12-31/2020-12-31"],
            "step": ["3"],
            "data_format": "grib",
            "variable": ["carbon_dioxide"],
            "area": [self.tl_lat, self.tl_lon, self.br_lat, self.br_lon]
        }
        
        client = cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api", key="***YOUR CDS API KEY***")
        target = "data/co2_data.grib"
        client.retrieve(dataset, request, target)
        
        ds = cfgrib.open_datasets(target)[0]
        data_array = ds['co2'].values

        data_min = np.min(data_array)
        data_max = np.max(data_array)
        normalized_data = 255 * (data_array - data_min) / (data_max - data_min)

        image_data = normalized_data.astype(np.uint8)

        image_resized = Image.fromarray(image_data).resize((1000, 1000))

        return self.__file_to_b64(image_resized)
    

    def __get_light(self) -> str:
        if (abs(self.br_lat-self.tl_lat)>4.5):
            raise ValueError("Too large area") 
        
        new_token = lambda: base64.b64encode(f"{int(time()*1000)};isuckdicks:)".encode()).decode()
        qk: str = new_token()
        ql: str = f"viirs_2023"
        qt: str = "raster"
        quad = {
            "c1": [self.tl_lon, self.br_lat],
            "c2": [self.tl_lon, self.tl_lat],
            "c3": [self.br_lon, self.tl_lat],
            "c4": [self.br_lon, self.br_lat],
        }
        qd: str = f"LINESTRING({quad['c1'][0]} {quad['c1'][1]},{quad['c2'][0]} {quad['c2'][1]},{quad['c3'][0]} {quad['c3'][1]},{quad['c4'][0]} {quad['c4'][1]},{quad['c1'][0]} {quad['c1'][1]})"
        url = " https://www.lightpollutionmap.info/QueryRaster/?qk=" + qk + "&ql=" + ql + "&qt=" + qt + "&qd=" + qd
        response = re.get(url)
        out_stream = io.BytesIO(response.content)
        
        lights = Image.open(out_stream)
        maped = lights.convert("L")
        maped = maped.resize((1000, 1000))
        return self.__file_to_b64(maped)
    
    def __get_health(self) -> str:
        with open("data_api/data.json", "r") as file:
            data = json.load(file)
        
        
        image = np.zeros((1000, 1000, 4), dtype=np.uint8)
        image[:, :, 3] = 255  # Set the alpha channel to fully opaque
        
        radius = 5000 // (abs(self.tl_lat - self.br_lat) * 111.1 )
        
        for location in data["coords"]:
            loc_lat = location[1]
            loc_lon = location[0]
            if loc_lat > self.tl_lat and loc_lat < self.br_lat and loc_lon > self.tl_lon and loc_lon < self.br_lon:
                x = int((loc_lon - self.tl_lon) / (self.br_lon - self.tl_lon) * 1000)
                y = int((self.tl_lat - loc_lat) / (self.tl_lat - self.br_lat) * 1000)
                
                # Create a blank image with transparency (RGBA)
               
                # Create a mask for the circle with opacity falloff
                circle_mask = np.zeros((1000, 1000), dtype=np.uint8)
                for r in range(0, int(radius), 5):
                    intensity = int(255 * (1 - r / radius))  # Linear falloff
                    temp_mask = np.zeros((1000, 1000), dtype=np.uint8)
                    cv2.circle(temp_mask, (x, y), r, intensity, -1)  # Create a temporary mask
                    circle_mask = np.maximum(circle_mask, temp_mask)  # Accumulate the intensity
                # Add the circle to the image
                for c in range(3):
                    image[:, :, c] = np.maximum(image[:, :, c], circle_mask)
                image[:, :, 3] = np.maximum(image[:, :, 3], circle_mask)

                # Convert the image to PIL format and save
        image_pil = Image.fromarray(image)
        # image_pil.save("data/health_data.png")
        return self.__file_to_b64(image_pil)

        
# data_get = Data_get(54, 19, 55, 18)
# stringdata = data_get.get("co2") -> returns base64 encoded string
