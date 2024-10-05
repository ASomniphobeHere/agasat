import base64
import cdsapi
from PIL import Image
import cfgrib
import numpy as np
import requests as re
from time import time
import io

DATASETS: tuple[str] = ("co2", "light")


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
        
        client = cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api", key="***REMOVED***")
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

        
# data_get = Data_get(54, 19, 55, 18)
# stringdata = data_get.get("co2") -> returns base64 encoded string
