# --------------------------------
# DEPRECATED
# --------------------------------
import requests as re
# from geotiff import GeoTiff
import numpy as np


def get_tiff(top_left_lat: float, top_left_lon: float, bottom_right_lat: float, bottom_right_lon: float, year: int = 2023):
    qk: str = "MTcyODExNzg5MzExNTtpc3Vja2RpY2tzOik="
    ql: str = f"viirs_{year}"
    qt: str = "raster"
    quad = {
        "c1": [top_left_lon, bottom_right_lat],
        "c1r": [56.90246278059922, 23.936897289759077],
        "c2": [top_left_lon, top_left_lat],
        "c2r": [57.0829061552013, 23.936897289759077],
        "c3": [bottom_right_lon, top_left_lat],
        "c3r": [57.0829061552013, 24.268141145561017],
        "c4": [bottom_right_lon, bottom_right_lat],
        "c4r": [56.90246278059922, 24.268141145561017],
    }
    qd: str = f"LINESTRING({quad['c1'][0]} {quad['c1'][1]},{quad['c2'][0]} {quad['c2'][1]},{quad['c3'][0]} {quad['c3'][1]},{quad['c4'][0]} {quad['c4'][1]},{quad['c1'][0]} {quad['c1'][1]})"
    url = " https://www.lightpollutionmap.info/QueryRaster/?qk=" + qk + "&ql=" + ql + "&qt=" + qt + "&qd=" + qd
    response = re.get(url)
    filename = "light2.tif"
    with open(filename, 'wb') as f:
        f.write(response.content)
    return

get_tiff(57.0829061552013, 23.936897289759077, 56.90246278059922, 24.268141145561017)
# gt = GeoTiff("light2.tif")