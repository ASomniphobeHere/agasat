import cdsapi

dataset = "cams-global-ghg-reanalysis-egg4"
request = {
    "pressure_level": ["950"],
    "model_level": ["56"],
    "date": ["2020-12-31/2020-12-31"],
    "step": ["3"],
    "data_format": "grib",
    "variable": ["carbon_dioxide"],
    "area": [70.88, -8.37, 32.09, 36.49]
}
target ="co2_data.grib"
client = cdsapi.Client()
result = client.retrieve(dataset, request, target)

