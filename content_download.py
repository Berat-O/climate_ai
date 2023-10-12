import climate_learn as cl


cl.data.download_weatherbench(
    dst="./content/ClimateLearn/temperature",
    dataset="era5",
    variable="temperature_850"
)

cl.data.download_weatherbench(
    dst="./content/ClimateLearn/geopotential",
    dataset="era5",
    variable="geopotential_500"
)




