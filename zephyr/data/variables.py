"""Atomospheric and surface variables used in the model."""

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 121


# (TODO) Allow this to be configurable via a YAML file at project level.
ATMOSPHERIC_VARIABLE_NAMES = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
]

ATMOSPHERIC_LEVELS = [50, 100, 200, 300]

ATMOSPHERIC_VARIABLES = [
    f"{var}_{level}" for var in ATMOSPHERIC_VARIABLE_NAMES for level in ATMOSPHERIC_LEVELS
]

SURFACE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure"
]

FORCED_VARIABLES = [
    "land_sea_mask"
]


VARIABLES_CATALOG = ATMOSPHERIC_VARIABLES + SURFACE_VARIABLES + FORCED_VARIABLES


VARIABLE_NORMALIZATION_CONSTANTS = {
    "10m_u_component_of_wind": (0.006723432336002588, 5.393361568450928),
    "10m_v_component_of_wind": (0.18406322598457336, 4.658501625061035),
    "2m_temperature": (277.9703369140625, 21.307506561279297),
    "geopotential_50": (199424.828125, 5549.091796875),
    "geopotential_100": (157548.5625, 5401.45751953125),
    "geopotential_200": (115189.390625, 5809.91943359375),
    "geopotential_300": (89288.6953125, 5089.02099609375),
    "land_sea_mask": (0.3370121419429779, 0.45053866505622864),
    "mean_sea_level_pressure": (100946.703125, 1291.50634765625),
    "specific_humidity_50": (2.772770585579565e-06, 2.585598224413843e-07),
    "specific_humidity_100": (2.727234686972224e-06, 4.866903395850386e-07),
    "specific_humidity_200": (1.915411303343717e-05, 2.1535524865612388e-05),
    "specific_humidity_300": (0.00012385605077724904, 0.0001588654558872804),
    "temperature_100": (208.9010009765625, 12.456430435180664),
    "temperature_200": (218.06495666503906, 6.846928119659424),
    "temperature_300": (228.61935424804688, 10.673646926879883),
    "temperature_50": (213.32025146484375, 9.67135238647461),
    "u_component_of_wind_50": (5.808347702026367, 14.504850387573242),
    "u_component_of_wind_100": (10.369661331176758, 13.038705825805664),
    "u_component_of_wind_200": (14.446528434753418, 17.26558494567871),
    "u_component_of_wind_300": (12.048169136047363, 16.824033737182617),
    "v_component_of_wind_50": (-0.0019055908778682351, 6.836955547332764),
    "v_component_of_wind_100": (0.030090900138020515, 7.423751354217529),
    "v_component_of_wind_200": (-0.023692801594734192, 11.7735013961792),
    "v_component_of_wind_300": (-0.023223580792546272, 13.246015548706055),
    "vertical_velocity_50": (-4.9137470341520384e-05, 0.010115926153957844),
    "vertical_velocity_100": (-5.608960236713756e-06, 0.0198441781103611),
    "vertical_velocity_200": (4.175609501544386e-05, 0.06212469935417175),
    "vertical_velocity_300": (0.0001021720891003497, 0.11398006230592728),
}
