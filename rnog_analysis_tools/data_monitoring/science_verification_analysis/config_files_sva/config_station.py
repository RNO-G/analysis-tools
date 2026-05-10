from NuRadioReco.utilities import units


DEFAULT_CONFIG = {
    "all_channels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 , 19 , 20, 21, 22, 23],
    "surface_channels": [12, 13, 14, 15, 16, 17, 18 , 19 , 20],
    "deep_channels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23],
    "upward_channels": [13, 16, 19],
    "downward_channels": [12, 14, 15, 17, 18, 20],
    "vpol_channels": [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23],
    "hpol_channels": [4, 8, 11, 21],
    "phased_array_channels": [0, 1, 2, 3],
    "reference_channels": [12, 14, 15, 17, 18, 20], # downward facing surface channels
    "reference_channels_galaxy": [12, 14, 15, 17, 18, 20], # downward facing surface channels
}

STATION_SPECIFIC_ADJUSTMENTS = {
    14: {
        "surface_channels": [12, 13, 14, 15, 16, 17, 18 , 19],
        "deep_channels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23],
        "upward_channels": [13, 15, 16, 18],
        "downward_channels": [12, 14, 17, 19],
        "vpol_channels": [0, 1, 2, 3, 5, 6, 7, 9, 10, 20, 22, 23], # 20 is a vpol channel in station 14 instead of lpda
        "reference_channels": [12, 14, 17, 19], # downward facing surface channels for station 14
        "reference_channels_galaxy": [12, 14, 19], # downward facing surface channels for station 14 except for 17, which behaves weirdly in that region
    }
}

def get_station_config(station_id):
    cfg = DEFAULT_CONFIG.copy()
    if station_id in STATION_SPECIFIC_ADJUSTMENTS:
        adjustments = STATION_SPECIFIC_ADJUSTMENTS[station_id]
        cfg.update(adjustments)
    return cfg

sampling_rate = {"after_2024": 2.4*units.GHz,
                 "before_2024": 3.2*units.GHz}