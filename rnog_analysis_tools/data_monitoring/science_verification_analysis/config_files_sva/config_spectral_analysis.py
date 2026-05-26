from NuRadioReco.utilities import units

SPECTRAL_BANDS = {
    "galactic_excess": {"freq_min": 80*units.MHz, "freq_max": 120*units.MHz},
    "freq_360_380MHz": {"freq_min": 360*units.MHz, "freq_max": 380*units.MHz},
    "freq_482_485MHz": {"freq_min": 482*units.MHz, "freq_max": 485*units.MHz},
    "freq_240_272MHz": {"freq_min": 240*units.MHz, "freq_max": 272*units.MHz},
}

ALPHA_SPEC = 0.005
CI_THRESHOLDS_SPEC = (0.6, 0.75)
NORMALIZATION_BAND = {"freq_min": 500*units.MHz, "freq_max": 650*units.MHz}
LOG_RATIO_THRESHOLDS_SPEC = (0.04, 0.07, 0.1) # (no_excess, weak_excess, moderate_excess)