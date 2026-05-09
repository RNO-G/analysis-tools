kde_modality_function_parameters = {
    "bandwidth": None, # bw_method for gaussian_kde, can be 'scott', 'silverman', or a scalar. None defaults to 'scott'.
    "grid_points": 512,
    "peak_prominence": 0.01,
    "height_threshold": 0.05
}

skewness_function_parameters = {
    "lower_percentile": 25,
    "upper_percentile": 75,
    "extreme_k": 2,
    "min_events_for_skew": 30,
    "max_tail_frac_for_trimmed_skew": 0.05
}

report_vrms_function_parameters = {
    "strong_skew": 0.3,
    "extreme_skew": 0.5,
    "delta_skew_min": 0.25,
    "rare_max_high_frac": 0.01,
    "rare_max_low_frac": 0.01,
    "mod_max_high_frac": 0.05,
    "mod_max_low_frac": 0.05
}


