import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, skew
import os
import logging

logger = logging.getLogger(__name__)

# For dataProbiderRNOG
def calculate_vrms(trace_arr, event_info):
    '''Calculate Vrms for each channel and event according to trigger types.'''
    vrms_arr = np.std(trace_arr, axis=2)  # (n_channels, n_events)

    force_mask = event_info["triggerType"] == "FORCE"
    radiant0_mask = event_info["triggerType"] == "RADIANT0"
    radiant1_mask = event_info["triggerType"] == "RADIANT1"
    lt_mask = event_info["triggerType"] == "LT"

    vrms_arr_force = vrms_arr[:, force_mask]
    vrms_arr_radiant0 = vrms_arr[:, radiant0_mask]
    vrms_arr_radiant1 = vrms_arr[:, radiant1_mask]
    vrms_arr_lt = vrms_arr[:, lt_mask]

    return vrms_arr, vrms_arr_force, vrms_arr_radiant0, vrms_arr_radiant1, vrms_arr_lt

def get_rms_per_trigger_monitoring(rms_arr, force_mask, lt_mask, radiant0_mask, radiant1_mask):
    '''Get RMS values for each channel and event according to trigger types for monitoring data.'''
    
    # Still named Vrms to keep consistent with the dataProviderRNOG case, but these are actually RMS values calculated in the monitoring pipeline, not Vrms calculated from traces as in the dataProviderRNOG case. The analysis functions will be the same for both cases, just the input values are different.
    vrms_arr = rms_arr  # (n_channels, n_events)
    vrms_arr_force = rms_arr[:, force_mask]
    vrms_arr_lt = rms_arr[:, lt_mask]
    vrms_arr_radiant0 = rms_arr[:, radiant0_mask]
    vrms_arr_radiant1 = rms_arr[:, radiant1_mask]

    return vrms_arr,vrms_arr_force, vrms_arr_radiant0, vrms_arr_radiant1, vrms_arr_lt

def kde_modality(vrms_arr, channel_list, kde_modality_config=None):
    '''Calculate KDE and modality for Vrms distributions.'''

    if kde_modality_config is None:
        kde_modality_config = {}
    
    bandwidth = kde_modality_config.get("bandwidth", None)
    grid_points = kde_modality_config.get("grid_points", 512)
    peak_prominence = kde_modality_config.get("peak_prominence", 0.01)
    height_threshold = kde_modality_config.get("height_threshold", 0.05)

    modality_dict = {}

    for ch in channel_list:
        vrms_ch = vrms_arr[ch]
        vrms_ch = vrms_ch[~np.isnan(vrms_ch)]

        kde = gaussian_kde(vrms_ch, bw_method=bandwidth)
        vrms_min = np.min(vrms_ch)
        vrms_max = np.max(vrms_ch)
        vrms_grid = np.linspace(vrms_min, vrms_max, grid_points)
        kde_values = kde(vrms_grid)

        # Adjust prominence to be relative to the KDE range
        abs_prom = peak_prominence * (np.max(kde_values) - np.min(kde_values))

        peaks, properties = find_peaks(kde_values, prominence=abs_prom, height=height_threshold*np.max(kde_values))

        modality_dict[ch] = {
            "kde": kde,
            "vrms_grid": vrms_grid,
            "kde_values": kde_values,
            "n_peaks": len(peaks),
            "peaks": peaks,
            "prominences": properties["prominences"],
        }

    return modality_dict

def tail_fraction_and_trimmed_skew_two_sided(vrms_arr, channel_list, skewness_config=None):
    '''Calculate tail fraction and two-sided trimmed skewness for Vrms distributions.'''
    tail_dict = {}

    if skewness_config is None:
     skewness_config = {}

    lower_percentile = skewness_config.get("lower_percentile", 25)
    upper_percentile = skewness_config.get("upper_percentile", 75)
    extreme_k = skewness_config.get("extreme_k", 2)
    min_events_for_skew = skewness_config.get("min_events_for_skew", 30)
    max_tail_frac_for_trimmed_skew = skewness_config.get("max_tail_frac_for_trimmed_skew", 0.05)

    for ch in channel_list:
        vrms_ch = vrms_arr[ch]
        vrms_ch = vrms_ch[~np.isnan(vrms_ch)]
        n_events = len(vrms_ch)

        full_skew = skew(vrms_ch, bias=False)

        q1, q3 = np.percentile(vrms_ch, [lower_percentile, upper_percentile])
        iqr = q3 - q1

        lower_bound = q1 - extreme_k * iqr
        upper_bound = q3 + extreme_k * iqr

        high_mask = vrms_ch > upper_bound
        high_frac = np.mean(high_mask)

        low_mask = vrms_ch < lower_bound
        low_frac = np.mean(low_mask)

        core_high = vrms_ch[~high_mask]
        skew_trim_high = skew(core_high, bias=False) if len(core_high) > min_events_for_skew and high_frac < max_tail_frac_for_trimmed_skew else np.nan

        core_low = vrms_ch[~low_mask]
        skew_trim_low = skew(core_low, bias=False) if len(core_low) > min_events_for_skew and low_frac < max_tail_frac_for_trimmed_skew else np.nan

        tail_dict[ch] = {
            "n_events": n_events,
            "full_skew": full_skew,
            "high_tail_fraction": high_frac,
            "low_tail_fraction": low_frac,
            "trimmed_skew_high": skew_trim_high,
            "trimmed_skew_low": skew_trim_low,
        }

    return tail_dict

def report_vrms_characteristics(modality_dict, tail_dict, channel_list, report_config=None):
    
    if report_config is None:
        report_config = {}

    strong_skew = report_config.get("strong_skew", 0.3)
    extreme_skew = report_config.get("extreme_skew", 0.5)
    delta_skew_min = report_config.get("delta_skew_min", 0.25)
    rare_max_high_frac = report_config.get("rare_max_high_frac", 0.01)
    rare_max_low_frac = report_config.get("rare_max_low_frac", 0.01)
    mod_max_high_frac = report_config.get("mod_max_high_frac", 0.05)
    mod_max_low_frac = report_config.get("mod_max_low_frac", 0.05)
    
    modality_channels = {}
    tail_label_channels = {}

    for ch in channel_list:
        n_peaks = modality_dict[ch]["n_peaks"]
        if n_peaks == 0:
            modality = "flat/noisy"
        elif n_peaks == 1:
            modality = "unimodal"
        elif n_peaks == 2:
            modality = "bimodal"
        else:
            modality = f"multimodal ({n_peaks} peaks)"

        full_skew   = tail_dict[ch]["full_skew"]
        high_frac   = tail_dict[ch]["high_tail_fraction"]
        low_frac    = tail_dict[ch]["low_tail_fraction"]
        skew_trim_h = tail_dict[ch]["trimmed_skew_high"]
        skew_trim_l = tail_dict[ch]["trimmed_skew_low"]

        if np.isnan(full_skew):
            tail_label = "no significant tails"
            tail_frac = None
            if tail_frac is not None:
                tail_label += f" (fraction: {tail_frac:.3f})"
            continue

        if not np.isnan(skew_trim_h):
            dskew_h = full_skew - skew_trim_h
        else:
            dskew_h = 0
        if not np.isnan(skew_trim_l):
            dskew_l = full_skew - skew_trim_l
        else:
            dskew_l = 0

        if 0 < high_frac < rare_max_high_frac and full_skew > extreme_skew and dskew_h > delta_skew_min:
            tail_label = "rare high extremes"
            tail_frac = high_frac

        elif 0 < low_frac < rare_max_low_frac and full_skew < -extreme_skew and dskew_l < -delta_skew_min:
            tail_label = "rare low extremes"
            tail_frac = low_frac

        elif full_skew > strong_skew:
            if high_frac < mod_max_high_frac:
                tail_label = "moderate high skew"
            else:
                tail_label = "bulk high skew"
            tail_frac = high_frac

        elif full_skew < -strong_skew:
            if low_frac < mod_max_low_frac:
                tail_label = "moderate low skew"
            else:
                tail_label = "bulk low skew"
            tail_frac = low_frac

        else:
            tail_label = "no significant tails"
            tail_frac = None

        # output summary
        if tail_frac is not None:
            tail_label += f" (fraction: {tail_frac:.3f})"
        modality_channels[ch] = modality
        tail_label_channels[ch] = tail_label

    return modality_channels, tail_label_channels