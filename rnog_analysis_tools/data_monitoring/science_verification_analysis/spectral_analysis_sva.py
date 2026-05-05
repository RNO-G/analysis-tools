import numpy as np
from NuRadioReco.utilities import units
from scipy.signal import binomtest
import logging

logger = logging.getLogger(__name__)

def find_amplitude_ratio_in_band(freqs, norm_spec_arr, upward_channels, downward_channels, reference_channels, freq_min, freq_max):
    '''Find normalized amplitude ratio of upward vs downward channels in a given frequency band. This will be replaced by lab measurements in the future.'''
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    ratio_list = []
    spectrum_list = []

    ref_specs = np.stack([norm_spec_arr[ch][:, freq_mask] for ch in reference_channels], axis=0)
    ref_spectra_per_ch = np.median(ref_specs, axis=2)   # (n_ref_ch, n_events)
    ref_spec = np.median(ref_spectra_per_ch, axis=0) # (n_events,)

    for ch in upward_channels + downward_channels:
        masked_spec = norm_spec_arr[ch][:, freq_mask]        # (n_events, n_freqs)
        masked_spec_med = np.median(masked_spec, axis=1)       # (n_events,)

        spec_ratio = masked_spec_med / ref_spec
        ratio_list.append(spec_ratio)
        spectrum_list.append(masked_spec_med)

    return np.asarray(ratio_list), np.asarray(spectrum_list)


def find_amplitude_ratio_in_band_specific_bkg(freqs, norm_spec_arr, upward_channels, downward_channels, **bandconfig):
    '''Find normalized amplitude ratio of upward vs downward channels in specific frequency bands. Backgrouns were defined using wiki page: https://radio.uchicago.edu/wiki/index.php/Features_observed_in_data\n
    bandconfig should be a dict with keys as band names and values as dicts with keys: freq_min, freq_max, reference_channels. Example: {"galactic_excess": {"freq_min": 80*units.MHz, "freq_max": 120*units.MHz, "reference_channels": reference_channels_galaxy}, ...}'''

    ratio_arr_dict = {}
    for band_name, band_info in bandconfig.items():
        freq_min = band_info["freq_min"]
        freq_max = band_info["freq_max"]
        reference_channels = band_info["reference_channels"]

        ratio_arr, _ = find_amplitude_ratio_in_band(freqs, norm_spec_arr, upward_channels, downward_channels, reference_channels, freq_min, freq_max)
        ratio_arr_dict[band_name] = ratio_arr

    return ratio_arr_dict
    
def excess_info_from_ratio(ratio_arr, band_name, alpha, ci_thresholds):
    '''Calculate excess information from amplitude ratios in frequency bands.'''

    log_ratio = np.log10(np.asarray(ratio_arr))
    median_log_ratio = np.median(log_ratio)
    mean_log_ratio = np.mean(log_ratio)
    frac_pos_to_neg = np.mean(log_ratio > 0)/np.mean(log_ratio < 0) if np.mean(log_ratio < 0) != 0 else np.inf

    k = np.sum(log_ratio > 0)
    n = int(log_ratio.size)
    result = binomtest(k, n, p=0.5, alternative="greater")
    pval = result.pvalue
    statistic = result.statistic
    confidence_interval = result.proportion_ci(confidence_level=0.99)

    if pval > alpha:
        validation = "NO EXCESS"
    else:
        if confidence_interval.low > ci_thresholds[1]:
            validation = f"STRONG EXCESS in {band_name}"
        elif confidence_interval.low > ci_thresholds[0]:
            validation = f"MODERATE EXCESS in {band_name}"
        else: 
            validation = f"WEAK EXCESS in {band_name}"       

    return {
        "median_log_ratio": median_log_ratio,
        "mean_log_ratio": mean_log_ratio,
        "99% CI": confidence_interval,
        "statistic - k over n": statistic,
        "frac_pos_to_neg": frac_pos_to_neg,
        "pval": pval,
        "validation": validation
    }

def excess_info_from_ratio_specific_bkg(ratio_arr_dict, alpha=0.005, ci_thresholds=(0.6, 0.75)):
    '''Calculate excess information from amplitude ratios in specific frequency bands.'''
    excess_info_dict = {}
    for band_name, ratio_arr in ratio_arr_dict.items():
        excess_info = excess_info_from_ratio(ratio_arr, band_name, alpha, ci_thresholds)
        excess_info_dict[band_name] = excess_info

    return excess_info_dict

def validate_excess_in_bands(excess_info_dict):
    '''Validate excess in different frequency bands based on excess information.'''
    validation_dict = {}
    for band_name, excess_info in excess_info_dict.items():
        validation_dict[band_name] = excess_info["validation"]

    return validation_dict