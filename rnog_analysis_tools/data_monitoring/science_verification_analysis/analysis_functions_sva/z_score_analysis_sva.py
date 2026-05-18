''' Z-score analysis for SNR and RMS values for RNO-G Science Verification Analysis. '''

import numpy as np
import logging
from scipy.stats import skew
import os
import json

logger = logging.getLogger(__name__)

def calculate_statistics_log_paramater(parameter_arr):
    '''Calculate log10 statistics (mean, median, std, mean-median) for a given parameter array (e.g. SNR or RMS values).'''
    log_parameter_arr = np.zeros((len(parameter_arr), len(parameter_arr[0])))
    for ch in range(len(parameter_arr)):
        parameter_arr_ch = parameter_arr[ch]
        log_parameter_arr_ch = np.log10(parameter_arr_ch)
        log_parameter_arr[ch, :] = log_parameter_arr_ch
    
    log_mean_dict = {}
    log_median_dict = {}
    log_std_dict = {}
    log_difference_dict = {}

    for ch in range(len(log_parameter_arr)):
        log_mean_dict[ch] = np.mean(log_parameter_arr[ch])
        log_median_dict[ch] = np.median(log_parameter_arr[ch])
        log_std_dict[ch] = np.std(log_parameter_arr[ch])
        log_difference_dict[ch] = np.mean(log_parameter_arr[ch]) - np.median(log_parameter_arr[ch])

    return log_parameter_arr, log_mean_dict, log_median_dict, log_std_dict, log_difference_dict

def calculate_z_score_parameter(parameter_arr, ref_mean_dict, ref_std_dict, channel_list):
    '''Calculate the z-score for a given parameter array (e.g. SNR or RMS values) given mean and standard deviation lists for each channel.'''
    z_score_arr = np.zeros((len(parameter_arr), len(parameter_arr[0])))
    for ch in channel_list:
        parameter_arr_ch = parameter_arr[ch]
        mean_ch = ref_mean_dict[ch]
        std_ch = ref_std_dict[ch]
        z_score_arr_ch = (parameter_arr_ch - mean_ch) / std_ch
        z_score_arr[ch,:] = z_score_arr_ch

    return z_score_arr

def symmetry_metrics_channel_z_score(z_score):
    return {
        "mean": np.mean(z_score),
        "median": np.median(z_score),
        "skew": skew(z_score, bias=False),
        "p_pos_3": np.mean(z_score > 3),
        "p_neg_3": np.mean(z_score < -3),
        "p_pos_5": np.mean(z_score > 5),
        "p_neg_5": np.mean(z_score < -5),
    }

def symmetry_metrics_z_score(z_score):
    metrics = {}
    for ch in range(len(z_score)):
        z_score_ch = z_score[ch]
        metrics[f"ch_{ch}"] = symmetry_metrics_channel_z_score(z_score_ch)
    return metrics

#### For finding expected values and k-values for SNR and RMS monitoring
def find_k_value(z_score_log, channel_list, quantile=0.999):
    '''Find the k-value corresponding to the given reference z-score array and quantile.'''
    k_ch_list = {}
    for ch in channel_list:
        z_score_ch = z_score_log[ch]
        k_ch = np.quantile(np.abs(z_score_ch), quantile)
        k_ch_list[ch] = k_ch
    return k_ch_list

def save_values_json(k_values_log, mean_values_log, std_values_log, filename, SCRIPT_DIR, metadata=None):
    '''Save the reference values as a JSON file.'''
    if not os.path.isabs(filename):
        filepath = os.path.join(SCRIPT_DIR, filename)
    else:
        filepath = filename

    # Add metadata
    output = {
        "metadata": metadata if metadata is not None else {},
        "values": {
            ch: {
                "k_value": float(k_values_log[ch]),
                "mean": float(mean_values_log[ch]),
                "std": float(std_values_log[ch]),
            } for ch in k_values_log
        }
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Expected values saved to {filepath}.")

#### For loading expected values from JSON files
def load_values_json(script_dir, filename):
    ''' Load reference values from a JSON file.'''
    if not os.path.isabs(filename):
        filepath = os.path.join(script_dir, filename)
    else:
        filepath = filename
    with open(filepath, "r") as f:
        data = json.load(f)     
    
    k_values = {int(ch): float(info["k_value"]) for ch, info in data["values"].items()}
    mean_values = {int(ch): float(info["mean"]) for ch, info in data["values"].items()}
    std_values = {int(ch): float(info["std"]) for ch, info in data["values"].items()}

    # Report metadata if available
    if "metadata" in data:
        logger.info("Metadata for loaded values:")
        logger.info(f"Reference values for the analysis were calculated using the following metadata:\n")
        for key, value in data["metadata"].items():     
            logger.info(f"- {key}: {value}")
    else:
        logger.info("No metadata found in the loaded reference JSON file.")

    return k_values, mean_values, std_values

def outlier_flag(z_score_log, k_values_log, channel_list):
    '''Flag outlier events based on the k-values for each channel.'''
    flag = np.zeros((len(channel_list), len(z_score_log[0])), dtype=bool)
    for ch in channel_list:
        flag[ch, :] = np.abs(z_score_log[ch]) > k_values_log[ch]

    return flag

def find_outlier_details(z_score_log, k_values_log, flag, channel_list, run_no, event_number):
    '''Find details of outlier events for each channel.'''
    outlier_details = {}

    # Sanity check:
    if flag.shape[1] != len(run_no) or flag.shape[1] != len(event_number):
        raise ValueError("Length of run_no and event_number arrays must match the number of events in the z_score_log and flag arrays.")

    for ch in channel_list:
        outlier_indices = np.where(flag[ch, :])[0]
        details_ch = []
        for idx in outlier_indices:
            z_abs = np.abs(z_score_log[ch, idx])
            k_ch = k_values_log[ch]
            delta = z_abs - k_ch
            
            details_ch.append({
                "run": int(run_no[idx]),
                "eventNumber": int(event_number[idx]),
                "z_abs": float(z_abs),
                "k": float(k_ch),
                "z_minus_k": float(delta),
            })

        outlier_details[ch] = details_ch

    return outlier_details




