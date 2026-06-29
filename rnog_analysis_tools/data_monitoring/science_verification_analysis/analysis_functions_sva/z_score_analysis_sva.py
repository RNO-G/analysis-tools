''' Z-score analysis for SNR and RMS values for RNO-G Science Verification Analysis. '''

import numpy as np
import logging
from scipy.stats import skew
import os
import json
import pandas as pd
from astropy.time import Time

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

    # Track capped k-values
    capped_k_values_high = {f"Ch{ch}": float(k_values_log[ch]) for ch in k_values_log if float(k_values_log[ch]) >= 5}
    capped_k_values_low = {f"Ch{ch}": float(k_values_log[ch]) for ch in k_values_log if float(k_values_log[ch]) <= 3}
    
    # Add metadata
    if metadata is None:
        metadata = {}
    else:
        metadata = metadata.copy()
    
    comments = []
    if capped_k_values_high:
        comments.append(f"k values above 5 are set to 4: {capped_k_values_high}")
    if capped_k_values_low:
        comments.append(f"k values below 3 are set to 3: {capped_k_values_low}")
    
    if comments:
        metadata["comment"] = "\n".join(comments)
    
    output = {
        "metadata": metadata,
        "values": {
            ch: {
                "k_value": float(k_values_log[ch]) if 3 < float(k_values_log[ch]) < 5 else (4.0 if float(k_values_log[ch]) > 5 else 3.0),
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
        raise ValueError(f"Length of run_no and event_number arrays must match the number of events in the z_score_log and flag arrays. run_no length: {len(run_no)}, event_number length: {len(event_number)}, z_score_log shape: {z_score_log.shape}, flag shape: {flag.shape}")

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

#### Calculate z score from rolling mean and std
def calculate_z_score_rolling(parameter_arr, run_no, channel_list):
    '''Calculate z-score using a rolling mean and std for each channel.'''
    z_score_arr = np.zeros((len(parameter_arr), len(parameter_arr[0])))
    rolling_mean_arr = np.zeros((len(parameter_arr), len(parameter_arr[0])))
    rolling_std_arr = np.zeros((len(parameter_arr), len(parameter_arr[0])))

    run_no_unique = np.unique(run_no)
    n_runs = len(run_no_unique)

    for ch in channel_list:
        parameter_arr_ch = parameter_arr[ch]
        window_size = int(len(parameter_arr_ch)/n_runs) 
        parameter_series = pd.Series(parameter_arr_ch)
        rolling_mean = parameter_series.rolling(window=window_size, min_periods=1).mean()
        rolling_std = parameter_series.rolling(window=window_size, min_periods=1).std()
        z_score_arr_ch = (parameter_series - rolling_mean) / rolling_std
        z_score_arr[ch, :] = z_score_arr_ch.values
        rolling_mean_arr[ch, :] = rolling_mean.values
        rolling_std_arr[ch, :] = rolling_std.values

    return z_score_arr, rolling_mean_arr, rolling_std_arr

def metadata_dict(station_id, first_run, last_run, times, trigger_type="FORCE", excluded_runs=None, comment=""):
    '''Create a metadata dictionary to save with the expected values, containing station ID, run numbers and time period.'''
    if isinstance(times, Time):
        start_time = times.min().iso
        end_time = times.max().iso
    else:
        start_time = str(np.min(times))
        end_time = str(np.max(times))

    metadata = {
        "station_id": station_id,
        "run_range": f"{first_run} - {last_run}",
        "excluded_runs": excluded_runs if excluded_runs else None,
        "n_events": len(times),
        "trigger_type": trigger_type,
        "start_time": start_time,
        "end_time": end_time,
        "comment": comment
    }
    logger.info(f"Metadata for expected values: {metadata}")
    return metadata

def calculate_expected_values_per_trigger(station_id, first_run, last_run, vrms_arr_trigger, times_trigger, trigger_type, excluded_runs, run_no,all_channels, comment=""):
    '''Calculate expected values for a given trigger type.'''
    vrms_mean = np.mean(vrms_arr_trigger, axis=1)
    vrms_std = np.std(vrms_arr_trigger, axis=1)

    z_score = calculate_z_score_parameter(vrms_arr_trigger, vrms_mean, vrms_std, all_channels)
    z_score_rolling, rolling_mean, rolling_std = calculate_z_score_rolling(vrms_arr_trigger, run_no, all_channels)
    k_values = find_k_value(z_score, all_channels, quantile=0.999)

    metadata = metadata_dict(station_id, first_run, last_run, times_trigger, trigger_type=trigger_type, excluded_runs=excluded_runs, comment=comment)
    
    return z_score, z_score_rolling, k_values, vrms_mean, vrms_std, metadata, rolling_mean, rolling_std

def outlier_details(z_score, k_values, channels, run_no, event_number_arr, trigger_label, max_k=4):
    '''Find details of outlier events.'''
    k_values = k_values.copy()

    for ch in channels:
        if k_values[ch] > max_k:
            logger.warning(f"Calculated k-value for channel {ch} for trigger {trigger_label} is {k_values[ch]:.2f}. Setting it to {max_k}.")
            k_values[ch] = max_k

    flag_outliers = outlier_flag(z_score, k_values, channels)
    outlier_details = find_outlier_details(z_score, k_values, flag_outliers, channels, run_no, event_number_arr)

    return flag_outliers, outlier_details



