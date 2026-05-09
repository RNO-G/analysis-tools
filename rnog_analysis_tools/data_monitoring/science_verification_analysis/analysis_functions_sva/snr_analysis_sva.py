import numpy as np
import logging
from scipy.stats import skew
import os
import json

logger = logging.getLogger(__name__)


def calculate_statistics_log_snr(snr_arr):
    '''Calculate log10 statistics (mean, median, std, mean-median) for SNR values.'''
    log_snr_arr = np.zeros((len(snr_arr), len(snr_arr[0])))
    for ch in range(len(snr_arr)):
        snr_arr_ch = snr_arr[ch]
        log_snr_arr_ch = np.log10(snr_arr_ch)
        log_snr_arr[ch, :] = log_snr_arr_ch
    log_mean_dict = {}
    log_median_dict = {}
    log_std_dict = {}
    log_difference_dict = {}

    for ch in range(len(log_snr_arr)):
        log_mean_dict[ch] = np.mean(log_snr_arr[ch])
        log_median_dict[ch] = np.median(log_snr_arr[ch])
        log_std_dict[ch] = np.std(log_snr_arr[ch])
        log_difference_dict[ch] = np.mean(log_snr_arr[ch]) - np.median(log_snr_arr[ch])

    return log_snr_arr, log_mean_dict, log_median_dict, log_std_dict, log_difference_dict

def calculate_z_score_snr(snr_arr, ref_mean_dict, ref_std_dict, channel_list):
    '''Calculate the z-score for SNR values given mean and standard deviation lists for each channel.'''
    z_score_arr = np.zeros((len(snr_arr), len(snr_arr[0])))
    for ch in channel_list:
        snr_arr_ch = snr_arr[ch]
        mean_ch = ref_mean_dict[ch]
        std_ch = ref_std_dict[ch]
        z_score_arr_ch = (snr_arr_ch - mean_ch) / std_ch
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

def load_values_json(script_dir, filename):
    ''' Load k-values from a JSON file.'''
    if not os.path.isabs(filename):
        filepath = os.path.join(script_dir, filename)
    else:
        filepath = filename
    with open(filepath, "r") as f:
        data = json.load(f)     
    values = {int(ch): float(k) for ch, k in data.items()}

    return values

def outlier_flag(z_score_log, k_values_log, channel_list):
    '''Flag outlier events based on the k-values for each channel.'''
    flag = np.zeros((len(channel_list), len(z_score_log[0])), dtype=bool)
    for ch in channel_list:
        flag[ch, :] = np.abs(z_score_log[ch]) > k_values_log[ch]

    return flag

def find_outlier_details(z_score_log, k_values_log, flag, event_info, channel_list):
    '''Find details of outlier events for each channel.'''
    outlier_details = {}

    for ch in channel_list:
        outlier_indices = np.where(flag[ch, :])[0]
        details_ch = []
        for idx in outlier_indices:
            z_abs = np.abs(z_score_log[ch, idx])
            k_ch = k_values_log[ch]
            delta = z_abs - k_ch

            details_ch.append({
                "run": int(event_info["run"][idx]),
                "eventNumber": int(event_info["eventNumber"][idx]),
                "z_abs": float(z_abs),
                "k": float(k_ch),
                "z_minus_k": float(delta),
            })

        outlier_details[ch] = details_ch

    return outlier_details

