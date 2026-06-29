import numpy as np
import logging
from scipy.stats import mannwhitneyu as mw_test
import os
from scipy.stats import wasserstein_distance, linregress

logger = logging.getLogger(__name__)

def get_rms_per_run(rms_arr, run_no):
    '''Get RMS values for each channel and run number.'''
    run_no_unique = np.unique(run_no)
    rms_arr_per_run_dict = {}
    for run in run_no_unique:
        run_mask = run_no == run
        rms_arr_per_run_dict[run] = rms_arr[:, run_mask]
    return rms_arr_per_run_dict

def relative_median_shift(rms_arr_per_run_dict, channel_list):
    '''Calculate relative median shift of RMS distributions across runs for each channel.'''
    run_nos = sorted(rms_arr_per_run_dict.keys())
    median_shift_results = {}

    for ch in channel_list:
        medians = [np.median(rms_arr_per_run_dict[run][ch]) for run in run_nos]
        median_shift_matrix = np.full((len(run_nos), len(run_nos)), np.nan)
        np.fill_diagonal(median_shift_matrix, 0)
        for i in range(len(run_nos)):
            for j in range(i+1,len(run_nos)):
                shift = 2*(medians[j] - medians[i]) / (medians[i] + medians[j]) if (medians[i] + medians[j]) != 0 else np.nan
                median_shift_matrix[i, j] = shift
                median_shift_matrix[j, i] = -shift
        median_shift_results[int(ch)] = {"medians": [float(m) for m in medians], "median_shift_matrix": median_shift_matrix.tolist()}
    return median_shift_results

def wasserstein_distance_per_run(rms_arr_per_run_dict, channel_list):
    '''Calculate Wasserstein distance between RMS distributions of different runs for each channel.'''
    run_nos = sorted(rms_arr_per_run_dict.keys())
    wasserstein_results = {}

    for ch in channel_list:
        distance_matrix = np.full((len(run_nos), len(run_nos)), np.nan)
        np.fill_diagonal(distance_matrix, 0)
        for i in range(len(run_nos)):
            for j in range(i+1, len(run_nos)):
                dist = wasserstein_distance(rms_arr_per_run_dict[run_nos[i]][ch], rms_arr_per_run_dict[run_nos[j]][ch])
                distance_matrix[i, j] = float(dist)
                distance_matrix[j, i] = float(dist)
        wasserstein_results[int(ch)] = {"distance_matrix": distance_matrix.tolist()}

    return wasserstein_results

def wasserstein_statistics(wasserstein_results, channel_list):
    '''Calculate statistics of Wasserstein distances for each channel.'''
    wasserstein_stats = {}
    for ch in channel_list:
        distance_matrix = np.array(wasserstein_results[int(ch)]["distance_matrix"])
        overall_max_distance = np.nanmax(distance_matrix)
        diagonal_matrix = np.diag(distance_matrix, k=1)
        
        median_global_distance = np.nanmedian(distance_matrix)
        median_diagonal_distance = np.nanmedian(diagonal_matrix)
        mean_global_distance = np.nanmean(distance_matrix)
        mean_diagonal_distance = np.nanmean(diagonal_matrix)
        max_global_distance = np.nanmax(distance_matrix)
        max_diagonal_distance = np.nanmax(diagonal_matrix)
        std_global_distance = np.nanstd(distance_matrix)
        std_diagonal_distance = np.nanstd(diagonal_matrix)
        roughness = std_global_distance / mean_global_distance if mean_global_distance != 0 else np.nan
        wasserstein_stats[int(ch)] = {
            "overall_max_distance": float(overall_max_distance),
            "median_global_distance": float(median_global_distance),
            "median_diagonal_distance": float(median_diagonal_distance),
            "mean_global_distance": float(mean_global_distance),
            "mean_diagonal_distance": float(mean_diagonal_distance),
            "max_global_distance": float(max_global_distance),
            "max_diagonal_distance": float(max_diagonal_distance),
            "std_global_distance": float(std_global_distance),
            "std_diagonal_distance": float(std_diagonal_distance),
            "max_ratio": float(max_global_distance / max_diagonal_distance) if max_diagonal_distance != 0 else np.nan,
            "roughness": float(roughness),
            "fluctuation_index": float(std_global_distance / mean_global_distance) if mean_global_distance != 0 else np.nan ,
            "jump_ratio": float(max_diagonal_distance / median_diagonal_distance) if median_diagonal_distance != 0 else np.nan,
            "global_p90": float(np.nanpercentile(distance_matrix, 90)),
            "diagonal_p90": float(np.nanpercentile(diagonal_matrix, 90)),
        }
    return wasserstein_stats

def linregress_rolling_mean(times, rolling_mean, channel_list):
    '''Perform linear regression on the rolling mean values over time for each channel.'''
    slope_dict = {}
    intercept_dict = {}
    r_value_dict = {}
    p_value_dict = {}
    std_err_dict = {}
    intercept_std_err_dict = {}

    times = times.astype("datetime64[s]").astype(np.float64) # Convert to seconds since epoch for linregress
    times_rel = times - times.min() # Use relative time to avoid numerical issues with large values
    times_rel_hours = times_rel / 3600 # Convert to hours for better interpretability of slope

    for ch in channel_list:
        res = linregress(times_rel_hours[1:], rolling_mean[ch][1:])
        slope_dict[ch] = res.slope
        intercept_dict[ch] = res.intercept
        r_value_dict[ch] = res.rvalue
        p_value_dict[ch] = res.pvalue
        std_err_dict[ch] = res.stderr
        intercept_std_err_dict[ch] = res.intercept_stderr

    return slope_dict, intercept_dict, r_value_dict, p_value_dict, std_err_dict, intercept_std_err_dict

def write_linregress_results(slope_dict, intercept_dict, r_value_dict, p_value_dict, std_err_dict, intercept_std_err_dict, station_id, run_label, trigger_name, results_dir):
    '''Write linear regression results to a txt file.'''   
    results_file = os.path.join(results_dir, f"linear_regression_results_rolling_mean_{trigger_name}_station{station_id}_{run_label}.txt")
    with open(results_file, "w") as f:
        f.write(f"Linear Regression Results for Rolling Mean of Vrms - Station {station_id}, Trigger {trigger_name}, Runs {run_label}\n")
        for ch in slope_dict.keys():
            f.write(f"Channel {ch}:\n")
            f.write(f"  Slope: {slope_dict[ch]} ± {std_err_dict[ch]} ADC/hours \n")
            f.write(f"  Intercept: {intercept_dict[ch]} ± {intercept_std_err_dict[ch]} ADC\n")
            f.write(f"  R-value: {r_value_dict[ch]}\n")
            f.write(f"  R-squared: {r_value_dict[ch]**2}\n")
            f.write(f"  P-value: {p_value_dict[ch]}\n\n")
    logger.info(f"Linear regression results saved to {results_file}")

def decision_metric(outlier_details, relative_median_shift, n_events_force, channels):
    rms_results = {}

    for ch in channels:
        outlier_ch_info = outlier_details.get(ch, [])
        n_out = len(outlier_ch_info)
        frac_out = n_out / n_events_force if n_events_force > 0 else 0.0

        deltas = np.array(
            [abs(o.get("z_minus_k", 0.0)) for o in outlier_ch_info],
            dtype=float
        )

        max_delta = np.nanmax(deltas) if n_out > 0 else 0.0
        n_large_delta = np.sum(deltas >= 5.0) if n_out > 0 else 0
        frac_large_delta = n_large_delta / n_events_force if n_events_force > 0 else 0.0

        median_shift_matrix = np.asarray(relative_median_shift[ch]["median_shift_matrix"])
        abs_shift_matrix = np.abs(median_shift_matrix)

        q95_rel_median_shift = np.nanquantile(abs_shift_matrix, 0.95)
        max_rel_median_shift = np.nanmax(abs_shift_matrix)

        if max_delta >= 10:
            if q95_rel_median_shift >= 0.1 or frac_out >= 0.002 or frac_large_delta >= 0.0002:
                rms_value = "X"
            else:
                rms_value = "!!"

        elif max_delta >= 5:
            if q95_rel_median_shift >= 0.10 or frac_out >= 0.004 or n_large_delta >= 10 or frac_large_delta >= 0.0002:
                rms_value = "X"
            else:
                rms_value = "!!"

        elif max_delta >= 3:
            if q95_rel_median_shift >= 0.10 or frac_out >= 0.01 or frac_large_delta >= 0.0004:
                rms_value = "X"
            elif q95_rel_median_shift >= 0.05 or frac_out >= 0.002 or frac_large_delta >= 0.0002:
                rms_value = "!!"
            else:
                rms_value = "OK"

        else:
            if q95_rel_median_shift >= 0.10 or frac_out >= 0.01 or frac_large_delta >= 0.0004:
                rms_value = "X"
            elif q95_rel_median_shift >= 0.05 or frac_out >= 0.004 or frac_large_delta >= 0.0002:
                rms_value = "!!"
            else:
                rms_value = "OK"

        rms_results[ch] = {
            "decision": rms_value,
            "n_outliers": n_out,
            "frac_outliers": frac_out,
            "max_delta": max_delta,
            "n_large_delta": int(n_large_delta),
            "frac_large_delta": frac_large_delta,
            "q95_rel_median_shift": q95_rel_median_shift,
            "max_rel_median_shift": max_rel_median_shift,
        }

    return rms_results