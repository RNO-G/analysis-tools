'''This module can be used to find expected RMS (or Vrms) values for RNO-G stations from a known stable time period.'''
import logging
import os
import numpy as np
from argparse import ArgumentParser
import json
import sys
import pandas as pd

SCRIPT_DIR_REF = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR_REF)
sys.path.insert(0, PARENT_DIR)

logger = logging.getLogger(__name__)

from analysis_functions_sva.z_score_analysis_sva import outlier_details, calculate_z_score_parameter, find_k_value, save_values_json, outlier_flag, find_outlier_details, calculate_z_score_rolling, metadata_dict, calculate_expected_values_per_trigger
from plotting_functions_sva.plotting_sva_vrms import plot_vrms_values_against_time_single_trigger_zscore, plot_rolling_mean_std, plot_rolling_mean_linregress, create_heatmap_plot
from config_files_sva.config_station import get_station_config, sampling_rate
import science_verification_analysis as sva
from monitoring_data_functions_sva.get_monitoring_data_uproot import read_multiple_runs, choose_trigger_type_header
from analysis_functions_sva.vrms_analysis_sva import get_rms_per_trigger_monitoring, calculate_vrms
from analysis_functions_sva.vrms_stability_analysis_sva import get_rms_per_run, relative_median_shift, linregress_rolling_mean, write_linregress_results, decision_metric

def setup_logging(station_id, run_label):

    log_file = os.path.join(LOGS_DIR_REF, f"logging_science_verification_analysis_station{station_id}_{run_label}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )

    logger.info(f"Logging to {log_file}")

if __name__ == "__main__":

    argparser = ArgumentParser(description="RNO-G Science Verification Analysis - Expected SNR values.")
    
    argparser.add_argument("-m", "--method", type=str, default="monitoring", choices=["monitoring", "dataProviderRNOG"], required=True, help= "Method to read data, should be either 'monitoring' for reading monitoring data from monitoring.root with uproot or 'dataProviderRNOG' for reading with the data provider "
    "(default: monitoring, which is faster and goes through all events (can be only used for data later than 2026!), dataProviderRNOG only reads events stored in combined.root found in /inbox/), e.g. --method monitoring or --method dataProviderRNOG")
    
    argparser.add_argument("-st", "--station_id", type=int, required=True, help="Station to analyze, e.g --station_id 14")
    argparser.add_argument("-b", "--backend", type=str, default="pyroot", help="!!! Only needed for method 'monitoring' !!!. Backend to use for reading data, should be either pyroot or uproot (default: pyroot), e.g. --backend pyroot or --backend uproot")
    argparser.add_argument("-ex", "--exclude-runs", nargs="+", type=int, default=[], metavar="RUN", help="Run number(s) to exclude, e.g. --exclude-runs 1005 1010")
    argparser.add_argument("--sampling_rate", type=str, default= "after_2024", choices=["before_2024", "after_2024"], help="!!! Only needed for method 'monitoring' !!!. Sampling rate to use, choices are 'before_2024' (3.2 GHz) and 'after_2024' (2.4 GHz), default is 'after_2024'.")
    argparser.add_argument("--save-values", action="store_true", help="Whether to save the calculated reference values as JSON files in the script directory, e.g. --save-values")
    
    run_selection = argparser.add_mutually_exclusive_group(required=True)
    run_selection.add_argument("--runs", nargs="+", type=int, metavar="RUN_NUMBERS",
                           help="Run number(s) to analyze. Each run number should be given explicitly separated by a space, e.g. --runs 1001 1002 1005")
    run_selection.add_argument("--run_range", nargs=2, type=int, metavar=("START_RUN", "END_RUN"),
                            help="Range of run numbers to analyze (inclusive). Provide start and end run numbers separated by a space, e.g. --run_range 1000 1050")
    run_selection.add_argument("--time_range", nargs=2, type=str, metavar=("START_DATE", "END_DATE"),
                            help="Date range to analyze (inclusive). Provide start and end dates separated by a space in YYYY-MM-DD format, e.g. --time_range 2024-07-15 2024-09-30")

    args = argparser.parse_args()

    base_data_path = "/pnfs/ifh.de/acs/radio/diskonly/data/inbox/"

    use_monitoring = True # Default is monitoring data, will be set based on the method argument
    method = args.method

    if method == "dataProviderRNOG":
        use_monitoring = False
        parameter_label = "vrms"
        logger.info("Using dataProviderRNOG method to read data")
    else:
        parameter_label = "rms"
        logger.info("Using monitoring method to read data")

    station_id = args.station_id
    backend = args.backend
    if backend not in ["pyroot", "uproot"]:
        raise ValueError("Backend should be either 'pyroot' or 'uproot'")
    
    sampling_rate_choice = args.sampling_rate
    sr = sampling_rate[sampling_rate_choice]
   
    if args.runs:
        run_numbers = args.runs
    elif args.run_range:
        run_numbers = list(range(args.run_range[0], args.run_range[1] + 1))
    elif args.time_range:
        start_time, stop_time = args.time_range
        runtable = sva.read_rnog_runtable(station_id, start_time, stop_time)
        run_numbers = runtable["run"].tolist()
    else:
        raise ValueError("No run selection provided")
    
    # Exclude specified runs
    if args.exclude_runs:
        exclude_set = set(args.exclude_runs)
        run_numbers = [r for r in run_numbers if r not in exclude_set]

    run_numbers = sorted(run_numbers)
    first_run = run_numbers[0]
    last_run = run_numbers[-1]

    if first_run == last_run:
        run_label = f"run_{first_run}"
    else:
        run_label = f"runs_{first_run}_{last_run}"

    EXPECTED_VALUES_DIR_REF = os.path.join(SCRIPT_DIR_REF, f"expected_{parameter_label}")
    PLOTS_DIR_REF = os.path.join(SCRIPT_DIR_REF, f"plots_reference/{parameter_label}")
    LOGS_DIR_REF = os.path.join(SCRIPT_DIR_REF, f"logs_reference/{parameter_label}")
    RESULTS_DIR_REF = os.path.join(SCRIPT_DIR_REF, f"results_reference/{parameter_label}")

    os.makedirs(PLOTS_DIR_REF, exist_ok=True)
    os.makedirs(LOGS_DIR_REF, exist_ok=True)
    os.makedirs(RESULTS_DIR_REF, exist_ok=True)
    os.makedirs(EXPECTED_VALUES_DIR_REF, exist_ok=True)

    setup_logging(station_id, run_label)

    # Get channel lists from config
    config = get_station_config(station_id)
    all_channels = config["all_channels"]
    
    if use_monitoring == False:
        spec_arr, trace_arr, times_trace_arr, snr_arr, run_no, times, freqs, event_info, glitch_arr, block_offsets_arr = sva.read_rnog_data(station_id, run_numbers, backend=backend, sampling_rate=sr) 
        
        force_mask = sva.choose_trigger_type(event_info, "FORCE")
        lt_mask = sva.choose_trigger_type(event_info, "LT")
        radiant0_mask = sva.choose_trigger_type(event_info, "RADIANT0")
        radiant1_mask = sva.choose_trigger_type(event_info, "RADIANT1")

        times = np.array(times)
        run_no_force = event_info["run"][force_mask]
        event_number_force = event_info["eventNumber"][force_mask]

        run_no_radiant0 = event_info["run"][radiant0_mask]
        event_number_radiant0 = event_info["eventNumber"][radiant0_mask]

        run_no_radiant1 = event_info["run"][radiant1_mask]
        event_number_radiant1 = event_info["eventNumber"][radiant1_mask]

        run_no_lt = event_info["run"][lt_mask]
        event_number_lt = event_info["eventNumber"][lt_mask]

        vrms_arr, vrms_arr_force, vrms_arr_radiant0, vrms_arr_radiant1, vrms_arr_lt = calculate_vrms(trace_arr, event_info)

    else:
        combined_event_info = read_multiple_runs(base_path = base_data_path, station_id = station_id, run_numbers=run_numbers)
        rms_arr = combined_event_info["rms_arr"]
        trigger_type_arr = combined_event_info["triggerType"]
        times = combined_event_info["trigger_time_utc"]
        run_no = combined_event_info["run_no"]
        event_number_arr = combined_event_info["event_number_arr"]
        failed_run_info = combined_event_info["failed_run_info"] # dict with run number as key and value as reason for failure, only for runs that failed to be read
        failed_run_info = combined_event_info["failed_run_info"] or {}
        failed_runs = list(failed_run_info.keys())

        valid_times_mask = ~pd.isna(times)
        if np.any(~valid_times_mask):
            invalid_runs = np.unique(run_no[~valid_times_mask])
            logger.warning(f"Found {np.sum(~valid_times_mask)} invalid timestamps in runs {invalid_runs}. These events will be skipped in the analysis.")
            times = times[valid_times_mask]
            rms_arr = rms_arr[:, valid_times_mask]
            trigger_type_arr = trigger_type_arr[valid_times_mask]
            run_no = run_no[valid_times_mask]
            event_number_arr = event_number_arr[valid_times_mask]

            for invalid_run in invalid_runs:
                failed_run_info[invalid_run] = "Some events have been skipped in the analysis due to invalid timestamps, check logs for details"
        
        force_mask = choose_trigger_type_header(trigger_type_arr, "FORCE")
        lt_mask = choose_trigger_type_header(trigger_type_arr, "LT")
        radiant0_mask = choose_trigger_type_header(trigger_type_arr, "RADIANT0")
        radiant1_mask = choose_trigger_type_header(trigger_type_arr, "RADIANT1")
   
        if failed_run_info:
            sva.write_failed_runs_to_csv(station_id, failed_run_info, run_label, results_dir=RESULTS_DIR_REF)

        run_no_force = run_no[force_mask]
        event_number_force = event_number_arr[force_mask]

        run_no_radiant0 = run_no[radiant0_mask]
        event_number_radiant0 = event_number_arr[radiant0_mask]

        run_no_radiant1 = run_no[radiant1_mask]
        event_number_radiant1 = event_number_arr[radiant1_mask]

        run_no_lt = run_no[lt_mask]
        event_number_lt = event_number_arr[lt_mask]

        vrms_arr,vrms_arr_force, vrms_arr_radiant0, vrms_arr_radiant1, vrms_arr_lt = get_rms_per_trigger_monitoring(rms_arr=rms_arr, force_mask=force_mask, lt_mask=lt_mask, radiant0_mask=radiant0_mask, radiant1_mask=radiant1_mask)

    excluded_runs = args.exclude_runs if args.exclude_runs else []
    excluded_runs.extend(failed_runs)
    
    logger.info(f"Start calculating expected values for {parameter_label} parameter...") 
    times_force = times[force_mask]
    times_radiant0 = times[radiant0_mask]
    times_radiant1 = times[radiant1_mask]
    times_lt = times[lt_mask]

    z_score_force, z_score_rolling_force, k_values_force, vrms_mean_force, vrms_std_force, metadata_force, rolling_mean_force, rolling_std_force = calculate_expected_values_per_trigger(station_id, first_run, last_run, vrms_arr_force, times_force, trigger_type="FORCE", excluded_runs=excluded_runs, run_no=run_no, all_channels=all_channels)
    z_score_radiant0, z_score_rolling_radiant0, k_values_radiant0, vrms_mean_radiant0, vrms_std_radiant0, metadata_radiant0, rolling_mean_radiant0, rolling_std_radiant0 = calculate_expected_values_per_trigger(station_id, first_run, last_run, vrms_arr_radiant0, times_radiant0, trigger_type="RADIANT0", excluded_runs=excluded_runs, run_no=run_no, all_channels=all_channels)
    z_score_radiant1, z_score_rolling_radiant1, k_values_radiant1, vrms_mean_radiant1, vrms_std_radiant1, metadata_radiant1, rolling_mean_radiant1, rolling_std_radiant1 = calculate_expected_values_per_trigger(station_id, first_run, last_run, vrms_arr_radiant1, times_radiant1, trigger_type="RADIANT1", excluded_runs=excluded_runs, run_no=run_no, all_channels=all_channels)
    z_score_lt, z_score_rolling_lt, k_values_lt, vrms_mean_lt, vrms_std_lt, metadata_lt, rolling_mean_lt, rolling_std_lt = calculate_expected_values_per_trigger(station_id, first_run, last_run, vrms_arr_lt, times_lt, trigger_type="LT", excluded_runs=excluded_runs, run_no=run_no, all_channels=all_channels)

    if args.save_values:
        save_values_json(k_values_force, vrms_mean_force, vrms_std_force, filename=f"expected_{parameter_label}_station{station_id}.json", SCRIPT_DIR=EXPECTED_VALUES_DIR_REF, metadata=metadata_force)
        # save_values_json(k_values_radiant0, vrms_mean_radiant0, vrms_std_radiant0, filename=f"expected_{parameter_label}_radiant0_station{station_id}.json", SCRIPT_DIR=EXPECTED_VALUES_DIR_REF, metadata=metadata_radiant0)
        # save_values_json(k_values_radiant1, vrms_mean_radiant1, vrms_std_radiant1, filename=f"expected_{parameter_label}_radiant1_station{station_id}.json", SCRIPT_DIR=EXPECTED_VALUES_DIR_REF, metadata=metadata_radiant1)
        # save_values_json(k_values_lt, vrms_mean_lt, vrms_std_lt, filename=f"expected_{parameter_label}_lt_station{station_id}.json", SCRIPT_DIR=EXPECTED_VALUES_DIR_REF, metadata=metadata_lt)
 
    flag_outliers_force, outlier_details_force = outlier_details(z_score_force, k_values_force, all_channels, run_no_force, event_number_force, trigger_label="FORCE")
    flag_outliers_radiant0, outlier_details_radiant0 = outlier_details(z_score_radiant0, k_values_radiant0, all_channels, run_no_radiant0, event_number_radiant0, trigger_label="RADIANT0")
    flag_outliers_radiant1, outlier_details_radiant1 = outlier_details(z_score_radiant1, k_values_radiant1, all_channels, run_no_radiant1, event_number_radiant1, trigger_label="RADIANT1")
    flag_outliers_lt, outlier_details_lt = outlier_details(z_score_lt, k_values_lt, all_channels, run_no_lt, event_number_lt, trigger_label="LT")

    k_values_rolling = {int(ch): 4 for ch in all_channels} # Placeholder, as k-values for rolling z-score are not calculated in this script, but could be implemented in the future if needed
    flag_outliers_force_rolling, outlier_details_force_rolling = outlier_details(z_score_rolling_force, k_values_rolling, all_channels, run_no_force, event_number_force, trigger_label="FORCE_rolling")

    sva.write_vrms_outlier_details(outlier_details_force, station_id, run_label, trigger_label="FORCE", n_events = len(times_force), results_dir=RESULTS_DIR_REF)
    sva.write_vrms_outlier_details(outlier_details_radiant0, station_id, run_label, trigger_label="RADIANT0", n_events = len(times_radiant0), results_dir=RESULTS_DIR_REF)
    sva.write_vrms_outlier_details(outlier_details_radiant1, station_id, run_label, trigger_label="RADIANT1", n_events = len(times_radiant1), results_dir=RESULTS_DIR_REF)  
    sva.write_vrms_outlier_details(outlier_details_lt, station_id, run_label, trigger_label="LT", n_events = len(times_lt), results_dir=RESULTS_DIR_REF)
    sva.write_vrms_outlier_details(outlier_details_force_rolling, station_id, run_label, trigger_label="FORCE_rolling", n_events = len(times_force), results_dir=RESULTS_DIR_REF)

    plot_vrms_values_against_time_single_trigger_zscore(times_force, vrms_arr_force, flag_outliers_force, z_score_force, k_values_force, trigger_name="FORCE", channel_list=all_channels, station_id=station_id, run_label=run_label, save_location=PLOTS_DIR_REF, use_monitoring=True)
    plot_vrms_values_against_time_single_trigger_zscore(times_radiant0, vrms_arr_radiant0, flag_outliers_radiant0, z_score_radiant0, k_values_radiant0, trigger_name="RADIANT0", channel_list=all_channels, station_id=station_id, run_label=run_label, save_location=PLOTS_DIR_REF, use_monitoring=True)
    plot_vrms_values_against_time_single_trigger_zscore(times_radiant1, vrms_arr_radiant1, flag_outliers_radiant1, z_score_radiant1, k_values_radiant1, trigger_name="RADIANT1", channel_list=all_channels, station_id=station_id, run_label=run_label, save_location=PLOTS_DIR_REF, use_monitoring=True)
    plot_vrms_values_against_time_single_trigger_zscore(times_lt, vrms_arr_lt, flag_outliers_lt, z_score_lt, k_values_lt, trigger_name="LT", channel_list=all_channels, station_id=station_id, run_label=run_label, save_location=PLOTS_DIR_REF, use_monitoring=True)
    plot_vrms_values_against_time_single_trigger_zscore(times_force, vrms_arr_force, flag_outliers_force_rolling, z_score_rolling_force, k_values_rolling, trigger_name="FORCE_rolling", channel_list=all_channels, station_id=station_id, run_label=run_label, save_location=PLOTS_DIR_REF, use_monitoring=True)
    
    plot_rolling_mean_std(times_force, rolling_mean_force, rolling_std_force, channel_list=all_channels, station_id = station_id, run_label=run_label, trigger_name="FORCE", save_location= PLOTS_DIR_REF, use_monitoring=True)
    slope_dict_force, intercept_dict_force, r_value_dict_force, p_value_dict_force, std_err_dict_force, intercept_std_err_dict_force = linregress_rolling_mean(times_force, rolling_mean_force, all_channels)
    plot_rolling_mean_linregress(times_force, rolling_mean_force, all_channels, slope_dict_force, intercept_dict_force, station_id, run_label, trigger_name="FORCE", save_location=PLOTS_DIR_REF, use_monitoring=True)
    write_linregress_results(slope_dict_force, intercept_dict_force, r_value_dict_force, p_value_dict_force, std_err_dict_force, intercept_std_err_dict_force, station_id, run_label, trigger_name="FORCE", results_dir=RESULTS_DIR_REF)
    rms_arr_per_run_dict_force = get_rms_per_run(vrms_arr_force, run_no_force)
    
    relative_median_shift_results = relative_median_shift(rms_arr_per_run_dict_force, all_channels)

    with open(os.path.join(RESULTS_DIR_REF, f"rms_relative_median_shift_results_force_trigger_station{station_id}_{run_label}.json"), "w") as f:
        json.dump(relative_median_shift_results, f, indent=4)

    create_heatmap_plot(relative_median_shift_results, label = "Relative Median Shift", save_dir = PLOTS_DIR_REF, channel_list=all_channels, station_id = station_id,matrix_key = "median_shift_matrix", run_label=run_label, cmap="Reds")

    rms_results = decision_metric(outlier_details_force, relative_median_shift_results, n_events_force=len(times_force), channels=all_channels)
    with open(os.path.join(RESULTS_DIR_REF, f"rms_stability_decision_results_force_trigger_station{station_id}_{run_label}.json"), "w") as f:
        json.dump(rms_results, f, indent=4)
