'''This module can be used to find expected SNR values for RNO-G stations from a known stable time period.'''
import logging
import os
import numpy as np
from argparse import ArgumentParser
import logging
import sys
from astropy.time import Time
import pandas as pd

SCRIPT_DIR_REF = os.path.dirname(os.path.abspath(__file__))
EXPECTED_VALUES_DIR_REF = os.path.join(SCRIPT_DIR_REF, "expected_snr")
PLOTS_DIR_REF = os.path.join(SCRIPT_DIR_REF, "plots_reference", "snr")
LOGS_DIR_REF = os.path.join(SCRIPT_DIR_REF, "logs_reference", "snr")
RESULTS_DIR_REF = os.path.join(SCRIPT_DIR_REF, "results_reference", "snr")
os.makedirs(PLOTS_DIR_REF, exist_ok=True)
os.makedirs(LOGS_DIR_REF, exist_ok=True)
os.makedirs(RESULTS_DIR_REF, exist_ok=True)
os.makedirs(EXPECTED_VALUES_DIR_REF, exist_ok=True)

PARENT_DIR = os.path.dirname(SCRIPT_DIR_REF)
sys.path.insert(0, PARENT_DIR)

from analysis_functions_sva.z_score_analysis_sva import calculate_statistics_log_paramater, calculate_z_score_parameter, symmetry_metrics_channel_z_score, symmetry_metrics_z_score, find_k_value, save_values_json, load_values_json, outlier_flag, find_outlier_details
from plotting_functions_sva.plotting_sva_snr import choose_day_interval, plot_snr_against_time
from config_files_sva.config_station import get_station_config, sampling_rate
import science_verification_analysis as sva
from monitoring_data_functions_sva.get_monitoring_data_uproot import read_multiple_runs, choose_trigger_type_header

logger = logging.getLogger(__name__)

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

def metadata_dict(station_id, first_run, last_run, times, trigger_type="FORCE", comment=""):
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
        "excluded_runs": args.exclude_runs if args.exclude_runs else None,
        "n_events": len(times),
        "trigger_type": trigger_type,
        "start_time": start_time,
        "end_time": end_time,
        "comment": comment
    }
    logger.info(f"Metadata for expected values: {metadata}")
    return metadata

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
        logger.info("Using dataProviderRNOG method to read data")
    else:
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

    setup_logging(station_id, run_label)

    # Get channel lists from config
    config = get_station_config(station_id)
    all_channels = config["all_channels"]
    
    if use_monitoring == False:
        spec_arr, trace_arr, times_trace_arr, snr_arr, run_no, times, freqs, event_info, glitch_arr, block_offsets_arr = sva.read_rnog_data(station_id, run_numbers, backend=backend, sampling_rate=sr) 
        force_mask = sva.choose_trigger_type(event_info, "FORCE")
        times = np.array(times)
        run_no_force = event_info["run"][force_mask]
        event_number_force = event_info["eventNumber"][force_mask]

    else:
        combined_event_info = read_multiple_runs(base_path = base_data_path, station_id = station_id, run_numbers=run_numbers)
        snr_arr = combined_event_info["snr_arr"] 
        trigger_type_arr = combined_event_info["triggerType"]
        times = combined_event_info["trigger_time_utc"]
        run_no = combined_event_info["run_no"]
        event_number_arr = combined_event_info["event_number_arr"]

        force_mask = choose_trigger_type_header(trigger_type_arr, "FORCE")
        run_no_force = run_no[force_mask]
        event_number_force = event_number_arr[force_mask]
    
    logger.info("Start calculating expected values for SNR parameter...")
    times_force = times[force_mask]
    snr_arr_force = snr_arr[:, force_mask]
    log_snr_arr, log_mean_list, log_median_list, log_std_list, log_difference_list = calculate_statistics_log_paramater(snr_arr_force)
    z_score_arr_log_snr = calculate_z_score_parameter(log_snr_arr, log_mean_list, log_std_list, all_channels)

    k_values_log_snr = find_k_value(z_score_arr_log_snr, all_channels, quantile=0.999)
    metadata = metadata_dict(station_id, first_run, last_run, times_force, trigger_type="FORCE", comment="")
    if args.save_values:
        logger.info("Saving expected values for SNR as JSON files...")
        save_values_json(k_values_log_snr, log_mean_list, log_std_list, filename=f"expected_snr_values_station{station_id}.json", SCRIPT_DIR=EXPECTED_VALUES_DIR_REF, metadata=metadata)

    flag_outliers_snr = outlier_flag(z_score_arr_log_snr, k_values_log_snr, all_channels)

    outlier_details_snr = find_outlier_details(z_score_arr_log_snr, k_values_log_snr, flag_outliers_snr, all_channels, run_no_force, event_number_force)
    sva.write_snr_outlier_details(outlier_details_snr, station_id, run_label, n_events_force = len(times_force), results_dir=RESULTS_DIR_REF)

    day_interval = choose_day_interval(times)
    plot_snr_against_time(station_id, times_force, snr_arr_force, flag_outliers_snr, z_score_arr_log_snr, k_values_log_snr, all_channels, PLOTS_DIR_REF, run_label, nrows=12, ncols=2, day_interval=day_interval)