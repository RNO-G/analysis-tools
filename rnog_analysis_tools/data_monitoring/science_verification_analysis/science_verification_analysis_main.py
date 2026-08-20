'''
This module can be used to test if the stations are working as expected. See README.md for more details.
'''

import rnog_data.runtable as rt
import logging
import os
import datetime
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import pandas as pd
import matplotlib.dates as mdates
from datetime import timezone
import copy
import csv
import json
import random
import string

# Import config files
from config_files_sva.config_plotting import set_plot_style

# Import analysis functions
from monitoring_data_functions_sva.get_monitoring_data_uproot import choose_trigger_type_header, read_multiple_runs
from analysis_functions_sva.spectral_analysis_sva import normalize_channels, normalize_channels_new, find_amplitude_ratio_in_band, find_amplitude_ratio_in_band_specific_bkg, excess_info_from_ratio, excess_info_from_ratio_specific_bkg, validate_excess_in_bands
from analysis_functions_sva.z_score_analysis_sva import calculate_statistics_log_paramater, calculate_z_score_parameter, symmetry_metrics_channel_z_score, symmetry_metrics_z_score, load_values_json, outlier_flag, find_outlier_details, calculate_expected_values_per_trigger, outlier_details
from analysis_functions_sva.vrms_analysis_sva import calculate_vrms, kde_modality, tail_fraction_and_trimmed_skew_two_sided, report_vrms_characteristics, get_rms_per_trigger_monitoring
from analysis_functions_sva.glitching_analysis_sva import binomtest_glitch_fraction
from analysis_functions_sva.block_offsets_analysis_sva_monitoring import get_force_block_offsets_monitoring, block_offset_statistics_monitoring, plot_block_offsets_violin_monitoring
from analysis_functions_sva.vrms_stability_analysis_sva import get_rms_per_run, relative_median_shift, decision_metric

# Import helper functions
from helper_functions.output_writer import write_failed_runs_to_csv, write_spectral_results, write_snr_outlier_details, write_vrms_outlier_details, write_vrms_modality_results, write_glitching_results, write_block_offset_results, block_offset_channel_health, create_result_csv_file, create_result_csv_file_didaq, write_readme_for_shifters
from helper_functions.read_rnog_runtable import read_rnog_runtable
from helper_functions.config_helper import get_station_config

# Import plotting functions
from plotting_functions_sva.plotting_sva_spectrum import plot_time_integrated_surface_spectra_unnormalized, plot_time_integrated_surface_spectra_normalized, plot_time_integrated_deep_spectra, plot_time_integrated_surface_spectra_normalized_example_reference
from plotting_functions_sva.plotting_sva_snr import plot_snr_against_time_single_channel, choose_day_interval, plot_snr_against_time, plot_snr_against_time_per_trigger
from plotting_functions_sva.plotting_sva_vrms import plot_vrms_values_against_time_single_channel_zscore, plot_vrms_values_against_time, plot_vrms_values_against_time_single_trigger_zscore, create_heatmap_plot, plot_vrms_values_against_time_per_trigger
from plotting_functions_sva.plotting_sva_glitch import glitching_violin_plot, choose_bin_size, plot_glitch_q99_over_time
from plotting_functions_sva.plotting_sva_debug import debug_plot_vrms_distribution_single_channel, debug_plot_ratios, debug_plot_snr_distribution, debug_plot_z_score_snr, debug_plot_vrms_distribution, debug_plot_ratios_just_galaxy
from plotting_functions_sva.plotting_sva_trigger_rate import plot_trigger_rates_over_time, plot_trigger_rate_heatmap

#### Script directory 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

#### Reference and config directories
REFERENCE_DIR = os.path.join(SCRIPT_DIR, "expected_values")
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config_files_sva")

#### Logging
logger = logging.getLogger(__name__)

def setup_logging(station_id, run_label, LOGS_DIR):

    log_file = os.path.join(LOGS_DIR, f"logging_science_verification_analysis_station{station_id}_{run_label}.log")

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

def failed_check_plot(validation_arr, label, *plot_fns, note=""):
        """Log pass/fail for a channel-health test and run plot_fns(save_dir), routing to
        failed_test_plots on failure or detailed_plots otherwise."""
        failed = validation_arr.isin(["X", "!!"]).any()
        save_dir = failed_test_plots if failed else detailed_plots
        status = "Some channels failed" if failed else "All channels passed"
        (logger.warning if failed else logger.info)(f"{status} the {label}.{note} Detailed plots saved in {save_dir}.")
        for plot_fn in plot_fns:
            plot_fn(save_dir)

def failed_check_report(validation_arr, label, *report_fns, note=""):
        """Log pass/fail for a channel-health test and run report_fns(save_dir), routing to
        failed_test_results on failure or detailed_results otherwise."""
        failed = validation_arr.isin(["X", "!!"]).any()
        save_dir = failed_test_results_dir if failed else detailed_results_dir
        status = "Some channels failed" if failed else "All channels passed"
        (logger.warning if failed else logger.info)(f"{status} the {label}.{note} Detailed results saved in {save_dir}.")
        for report_fn in report_fns:
            report_fn(save_dir)

if __name__ == "__main__":

    argparser = ArgumentParser(description="RNO-G Science Verification Analysis - extracting data from monitoring.root files")

    argparser.add_argument("-st", "--station_id", type=int, required=True, help="Station to analyze, e.g --station_id 14")
    argparser.add_argument("--data_location", type=str, default="desy", help="Location of the data. Use 'desy' (inbox data), 'uchicago' (mirrored data) or provide a custom path to the data directory, e.g. --data_location /path/to/data")
    argparser.add_argument("-ex", "--exclude-runs", nargs="+", type=int, default=[], metavar="RUN", help="Run number(s) to exclude, e.g. --exclude-runs 1005 1010")
    
    run_selection = argparser.add_mutually_exclusive_group(required=True)
    run_selection.add_argument("--runs", nargs="+", type=int, metavar="RUN_NUMBERS",
                           help="Run number(s) to analyze. Each run number should be given explicitly separated by a space, e.g. --runs 1001 1002 1005")
    run_selection.add_argument("--run_range", nargs=2, type=int, metavar=("START_RUN", "END_RUN"),
                            help="Range of run numbers to analyze (inclusive). Provide start and end run numbers separated by a space, e.g. --run_range 1000 1050")
    run_selection.add_argument("--time_range", nargs=2, type=str, metavar=("START_DATE", "END_DATE"),
                            help="Date range to analyze (inclusive). Provide start and end dates separated by a space in YYYY-MM-DD format, e.g. --time_range 2024-07-15 2024-09-30")

    args = argparser.parse_args()

    use_monitoring = True
    rms_label = "rms"

    station_id = args.station_id

    if args.runs:
        run_numbers = args.runs
    elif args.run_range:
        run_numbers = list(range(args.run_range[0], args.run_range[1] + 1))
    elif args.time_range:
        start_time, stop_time = args.time_range
        runtable = read_rnog_runtable(station_id, start_time, stop_time)
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

    # Date and random string for unique save directory
    date_label = datetime.datetime.now().strftime("%y-%m-%d")
    random_string = ''.join(random.choices(string.digits, k=6))

    save_directory_label = (f"{date_label}_station-{station_id}_run{first_run}-run{last_run}_{random_string}")

    # Choose the data location based on the argument provided and define the save location for the results
    if args.data_location == "desy":
        logger.info("Using DESY inbox data location for the analysis.")
        base_data_path = "/pnfs/ifh.de/acs/radio/diskonly/data/inbox/"
        result_base_data_path = "/pnfs/ifh.de/acs/radio/diskonly/NuRadioMC/science_verification_analysis"

    elif args.data_location == "uchicago":
        logger.info("Using UChicago mirrored data location for the analysis.")
        base_data_path = "/data/satellite"
        result_base_data_path = "/data/sva"

    else:
        logger.info(f"Using custom data location {args.data_location} for the analysis.")
        base_data_path = args.data_location
        #result_base_data_path = os.path.join(args.data_location, "results")
        result_base_data_path = "/pnfs/ifh.de/acs/radio/diskonly/NuRadioMC/science_verification_analysis"
        
        

    result_save_location = os.path.join(result_base_data_path, save_directory_label)
    os.makedirs(result_save_location, exist_ok=True)

    logger.info(f"Results will be saved in {result_save_location}.")

    # Output directories for plots, results, and logs
    save_location = os.path.join(result_save_location, "plots")
    results_dir = os.path.join(result_save_location, "test_results")
    csv_dir = os.path.join(result_save_location, "channel_health_summary")
    logs_dir = os.path.join(result_save_location, "logs")

    # Create output directories if they don't exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(save_location, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Path to README for shifters
    shifters_readme_file = os.path.join(result_save_location, "README_shifters.txt")

    ## Results directories
    # Always save results, even if all channels pass the tests
    detailed_results_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_results_dir, exist_ok=True)

    # Save  only if some channels fail the tests
    failed_test_results_dir = os.path.join(results_dir, "failed_test_results")
    os.makedirs(failed_test_results_dir, exist_ok=True)

    ## Plot directories
    # Plot always
    standard_plots = os.path.join(save_location, "standard_plots")
    os.makedirs(standard_plots, exist_ok=True)

    # Plot only if some channels fail the tests
    failed_test_plots = os.path.join(save_location, "failed_test_plots")
    os.makedirs(failed_test_plots, exist_ok=True)

    # Debug plots, e.g. heatmaps
    debug_plots = os.path.join(save_location, "debug_plots")
    os.makedirs(debug_plots, exist_ok=True)

    # Detailed plots, e.g. per channel plots (failed plots will be saved here if no channels fail the tests)
    detailed_plots = os.path.join(save_location, "detailed_plots")
    os.makedirs(detailed_plots, exist_ok=True)

    # Start logging
    setup_logging(station_id, run_label, logs_dir)
    logger.info(f"Starting analysis for station {station_id}, runs: {run_numbers} using the monitoring.root files.")

    # Set the plotting style
    set_plot_style()

    # Get channel lists from config
    station_config_json = os.path.join(CONFIG_DIR, "config_station.json")
    with open(station_config_json, "r") as f:
        station_config_data = json.load(f)

    default_station_config = station_config_data.get("default_config", {})
    station_specific_adjustments = station_config_data.get("station_specific_adjustments", {})
    config = get_station_config(station_id, default_station_config, station_specific_adjustments)

    surface_channels = config["surface_channels"]
    deep_channels = config["deep_channels"]
    upward_channels = config["upward_channels"]
    downward_channels = config["downward_channels"]
    vpol_channels = config["vpol_channels"]
    hpol_channels = config["hpol_channels"]
    phased_array_channels = config["phased_array_channels"]
    all_channels = config["all_channels"]
    reference_channels_galaxy = config["reference_channels_galaxy"]
    reference_channels = config["reference_channels"]

    # Choose RADIANT or DIDAQ based on the station configuration
    digitizer_type = config["daq_type"]
    if digitizer_type not in ["radiant", "didaq"]:
        logger.error(f"Invalid daq_type {digitizer_type}. Must be either 'radiant' or 'didaq'.")
        raise ValueError(f"Invalid daq_type {digitizer_type}. Must be either 'radiant' or 'didaq'. Please check the station configuration in config_station.json.")

    if digitizer_type == "radiant":
        logger.info(f"Using RADIANT digitizer type for station {station_id}. Triggers are FORCE, LT, RADIANT0, RADIANT1.")
        trigger_types_daq = {"force" : "FORCE", "lt" : "LT", "radiant0" : "RADIANT0", "radiant1" : "RADIANT1"}
    elif digitizer_type == "didaq":
        logger.info(f"Using DIDAQ digitizer type for station {station_id}. Triggers are FORCE, DIDAQ_DEEP_PHASED, DIDAQ_SURF_UP, DIDAQ_SURF_DOWN.")
        trigger_types_daq = {"force" : "FORCE", "lt" : "DIDAQ_DEEP_PHASED", "radiant0" : "DIDAQ_SURF_UP", "radiant1" : "DIDAQ_SURF_DOWN"}

    # Load event information from the combined_event_info dictionary:
    combined_event_info = read_multiple_runs(base_path = base_data_path, station_id = station_id, run_numbers=run_numbers, daq_type=digitizer_type)

    times = combined_event_info["trigger_time_utc"]
    valid_times_mask = ~pd.isna(times)
    times = times[valid_times_mask]
    invalid_runs = np.unique(combined_event_info["run_no"][~valid_times_mask])

    run_no = combined_event_info["run_no"][valid_times_mask]
    trigger_type_arr = combined_event_info["triggerType"][valid_times_mask]

    max_abs_amplitude_arr = combined_event_info["max_abs_amplitude_arr"][:, valid_times_mask]
    event_number_arr = combined_event_info["event_number_arr"][valid_times_mask]
    run_event_counts = combined_event_info["run_event_counts"] # dict with run number as key and value as another dict with n_events, n_forced_triggers, n_lt_triggers, n_rf0_triggers, n_rf1_triggers for that run
    run_trigger_rates = combined_event_info["run_trigger_rates"] # dict with run number as key and value as another dict with trigger rates for different trigger types for that run

    failed_run_info = combined_event_info["failed_run_info"] or {} # dict with run number as key and value as reason for failure, only for runs that failed to be read
    failed_runs = list(failed_run_info.keys())

    if len(invalid_runs) > 0:
        logger.warning(f"Some events have invalid trigger times and will be excluded from the analysis, events belong to the runs: {list(map(int, invalid_runs))}. ")
        for invalid_run in invalid_runs:
            failed_run_info[int(invalid_run)] = "Some events have been skipped in the analysis due to invalid timestamps, check logs for details"


    excluded_runs = args.exclude_runs.copy() if args.exclude_runs else []
    if excluded_runs:
        logger.info(f"Excluding runs {excluded_runs} from the analysis as specified by the user.")
        for excluded_run in excluded_runs:
            failed_run_info[int(excluded_run)] = "Run excluded by user"

    if failed_run_info:
        write_failed_runs_to_csv(station_id, failed_run_info, run_label, results_dir=results_dir) # Save in the results directory, since always important

    n_events_force = combined_event_info["total_n_force_triggers"]
    n_lt_events = combined_event_info["total_n_lt_triggers"] # DIDAQ_DEEP_PHASED for DIDAQ, LT for RADIANT
    n_radiant0_events = combined_event_info["total_n_rf0_triggers"] # DIDAQ_SURF_UP for DIDAQ, RADIANT0 for RADIANT
    n_radiant1_events = combined_event_info["total_n_rf1_triggers"] # DIDAQ_SURF_DOWN for DIDAQ, RADIANT1 for RADIANT

    # Spectral info:
    freqs = combined_event_info["freqs"]
    avg_spectrum = combined_event_info["avg_spectrum"]
    spec_arr_force = combined_event_info["avg_spectrum_force"]
    spec_arr_lt = combined_event_info["avg_spectrum_lt"] # DIDAQ_DEEP_PHASED for DIDAQ, LT for RADIANT
    spec_arr_radiant0 = combined_event_info["avg_spectrum_rf0"] # DIDAQ_SURF_UP for DIDAQ, RADIANT0 for RADIANT
    spec_arr_radiant1 = combined_event_info["avg_spectrum_rf1"] # DIDAQ_SURF_DOWN for DIDAQ, RADIANT1 for RADIANT

    # Glitching, SNR and block offset info:
    rms_arr = combined_event_info["rms_arr"][:, valid_times_mask]
    snr_arr = combined_event_info["snr_arr"][:, valid_times_mask]

    if digitizer_type == "radiant":
        glitch_arr = combined_event_info["glitching_test_statistic_arr"][:, valid_times_mask]
        block_offsets_arr = combined_event_info["block_offsets_arr"][:, valid_times_mask]

    # Choose the day interval for plotting based on the time range of the events
    day_interval = choose_day_interval(times)

    # Spectral analysis configuration parameters
    spectral_analysis_config_json = os.path.join(CONFIG_DIR, "config_spectral_analysis.json")
    with open(spectral_analysis_config_json, "r") as f:
        spectral_analysis_config_dict = json.load(f)

    spectral_bands = spectral_analysis_config_dict["spectral_bands"]
    alpha_spec = spectral_analysis_config_dict["alpha_spec"]
    ci_threshold_spec = spectral_analysis_config_dict["ci_threshold_spec"]
    normalization_band = spectral_analysis_config_dict["normalization_band"]
    log_ratio_thresholds_spec = spectral_analysis_config_dict["log_ratio_thresholds_spec"]

    # Normalize the spectra for the FORCE trigger events
    norm_spec_arr_force, scale_factors_force = normalize_channels_new(spec_arr_force, freqs, downward_channels, upward_channels, normalization_band=normalization_band)

    # Masks for different trigger types
    force_mask = choose_trigger_type_header(trigger_type_arr, trigger_types_daq["force"], digitizer_type) # FORCE for both DIDAQ and RADIANT
    lt_mask = choose_trigger_type_header(trigger_type_arr, trigger_types_daq["lt"], digitizer_type) # DIDAQ_DEEP_PHASED for DIDAQ, LT for RADIANT
    radiant0_mask = choose_trigger_type_header(trigger_type_arr, trigger_types_daq["radiant0"], digitizer_type) # DIDAQ_SURF_UP for DIDAQ, RADIANT0 for RADIANT
    radiant1_mask = choose_trigger_type_header(trigger_type_arr, trigger_types_daq["radiant1"], digitizer_type) # DIDAQ_SURF_DOWN for DIDAQ, RADIANT1 for RADIANT

    run_no_force = run_no[force_mask]
    event_number_force = event_number_arr[force_mask]

    snr_arr_radiant0 = snr_arr[:, radiant0_mask]
    snr_arr_radiant1 = snr_arr[:, radiant1_mask]
    snr_arr_lt = snr_arr[:, lt_mask]

    times_radiant0 = times[radiant0_mask]
    times_radiant1 = times[radiant1_mask]
    times_lt = times[lt_mask]

    # Bands for spectral analysis
    band_config = copy.deepcopy(spectral_bands)

    for band_name in band_config:
        if band_name != "galactic_excess":
            band_config[band_name]["reference_channels"] = reference_channels
        elif band_name == "galactic_excess":
            band_config[band_name]["reference_channels"] = reference_channels_galaxy
        else:
            logger.error(f"Unknown band name {band_name} in SPECTRAL_BANDS config")
            raise ValueError(f"Unknown band name {band_name} in SPECTRAL_BANDS config")

    logger.debug(f"Band configuration for spectral analysis: {band_config}")

    ratio_arr_dict = find_amplitude_ratio_in_band_specific_bkg(freqs, norm_spec_arr_force, upward_channels, downward_channels, **band_config)

    channels_order = upward_channels + downward_channels
    ch_to_idx = {ch: i for i, ch in enumerate(channels_order)}
    logger.debug(f"Channel to index mapping: {ch_to_idx}")

    all_excess_info = {}
    all_validation_results = {}

    logger.info("Starting spectral analysis for FORCE trigger events. !!! Different methods for monitoring and dataProviderRNOG !!! ")
    for ch in surface_channels:
        i = ch_to_idx[ch]
        ratio_arr_dict_ch = {}

        for band_name, ratio_arr in ratio_arr_dict.items():
            ratio_arr_dict_ch[band_name] = ratio_arr[i]

        excess_info_results = excess_info_from_ratio_specific_bkg(ratio_arr_dict_ch, alpha_spec, ci_threshold_spec, use_monitoring=use_monitoring, log_ratio_thresholds=log_ratio_thresholds_spec)
        validation_results = validate_excess_in_bands(excess_info_results)

        all_excess_info[ch] = excess_info_results
        all_validation_results[ch] = validation_results

    ###### SNR analysis
    logger.info("Starting SNR analysis for FORCE trigger events...")
    snr_arr_force = snr_arr[:, force_mask]
    times = np.array(times)
    times_force = times[force_mask]

    log_snr_arr, log_mean_list, log_median_list, log_std_list, log_difference_list = calculate_statistics_log_paramater(snr_arr_force)
    reference_filename = f"expected_snr/expected_snr_values_station{station_id}.json"
    k_values_log_snr, ref_log_mean_list, ref_log_std_list = load_values_json(REFERENCE_DIR, reference_filename)
    z_score_arr_log_snr = calculate_z_score_parameter(log_snr_arr, ref_log_mean_list, ref_log_std_list, all_channels)
    flag_outliers_snr = outlier_flag(z_score_arr_log_snr, k_values_log_snr, all_channels)

    outlier_details_snr = find_outlier_details(z_score_arr_log_snr, k_values_log_snr, flag_outliers_snr, all_channels, run_no_force, event_number_force)
    
    ##### Vrms analysis
    logger.info("Starting Vrms analysis for monitoring data...")
    
    rms_arr, rms_arr_force, rms_arr_radiant0, rms_arr_radiant1, rms_arr_lt = get_rms_per_trigger_monitoring(rms_arr=rms_arr, force_mask=force_mask, lt_mask=lt_mask, radiant0_mask=radiant0_mask, radiant1_mask=radiant1_mask)

    logger.info(f"Number of {trigger_types_daq['radiant0']} trigger events: {len(rms_arr_radiant0[1])}, Number of {trigger_types_daq['radiant1']} trigger events: {len(rms_arr_radiant1[1])}, Number of {trigger_types_daq['lt']} trigger events: {len(rms_arr_lt[1])}")
    logger.info(f"Calculating RMS modality and tail characteristics for each trigger type...")

    # Load the configuration parameters for the RMS analysis from the JSON file
    rms_config_json = os.path.join(CONFIG_DIR, "config_rms.json")
    with open(rms_config_json, "r") as f:
        rms_config_dict = json.load(f)

    kde_modality_function_parameters = rms_config_dict["kde_modality_function_parameters"]
    skewness_function_parameters = rms_config_dict["skewness_function_parameters"]
    report_vrms_function_parameters = rms_config_dict["report_vrms_function_parameters"]

    modality_dict_force = kde_modality(rms_arr_force, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_force = tail_fraction_and_trimmed_skew_two_sided(rms_arr_force, all_channels, skewness_config=skewness_function_parameters)
    if len(rms_arr_force[1]) < 100:
        logger.warning(f"{trigger_types_daq['force']} trigger has less than 100 valid RMS entries ({len(rms_arr_force[1])}). Results for the Vrms statistics may be unreliable.")
    modality_force, tail_label_force = report_vrms_characteristics(modality_dict_force, tail_dict_force, all_channels, report_config=report_vrms_function_parameters)

    modality_dict_radiant0 = kde_modality(rms_arr_radiant0, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_radiant0 = tail_fraction_and_trimmed_skew_two_sided(rms_arr_radiant0, all_channels, skewness_config=skewness_function_parameters)
    if len(rms_arr_radiant0[1]) < 100:
        logger.warning(f"{trigger_types_daq['radiant0']} trigger has less than 100 valid RMS entries ({len(rms_arr_radiant0[1])}). Results for the Vrms statistics may be unreliable.")
    modality_radiant0, tail_label_radiant0 = report_vrms_characteristics(modality_dict_radiant0, tail_dict_radiant0, all_channels, report_config=report_vrms_function_parameters)

    modality_dict_radiant1 = kde_modality(rms_arr_radiant1, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_radiant1 = tail_fraction_and_trimmed_skew_two_sided(rms_arr_radiant1, all_channels, skewness_config=skewness_function_parameters)
    if len(rms_arr_radiant1[1]) < 100:
        logger.warning(f"{trigger_types_daq['radiant1']} trigger has less than 100 valid RMS entries ({len(rms_arr_radiant1[1])}). Results for the Vrms statistics may be unreliable.")
    modality_radiant1, tail_label_radiant1 = report_vrms_characteristics(modality_dict_radiant1, tail_dict_radiant1, all_channels, report_config=report_vrms_function_parameters)

    modality_dict_lt = kde_modality(rms_arr_lt, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_lt = tail_fraction_and_trimmed_skew_two_sided(rms_arr_lt, all_channels, skewness_config=skewness_function_parameters)
    if len(rms_arr_lt[1]) < 100:
        logger.warning(f"{trigger_types_daq['lt']} trigger has less than 100 valid RMS entries ({len(rms_arr_lt[1])}). Results for the Vrms statistics may be unreliable.")

    modality_lt, tail_label_lt = report_vrms_characteristics(modality_dict_lt, tail_dict_lt, all_channels, report_config=report_vrms_function_parameters)

    ## Vrms stability
    reference_filename_rms = f"expected_{rms_label}/expected_{rms_label}_station{station_id}.json"
    vrms_k_values, vrms_ref_mean, vrms_ref_std = load_values_json(REFERENCE_DIR, reference_filename_rms)
    z_score_arr_vrms_force = calculate_z_score_parameter(rms_arr_force, vrms_ref_mean, vrms_ref_std, all_channels)
    flag_outliers_vrms_force = outlier_flag(z_score_arr_vrms_force, vrms_k_values, all_channels)
    outlier_details_vrms_force = find_outlier_details(z_score_arr_vrms_force, vrms_k_values, flag_outliers_vrms_force, channel_list=all_channels, run_no=run_no_force, event_number=event_number_force)

    rms_arr_per_run_dict_force = get_rms_per_run(rms_arr_force, run_no_force)
    relative_median_shift_results = relative_median_shift(rms_arr_per_run_dict_force, all_channels)
    
    rms_results = decision_metric(outlier_details_vrms_force, relative_median_shift_results, n_events_force=n_events_force, channels=all_channels)

    ##### Glitching and block offset analysis - only for RADIANT digitizer type
    if digitizer_type == "radiant":
        ##### Glitching analysis
        logger.info("Starting glitching analysis...")

        # Load the configuration parameters for the glitching analysis from the JSON file
        glitching_config_json = os.path.join(CONFIG_DIR, "config_glitching.json")
        with open(glitching_config_json, "r") as f:
            glitching_config_dict = json.load(f)

        config_glitching = glitching_config_dict["config_glitching_values"]

        glitch_info = binomtest_glitch_fraction(glitch_arr, all_channels, config_glitching=config_glitching)

        ##### Block offsets analysis
        # Get the reference block offset results for the station
        ref_block_offset_results_file = os.path.join(REFERENCE_DIR, "expected_block_offsets", f"expected_block_offsets_station{station_id}.json")
        with open(ref_block_offset_results_file, "r") as f:
            ref_block_offset_results = json.load(f)

        logger.info("Starting block offset analysis (monitoring.root), results are not used to determine channel health, see warnings in the log file for channels with potential block offset issues. The block offsets are then removed.")
        block_offset_arr_force = get_force_block_offsets_monitoring(block_offsets_arr, force_mask)
        block_offset_stats = block_offset_statistics_monitoring(block_offset_arr_force=block_offset_arr_force, channel_list=all_channels)

        block_offset_results_dict = block_offset_channel_health(block_offset_stats, ref_block_off_dict=ref_block_offset_results, use_monitoring=use_monitoring)


    # Create summary CSV file
    if digitizer_type == "radiant":
        results_df = create_result_csv_file(
            station_id,
            run_label,
            n_events_force,
            surface_channels,
            downward_channels,
            upward_channels,
            all_channels,
            all_validation_results,
            glitch_info,
            block_offset_results_dict,
            rms_results,
            modality_dict_force,
            modality_dict_lt,
            modality_dict_radiant0,
            modality_dict_radiant1,
            outlier_details_snr,
            csv_dir,
            rms_label
        )
        rms_modality_lt_validation_arr = results_df[f"{rms_label.capitalize()} (LT)"]
        rms_modality_radiant0_validation_arr = results_df[f"{rms_label.capitalize()} (RADIANT0)"]
        rms_modality_radiant1_validation_arr = results_df[f"{rms_label.capitalize()} (RADIANT1)"]
        glitching_validation_arr = results_df["Glitching"]
        block_offset_validation_arr = results_df["Block Offsets"]

    elif digitizer_type == "didaq":
        results_df = create_result_csv_file_didaq(
            station_id,
            run_label,
            n_events_force,
            surface_channels,
            downward_channels,
            upward_channels,
            all_channels,
            all_validation_results,
            rms_results,
            modality_dict_force,
            modality_dict_lt,
            modality_dict_radiant0,
            modality_dict_radiant1,
            outlier_details_snr,
            csv_dir,
            rms_label
        )
        rms_modality_lt_validation_arr = results_df[f"{rms_label.capitalize()} (DEEP PHASED)"]
        rms_modality_radiant0_validation_arr = results_df[f"{rms_label.capitalize()} (SURF UP)"]
        rms_modality_radiant1_validation_arr = results_df[f"{rms_label.capitalize()} (SURF DOWN)"]

    snr_validation_arr = results_df["SNR"]
    galaxy_validation_arr = results_df["Galaxy (FORCE)"]
    rms_stability_validation_arr = results_df[f"{rms_label.capitalize()} Stability (FORCE)"]
    rms_modality_force_validation_arr = results_df[f"{rms_label.capitalize()} (FORCE)"]


    #### Plotting ####

    #### Standard plots for the analysis results
    # FORCE trigger spectra - normalized, unnormalized
    plot_time_integrated_surface_spectra_normalized(station_id, norm_spec_arr_force, freqs, upward_channels, downward_channels, standard_plots, run_label, use_monitoring=use_monitoring, run_event_counts=run_event_counts)
    plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr_force, freqs, upward_channels, downward_channels, standard_plots, run_label, trigger_label="force", use_monitoring=use_monitoring, run_event_counts=run_event_counts, daq_type=digitizer_type)
    plot_time_integrated_deep_spectra(station_id, spec_arr_force, freqs, vpol_channels, hpol_channels, standard_plots, run_label, trigger_label="force", use_monitoring=use_monitoring, run_event_counts=run_event_counts, daq_type=digitizer_type)

    # SNR against time (FORCE trigger)
    plot_snr_against_time(station_id, times_force, snr_arr_force, flag_outliers_snr, z_score_arr_log_snr, k_values_log_snr, all_channels, standard_plots, run_label, nrows=12, ncols=2, day_interval=day_interval)

    # FORCE trigger RMS against time
    plot_vrms_values_against_time_single_trigger_zscore(times_force, rms_arr_force, flag_outliers_vrms_force, z_score_arr_vrms_force, vrms_k_values, trigger_name="FORCE", channel_list=all_channels, station_id=station_id, run_label=run_label, save_location=standard_plots, n_rows=12, n_cols=2, day_interval=day_interval, use_monitoring=use_monitoring)

    # Trigger rate plots over time for all trigger types
    plot_trigger_rates_over_time(run_trigger_rates, standard_plots, station_id, run_label, daq_type = digitizer_type)

    #### Debug plots/reports for the analysis results, routed by pass/fail status of each test
    # SNR test: plot SNR distribution and z-score distribution, write outlier details to text file
    failed_check_plot(snr_validation_arr, "SNR test",
        lambda d: debug_plot_snr_distribution(log_snr_arr, channel_list=all_channels, save_location=d, station_id=station_id, run_label=run_label, bins=30),
        lambda d: debug_plot_z_score_snr(z_score_arr_log_snr, channel_list=all_channels, save_location=d, station_id=station_id, run_label=run_label, bins=30))

    failed_check_report(snr_validation_arr, "SNR test",
        lambda d: write_snr_outlier_details(outlier_details_snr, station_id, run_label, n_events_force, results_dir=d))

    # Galaxy test: plot ratios for each channel, write spectral results to text file
    failed_check_plot(galaxy_validation_arr, "Galaxy test for FORCE trigger",
        lambda d: debug_plot_ratios(ratio_arr_dict=ratio_arr_dict, channels_order=channels_order, save_location=d, station_id=station_id, run_label=run_label, bins=30))

    failed_check_report(galaxy_validation_arr, "Galaxy test for FORCE trigger",
        lambda d: write_spectral_results(all_excess_info, surface_channels, station_id, run_label, results_dir=d))

    # RMS stability test: plot relative median shift heatmap, write results to text file
    failed_check_report(rms_stability_validation_arr, "RMS stability test",
        lambda d: write_vrms_outlier_details(outlier_details_vrms_force, station_id, run_label, trigger_label="FORCE", n_events=n_events_force, results_dir=d, use_monitoring=use_monitoring),
        lambda d: (lambda f: json.dump(relative_median_shift_results, f, indent=4))(open(os.path.join(d, f"{rms_label}_relative_median_shift_results_force_trigger_station{station_id}_{run_label}.json"), "w")),
        lambda d: (lambda f: json.dump(rms_results, f, indent=4))(open(os.path.join(d, f"rms_stability_decision_results_force_trigger_station{station_id}_{run_label}.json"), "w")))
        
    
    # RMS modality tests: FORCE is used for overall channel health, the others are informational only
    rms_modality_checks = [
        (rms_modality_force_validation_arr, "FORCE", rms_arr_force, modality_dict_force, ""),
        (rms_modality_lt_validation_arr, trigger_types_daq["lt"], rms_arr_lt, modality_dict_lt, " (isn't included in overall channel health but might indicate a problem)"),
        (rms_modality_radiant0_validation_arr, trigger_types_daq["radiant0"], rms_arr_radiant0, modality_dict_radiant0, " (isn't included in overall channel health but might indicate a problem)"),
        (rms_modality_radiant1_validation_arr, trigger_types_daq["radiant1"], rms_arr_radiant1, modality_dict_radiant1, " (isn't included in overall channel health but might indicate a problem)"),
    ]
    for validation_arr, trigger_label, rms_arr_trig, modality_dict, note in rms_modality_checks:
        failed_check_plot(validation_arr, f"{rms_label} modality test for {trigger_label} trigger",
            lambda d, rms_arr_trig=rms_arr_trig, modality_dict=modality_dict, trigger_label=trigger_label: debug_plot_vrms_distribution(rms_arr_trig, modality_dict, channel_list=all_channels, station_id=station_id, run_label=run_label, trigger_label=trigger_label, save_location=d, n_rows=12, n_cols=2, use_monitoring=use_monitoring),
            note=note)

    failed_check_report(rms_modality_force_validation_arr, f"{rms_label} modality test for FORCE trigger",
        lambda d: write_vrms_modality_results(modality_force, tail_label_force, trigger_label=trigger_types_daq['force'], station_id=station_id, run_label=run_label, results_dir=d, use_monitoring=use_monitoring),
        lambda d: write_vrms_modality_results(modality_radiant0, tail_label_radiant0, trigger_label=trigger_types_daq['radiant0'], station_id=station_id, run_label=run_label, results_dir=d, use_monitoring=use_monitoring),
        lambda d: write_vrms_modality_results(modality_radiant1, tail_label_radiant1, trigger_label=trigger_types_daq['radiant1'], station_id=station_id, run_label=run_label, results_dir=d, use_monitoring=use_monitoring),
        lambda d: write_vrms_modality_results(modality_lt, tail_label_lt, trigger_label=trigger_types_daq['lt'], station_id=station_id, run_label=run_label, results_dir=d, use_monitoring=use_monitoring))
 

    if digitizer_type == "radiant":
        failed_check_plot(glitching_validation_arr, "glitching test",
            lambda d: plot_glitch_q99_over_time(np.array(times), glitch_arr, all_channels, station_id, run_label, d),
            lambda d: glitching_violin_plot(glitch_arr, all_channels, station_id, run_label, d))

        failed_check_report(glitching_validation_arr, "glitching test",
            lambda d: write_glitching_results(glitch_info, station_id, run_label, all_channels, results_dir=d))

        failed_check_plot(block_offset_validation_arr, "block offset test",
            lambda d: plot_block_offsets_violin_monitoring(block_offset_arr_force, all_channels, station_id, run_label, d))

        failed_check_report(block_offset_validation_arr, "block offset test",
            lambda d: write_block_offset_results(block_offset_stats, station_id, run_label, ref_block_off_dict=ref_block_offset_results, results_dir=d, use_monitoring=use_monitoring))

    #### Other plots (always saved to other_debug_plots)
    for trig_key in ("lt", "radiant0", "radiant1"):
        spec_arr_trig = {"lt": spec_arr_lt, "radiant0": spec_arr_radiant0, "radiant1": spec_arr_radiant1}[trig_key]
        trigger_label = trigger_types_daq[trig_key]
        # Surface spectrum
        plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr_trig, freqs, upward_channels, downward_channels, detailed_plots, run_label, trigger_label=trigger_label, use_monitoring=use_monitoring, run_event_counts=run_event_counts, daq_type=digitizer_type)
        # Deep spectrum (unnormalized)
        plot_time_integrated_deep_spectra(station_id, spec_arr_trig, freqs, vpol_channels, hpol_channels, detailed_plots, run_label, trigger_label=trigger_label, use_monitoring=use_monitoring, run_event_counts=run_event_counts, daq_type=digitizer_type)

    # RMS
    plot_vrms_values_against_time(times, rms_arr, all_channels, station_id, run_label, detailed_plots, force_mask, radiant0_mask, radiant1_mask, lt_mask, daq_type=digitizer_type, n_rows=12, n_cols=2, day_interval=day_interval, use_monitoring=use_monitoring)
    plot_vrms_values_against_time_per_trigger(times, rms_arr, all_channels, station_id, run_label, detailed_plots, force_mask, radiant0_mask, radiant1_mask, lt_mask, daq_type=digitizer_type, n_rows=12, n_cols=2, day_interval=day_interval, use_monitoring=use_monitoring)

    # Trigger rate plots
    plot_trigger_rate_heatmap(run_trigger_rates, detailed_plots, station_id, run_label, daq_type=digitizer_type)

    # SNR
    for times_trig, snr_arr_trig, color, trig_key in (
        (times_radiant0, snr_arr_radiant0, "tab:orange", "radiant0"),
        (times_radiant1, snr_arr_radiant1, "tab:green", "radiant1"),
        (times_lt, snr_arr_lt, "tab:red", "lt"),
    ):
        plot_snr_against_time_per_trigger(station_id, times_trig, snr_arr_trig, all_channels, detailed_plots, run_label, nrows=12, ncols=2, day_interval=day_interval, color=color, triggerlabel=trigger_types_daq[trig_key])

    # Debug plots - for now only median RMS shift heatmaps
    create_heatmap_plot(relative_median_shift_results, label="Relative Median Shift", save_dir=debug_plots, channel_list=all_channels, station_id=station_id, matrix_key="median_shift_matrix", run_label=run_label, cmap="Reds")

    logger.info(f"Analysis completed for station {station_id}, run label {run_label}. Results saved in {results_dir}. Standard plots saved in {standard_plots}. Detailed plots saved in {detailed_plots}. Plots for failed tests saved in {failed_test_plots}. Debug plots saved in {debug_plots}. Summary CSV saved in {csv_dir}. Logs saved in {logs_dir}.")

    ## Write README for shifters

    write_readme_for_shifters(shifters_readme_file, station_id, run_numbers, times, run_label)



