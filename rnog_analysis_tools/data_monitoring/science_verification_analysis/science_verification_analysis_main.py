'''
This module can be used to test if the stations are working as expected.
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
from helper_functions.output_writer import write_failed_runs_to_csv, write_spectral_results, write_snr_outlier_details, write_vrms_outlier_details, write_vrms_modality_results, write_glitching_results, write_block_offset_results, create_result_csv_file, create_result_csv_file_didaq
from helper_functions.read_rnog_runtable import read_rnog_runtable
from helper_functions.config_helper import get_station_config

# Import plotting functions
from plotting_functions_sva.plotting_sva_spectrum import plot_time_integrated_surface_spectra_unnormalized, plot_time_integrated_surface_spectra_normalized, plot_time_integrated_deep_spectra, plot_time_integrated_surface_spectra_normalized_example_reference
from plotting_functions_sva.plotting_sva_snr import plot_snr_against_time_single_channel, choose_day_interval, plot_snr_against_time, plot_snr_against_time_per_trigger
from plotting_functions_sva.plotting_sva_vrms import plot_vrms_values_against_time_single_channel_zscore, plot_vrms_values_against_time, plot_vrms_values_against_time_single_trigger_zscore, create_heatmap_plot, plot_vrms_values_against_time_per_trigger
from plotting_functions_sva.plotting_sva_glitch import glitching_violin_plot, choose_bin_size, plot_glitch_q99_over_time
from plotting_functions_sva.plotting_sva_debug import debug_plot_vrms_distribution_single_channel, debug_plot_ratios, debug_plot_snr_distribution, debug_plot_z_score_snr, debug_plot_vrms_distribution, debug_plot_ratios_just_galaxy
from plotting_functions_sva.plotting_sva_trigger_rate import plot_trigger_rates_over_time, plot_trigger_rate_heatmap

#### Script directory for json files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

#### Output directories for plots, results, and logs
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "detailed_results")
CSV_DIR = os.path.join(SCRIPT_DIR, "channel_health_summary")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
REFERENCE_DIR = os.path.join(SCRIPT_DIR, "expected_values")
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config_files_sva")

# Create output directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


#### Logging
logger = logging.getLogger(__name__)

def setup_logging(station_id, run_label):

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
    
if __name__ == "__main__":

    argparser = ArgumentParser(description="RNO-G Science Verification Analysis - extracting data from monitoring.root files")
    
    argparser.add_argument("-st", "--station_id", type=int, required=True, help="Station to analyze, e.g --station_id 14")
    argparser.add_argument("-sl", "--save_location", type=str, default=PLOTS_DIR, help="Location to save the output plots (default: plots directory under script directory), e.g. --save_location /path/to/save/plots")
    argparser.add_argument("-ex", "--exclude-runs", nargs="+", type=int, default=[], metavar="RUN", help="Run number(s) to exclude, e.g. --exclude-runs 1005 1010")
    argparser.add_argument("--debug_plot", action="store_true", help="If set, will create debug plots.")
    argparser.add_argument("--base_data_path", type = str, default = "/pnfs/ifh.de/acs/radio/diskonly/data/inbox/", help="Base path to the data directory (default: /pnfs/ifh.de/acs/radio/diskonly/data/inbox/), e.g. --base_data_path /path/to/data")

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

    base_data_path = args.base_data_path

    # Start logging
    setup_logging(station_id, run_label)
    logger.info(f"Starting analysis for station {station_id}, runs: {run_numbers} using the monitoring.root files.")
    
    # Set the plotting style
    set_plot_style()

    # Create save location directory if it doesn't exist
    save_location = os.path.expanduser(args.save_location)
    os.makedirs(save_location, exist_ok=True)

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
    elif digitizer_type == "didaq":
        logger.info(f"Using DIDAQ digitizer type for station {station_id}. Triggers are FORCE, DIDAQ_DEEP_PHASED, DIDAQ_SURF_UP, DIDAQ_SURF_DOWN.")

    # base_data_path = "/cephfs/users/zeynepsu/analysis-tools/rnog_analysis_tools/data_monitoring/science_verification_analysis/new_station_first_runs"
    
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
        write_failed_runs_to_csv(station_id, failed_run_info, run_label, results_dir=RESULTS_DIR)
    
    n_events_force = combined_event_info["total_n_force_triggers"]
    n_lt_events = combined_event_info["total_n_lt_triggers"]
    n_radiant0_events = combined_event_info["total_n_rf0_triggers"]
    n_radiant1_events = combined_event_info["total_n_rf1_triggers"]

    # Spectral info:
    freqs = combined_event_info["freqs"]
    avg_spectrum = combined_event_info["avg_spectrum"]
    spec_arr_force = combined_event_info["avg_spectrum_force"]
    spec_arr_lt = combined_event_info["avg_spectrum_lt"]
    spec_arr_radiant0 = combined_event_info["avg_spectrum_rf0"]
    spec_arr_radiant1 = combined_event_info["avg_spectrum_rf1"]

    # Glitching, SNR and block offset info:
    rms_arr = combined_event_info["rms_arr"][:, valid_times_mask]
    glitch_arr = combined_event_info["glitching_test_statistic_arr"][:, valid_times_mask]
    block_offsets_arr = combined_event_info["block_offsets_arr"][:, valid_times_mask]
    snr_arr = combined_event_info["snr_arr"][:, valid_times_mask]

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
    norm_spec_arr_force, scale_factors_force = normalize_channels_new(spec_arr_force, freqs, downward_channels, upward_channels, normalization_band = normalization_band)
    
    # Masks for different trigger types
    force_mask = choose_trigger_type_header(trigger_type_arr, "FORCE")
    lt_mask = choose_trigger_type_header(trigger_type_arr, "LT")
    radiant0_mask = choose_trigger_type_header(trigger_type_arr, "RADIANT0")
    radiant1_mask = choose_trigger_type_header(trigger_type_arr, "RADIANT1")

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

        # Write detailed spectral results to text file for each channel
        write_spectral_results(ch, excess_info_results, station_id, run_label, results_dir=RESULTS_DIR, log_once=(ch==surface_channels[-1]), reset_file=(ch==surface_channels[0]))    

    # Surface spectrum
    plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr_force, freqs, upward_channels, downward_channels, save_location, run_label, trigger_label="force", use_monitoring = use_monitoring, run_event_counts = run_event_counts)
    plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr_lt, freqs, upward_channels, downward_channels, save_location, run_label, trigger_label="lt", use_monitoring = use_monitoring, run_event_counts = run_event_counts)
    plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr_radiant0, freqs, upward_channels, downward_channels, save_location, run_label, trigger_label="radiant0", use_monitoring = use_monitoring, run_event_counts = run_event_counts)
    plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr_radiant1, freqs, upward_channels, downward_channels, save_location, run_label, trigger_label="radiant1", use_monitoring = use_monitoring, run_event_counts = run_event_counts)
    
    # Normalized surface spectrum - only force
    plot_time_integrated_surface_spectra_normalized(station_id, norm_spec_arr_force, freqs, upward_channels, downward_channels, save_location, run_label, use_monitoring = use_monitoring, run_event_counts = run_event_counts)

    # Deep spectrum (unnormalized)
    plot_time_integrated_deep_spectra(station_id, spec_arr_force, freqs, vpol_channels, hpol_channels, save_location, run_label, trigger_label="force", use_monitoring = use_monitoring, run_event_counts = run_event_counts)
    plot_time_integrated_deep_spectra(station_id, spec_arr_lt, freqs, vpol_channels, hpol_channels, save_location, run_label, trigger_label="lt", use_monitoring = use_monitoring, run_event_counts = run_event_counts)
    plot_time_integrated_deep_spectra(station_id, spec_arr_radiant0, freqs, vpol_channels, hpol_channels, save_location, run_label, trigger_label="radiant0", use_monitoring = use_monitoring, run_event_counts = run_event_counts)
    plot_time_integrated_deep_spectra(station_id, spec_arr_radiant1, freqs, vpol_channels, hpol_channels, save_location, run_label, trigger_label="radiant1", use_monitoring = use_monitoring, run_event_counts = run_event_counts)
    
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
    write_snr_outlier_details(outlier_details_snr, station_id, run_label, n_events_force, results_dir=RESULTS_DIR)

    day_interval = choose_day_interval(times)
    plot_snr_against_time(station_id, times_force, snr_arr_force, flag_outliers_snr, z_score_arr_log_snr, k_values_log_snr, all_channels, save_location, run_label, nrows=12, ncols=2, day_interval=day_interval)
       
    ##### Vrms analysis
    logger.info("Starting Vrms analysis for monitoring data...")
    # Still named as Vrms for consistency but they are actually RMS values
    vrms_arr,vrms_arr_force, vrms_arr_radiant0, vrms_arr_radiant1, vrms_arr_lt = get_rms_per_trigger_monitoring(rms_arr=rms_arr, force_mask=force_mask, lt_mask=lt_mask, radiant0_mask=radiant0_mask, radiant1_mask=radiant1_mask)

    logger.info(f"Number of RADIANT0 trigger events: {len(vrms_arr_radiant0[1])}, Number of RADIANT1 trigger events: {len(vrms_arr_radiant1[1])}, Number of LT trigger events: {len(vrms_arr_lt[1])}")
    logger.info(f"Calculating RMS (for monitoring.root) or Vrms (for dataProviderRNOG) modality and tail characteristics for each trigger type...")
    
    # Load the configuration parameters for the RMS analysis from the JSON file
    rms_config_json = os.path.join(CONFIG_DIR, "config_rms.json")
    with open(rms_config_json, "r") as f:
        rms_config_dict = json.load(f)
    
    kde_modality_function_parameters = rms_config_dict["kde_modality_function_parameters"]
    skewness_function_parameters = rms_config_dict["skewness_function_parameters"]
    report_vrms_function_parameters = rms_config_dict["report_vrms_function_parameters"]
    
    modality_dict_force = kde_modality(vrms_arr_force, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_force = tail_fraction_and_trimmed_skew_two_sided(vrms_arr_force, all_channels, skewness_config=skewness_function_parameters)
    if len(vrms_arr_force[1]) < 100:
        logger.warning(f"FORCE trigger has less than 100 valid RMS entries ({len(vrms_arr_force[1])}). Results for the Vrms statistics may be unreliable.")
    modality_force, tail_label_force = report_vrms_characteristics(modality_dict_force, tail_dict_force, all_channels, report_config=report_vrms_function_parameters)

    modality_dict_radiant0 = kde_modality(vrms_arr_radiant0, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_radiant0 = tail_fraction_and_trimmed_skew_two_sided(vrms_arr_radiant0, all_channels, skewness_config=skewness_function_parameters)
    if len(vrms_arr_radiant0[1]) < 100:
        logger.warning(f"RADIANT0 trigger has less than 100 valid RMS entries ({len(vrms_arr_radiant0[1])}). Results for the Vrms statistics may be unreliable.")
    modality_radiant0, tail_label_radiant0 = report_vrms_characteristics(modality_dict_radiant0, tail_dict_radiant0, all_channels, report_config=report_vrms_function_parameters)

    modality_dict_radiant1 = kde_modality(vrms_arr_radiant1, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_radiant1 = tail_fraction_and_trimmed_skew_two_sided(vrms_arr_radiant1, all_channels, skewness_config=skewness_function_parameters)
    if len(vrms_arr_radiant1[1]) < 100:
        logger.warning(f"RADIANT1 trigger has less than 100 valid RMS entries ({len(vrms_arr_radiant1[1])}). Results for the Vrms statistics may be unreliable.")
    modality_radiant1, tail_label_radiant1 = report_vrms_characteristics(modality_dict_radiant1, tail_dict_radiant1, all_channels, report_config=report_vrms_function_parameters)

    modality_dict_lt = kde_modality(vrms_arr_lt, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_lt = tail_fraction_and_trimmed_skew_two_sided(vrms_arr_lt, all_channels, skewness_config=skewness_function_parameters)
    if len(vrms_arr_lt[1]) < 100:
        logger.warning(f"LT trigger has less than 100 valid RMS entries ({len(vrms_arr_lt[1])}). Results for the Vrms statistics may be unreliable.")
    
    modality_lt, tail_label_lt = report_vrms_characteristics(modality_dict_lt, tail_dict_lt, all_channels, report_config=report_vrms_function_parameters)
    plot_vrms_values_against_time(times, vrms_arr, all_channels, station_id, run_label, save_location, force_mask, radiant0_mask, radiant1_mask, lt_mask, n_rows=12, n_cols=2, day_interval=day_interval, use_monitoring=use_monitoring)
    plot_vrms_values_against_time_per_trigger(times, vrms_arr, all_channels, station_id, run_label, save_location, force_mask, radiant0_mask, radiant1_mask, lt_mask, n_rows=12, n_cols=2, day_interval=day_interval, use_monitoring=use_monitoring)

    # Write detailed Vrms modality results to text files for each trigger type
    write_vrms_modality_results(modality_force, tail_label_force, trigger_label="FORCE", station_id=station_id, run_label=run_label, results_dir=RESULTS_DIR, use_monitoring=use_monitoring)
    write_vrms_modality_results(modality_radiant0, tail_label_radiant0, trigger_label="RADIANT0", station_id=station_id, run_label=run_label, results_dir=RESULTS_DIR, use_monitoring=use_monitoring)
    write_vrms_modality_results(modality_radiant1, tail_label_radiant1, trigger_label="RADIANT1", station_id=station_id, run_label=run_label, results_dir=RESULTS_DIR, use_monitoring=use_monitoring)
    write_vrms_modality_results(modality_lt, tail_label_lt, trigger_label="LT", station_id=station_id, run_label=run_label, results_dir=RESULTS_DIR, use_monitoring=use_monitoring)

    # The Vrms statistics can be misleading (especially for low event number) so the debugging plots are always generated
    debug_plot_vrms_distribution(vrms_arr_force, modality_dict_force, channel_list=all_channels, station_id=station_id, run_label=run_label, trigger_label="FORCE", save_location=save_location, n_rows=12, n_cols=2, use_monitoring=use_monitoring)
    debug_plot_vrms_distribution(vrms_arr_radiant0, modality_dict_radiant0, channel_list=all_channels, station_id=station_id, run_label=run_label, trigger_label="RADIANT0", save_location=save_location, n_rows=12, n_cols=2, use_monitoring=use_monitoring)
    debug_plot_vrms_distribution(vrms_arr_radiant1, modality_dict_radiant1, channel_list=all_channels, station_id=station_id, run_label=run_label, trigger_label="RADIANT1", save_location=save_location, n_rows=12, n_cols=2, use_monitoring=use_monitoring)
    debug_plot_vrms_distribution(vrms_arr_lt, modality_dict_lt, channel_list=all_channels, station_id=station_id, run_label=run_label, trigger_label="LT", save_location=save_location, n_rows=12, n_cols=2, use_monitoring=use_monitoring)
    
    ## Vrms stability
    reference_filename_rms = f"expected_{rms_label}/expected_{rms_label}_station{station_id}.json"
    vrms_k_values, vrms_ref_mean, vrms_ref_std = load_values_json(REFERENCE_DIR, reference_filename_rms)
    z_score_arr_vrms_force = calculate_z_score_parameter(vrms_arr_force, vrms_ref_mean, vrms_ref_std, all_channels)
    flag_outliers_vrms_force = outlier_flag(z_score_arr_vrms_force, vrms_k_values, all_channels)
    outlier_details_vrms_force = find_outlier_details(z_score_arr_vrms_force, vrms_k_values, flag_outliers_vrms_force, channel_list=all_channels, run_no=run_no_force, event_number=event_number_force)
    write_vrms_outlier_details(outlier_details_vrms_force, station_id, run_label, trigger_label="FORCE", n_events=n_events_force, results_dir=RESULTS_DIR, use_monitoring=use_monitoring)
    plot_vrms_values_against_time_single_trigger_zscore(times_force, vrms_arr_force, flag_outliers_vrms_force, z_score_arr_vrms_force, vrms_k_values, trigger_name = "FORCE", channel_list = all_channels, station_id=station_id, run_label=run_label, save_location=save_location, n_rows=12, n_cols=2, day_interval=day_interval, use_monitoring=use_monitoring)

    rms_arr_per_run_dict_force = get_rms_per_run(vrms_arr_force, run_no_force)
    relative_median_shift_results = relative_median_shift(rms_arr_per_run_dict_force, all_channels)
    with open(os.path.join(RESULTS_DIR, f"{rms_label}_relative_median_shift_results_force_trigger_station{station_id}_{run_label}.json"), "w") as f:
        json.dump(relative_median_shift_results, f, indent=4)
    create_heatmap_plot(relative_median_shift_results, label = "Relative Median Shift", save_dir = PLOTS_DIR, channel_list=all_channels, station_id = station_id,matrix_key = "median_shift_matrix", run_label=run_label, cmap="Reds")
    rms_results = decision_metric(outlier_details_vrms_force, relative_median_shift_results, n_events_force=n_events_force, channels=all_channels)
    with open(os.path.join(RESULTS_DIR, f"rms_stability_decision_results_force_trigger_station{station_id}_{run_label}.json"), "w") as f:
        json.dump(rms_results, f, indent=4)
    
    ##### Glitching and block offset analysis - only for RADIANT digitizer type
    if digitizer_type:
        ##### Glitching analysis
        logger.info("Starting glitching analysis...")

        # Load the configuration parameters for the glitching analysis from the JSON file
        glitching_config_json = os.path.join(CONFIG_DIR, "config_glitching.json")
        with open(glitching_config_json, "r") as f:
            glitching_config_dict = json.load(f)

        config_glitching = glitching_config_dict["config_glitching_values"]
        
        glitch_info = binomtest_glitch_fraction(glitch_arr, all_channels, config_glitching=config_glitching)
        write_glitching_results(glitch_info, station_id, run_label, all_channels, results_dir = RESULTS_DIR)
        
        glitching_violin_plot(glitch_arr, all_channels, station_id, run_label, save_location)
        plot_glitch_q99_over_time(np.array(times), glitch_arr, all_channels, station_id, run_label, save_location)
        
        ##### Block offsets analysis
        # Get the reference block offset results for the station

        ref_block_offset_results_file = os.path.join(REFERENCE_DIR, "expected_block_offsets",f"expected_block_offsets_station{station_id}.json")
        with open(ref_block_offset_results_file, "r") as f:
            ref_block_offset_results = json.load(f)

        logger.info("Starting block offset analysis (monitoring.root), results are not used to determine channel health, see warnings in the log file for channels with potential block offset issues. The block offsets are then removed.")
        block_offset_arr_force = get_force_block_offsets_monitoring(block_offsets_arr, force_mask)
        block_offset_stats = block_offset_statistics_monitoring(block_offset_arr_force=block_offset_arr_force, channel_list=all_channels)
        
        block_offset_results_dict = write_block_offset_results(block_offset_stats, station_id, run_label, ref_block_off_dict = ref_block_offset_results, results_dir = RESULTS_DIR, use_monitoring=use_monitoring)
        plot_block_offsets_violin_monitoring(block_offset_arr_force, all_channels, station_id, run_label, save_location)

    # Trigger rate plots
    logger.info("Plotting trigger rates over time and heatmap of trigger rates for different trigger types...")
    plot_trigger_rates_over_time(run_trigger_rates, save_location, station_id, run_label)
    plot_trigger_rate_heatmap(run_trigger_rates, save_location, station_id, run_label)
    
    # Debug plots
    if args.debug_plot:   
        debug_plot_ratios(ratio_arr_dict=ratio_arr_dict, channels_order=channels_order, save_location=save_location, station_id=station_id, run_label=run_label, bins=30,)
        debug_plot_snr_distribution(log_snr_arr, channel_list=all_channels, save_location=save_location, station_id=station_id, run_label=run_label, bins=30)
        debug_plot_z_score_snr(z_score_arr_log_snr, channel_list=all_channels, save_location=save_location, station_id=station_id, run_label=run_label, bins=30)  

        plot_snr_against_time_per_trigger(station_id, times_radiant0, snr_arr_radiant0, all_channels, save_location, run_label, nrows=12, ncols=2, day_interval=day_interval, color = "tab:orange", triggerlabel="RADIANT0")
        plot_snr_against_time_per_trigger(station_id, times_radiant1, snr_arr_radiant1, all_channels, save_location, run_label, nrows=12, ncols=2, day_interval=day_interval, color = "tab:green", triggerlabel="RADIANT1")
        plot_snr_against_time_per_trigger(station_id, times_lt, snr_arr_lt, all_channels, save_location, run_label, nrows=12, ncols=2, day_interval=day_interval, color = "tab:red", triggerlabel="LT")
 
        
    # Create summary CSV file
    if digitizer_type == "radiant":
        create_result_csv_file(
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
            CSV_DIR,
            rms_label
        )

    elif digitizer_type == "didaq":
        create_result_csv_file_didaq(
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
                    CSV_DIR,
                    rms_label
                )
    
