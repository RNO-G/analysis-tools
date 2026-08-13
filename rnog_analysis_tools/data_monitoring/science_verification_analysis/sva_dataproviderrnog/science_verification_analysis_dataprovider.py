'''
This module is an outdated version of the science verification analysis. Instead of using the monitoring data, it uses dataProviderRNOG() to extract information from full waveforms.
It only reads events from the combined.root files, which contain only a subset of the full dataset. Data reading is much slower than the monitoring data. 
It is kept here for archival purposes and can be used if there are no monitoring.root files available for the chosen dataset. The analysis won't be updated anymore and might contain different/outdated methods than the current version.
!!!! Reference value files does not exist for this method, so the analysis will not be able to calculate z-scores and outlier flags for SNR and Vrms. Please first calculate expected values using the scripts under /analysis-tools/rnog_analysis_tools/data_monitoring/science_verification_analysis/expected_values/outdated and change the file paths!!!!
!!!! RMS analysis is done in ADC units, which is not correct for this analysis. The results will be wrong and should not be used. Please use the monitoring.root files for the RMS analysis. !!!!
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
from NuRadioReco.utilities import units
import sys

#### Script directory 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

#### Output directories for plots, results, and logs
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "detailed_results")
CSV_DIR = os.path.join(SCRIPT_DIR, "channel_health_summary")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

REFERENCE_DIR = os.path.join(PARENT_DIR, "expected_values", "outdated")
CONFIG_DIR = os.path.join(PARENT_DIR, "config_files_sva")

# Create output directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Import config files
from config_files_sva.config_plotting import set_plot_style

# Import analysis functions
from read_rnog_data_nuradio import convert_events_information, read_rnog_data
from monitoring_data_functions_sva.get_monitoring_data_uproot import choose_trigger_type_header, read_multiple_runs
from analysis_functions_sva.spectral_analysis_sva import normalize_channels, find_amplitude_ratio_in_band, find_amplitude_ratio_in_band_specific_bkg, excess_info_from_ratio, excess_info_from_ratio_specific_bkg, validate_excess_in_bands
from analysis_functions_sva.z_score_analysis_sva import calculate_statistics_log_paramater, calculate_z_score_parameter, symmetry_metrics_channel_z_score, symmetry_metrics_z_score, load_values_json, outlier_flag, find_outlier_details, calculate_expected_values_per_trigger, outlier_details
from analysis_functions_sva.vrms_analysis_sva import calculate_vrms, kde_modality, tail_fraction_and_trimmed_skew_two_sided, report_vrms_characteristics, get_rms_per_trigger_monitoring
from analysis_functions_sva.glitching_analysis_sva import binomtest_glitch_fraction
from analysis_functions_sva.block_offsets_analysis_sva_dataproviderrnog import get_block_offsets_after_removal, get_block_offsets_before_removal, plot_block_offsets_violin_before_after_comparison, block_offset_statistics
from analysis_functions_sva.block_offsets_analysis_sva_monitoring import get_force_block_offsets_monitoring, block_offset_statistics_monitoring, plot_block_offsets_violin_monitoring
from analysis_functions_sva.vrms_stability_analysis_sva import get_rms_per_run, relative_median_shift, decision_metric

# Import helper functions
from helper_functions.output_writer import write_failed_runs_to_csv, write_spectral_results, write_snr_outlier_details, write_vrms_outlier_details, write_vrms_modality_results, write_glitching_results, write_block_offset_results, create_result_csv_file 
from helper_functions.read_rnog_runtable import read_rnog_runtable
from helper_functions.config_helper import get_station_config

# Import plotting functions
from plotting_functions_sva.plotting_sva_spectrum import plot_time_integrated_surface_spectra_unnormalized, plot_time_integrated_surface_spectra_normalized, plot_time_integrated_deep_spectra, plot_time_integrated_surface_spectra_normalized_example_reference
from plotting_functions_sva.plotting_sva_snr import choose_day_interval, plot_snr_against_time, plot_snr_against_time_per_trigger
from plotting_functions_sva.plotting_sva_vrms import plot_vrms_values_against_time, plot_vrms_values_against_time_single_trigger_zscore, create_heatmap_plot, plot_vrms_values_against_time_per_trigger
from plotting_functions_sva.plotting_sva_glitch import glitching_violin_plot, choose_bin_size, plot_glitch_q99_over_time
from plotting_functions_sva.plotting_sva_debug import debug_plot_ratios, debug_plot_snr_distribution, debug_plot_z_score_snr, debug_plot_vrms_distribution
from plotting_functions_sva.plotting_sva_trigger_rate import plot_trigger_rates_over_time, plot_trigger_rate_heatmap


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

#### Choose events based on trigger type for dataProviderRNOG case
def choose_trigger_type(event_info, trigger_type: str):
    '''Choose events based on trigger type.'''
    mask = event_info["triggerType"] == trigger_type

    return mask


if __name__ == "__main__":

    argparser = ArgumentParser(description="RNO-G Science Verification Analysis using dataProviderRNOG() !!!! Outdated, please use science_verification_analysis_main.py !!!!")
    
    argparser.add_argument("-st", "--station_id", type=int, required=True, help="Station to analyze, e.g --station_id 14")
    argparser.add_argument("-b", "--backend", type=str, default="pyroot", help="!!! Only needed for method 'monitoring' !!!. Backend to use for reading data, should be either pyroot or uproot (default: pyroot), e.g. --backend pyroot or --backend uproot")
    argparser.add_argument("-sl", "--save_location", type=str, default=PLOTS_DIR, help="Location to save the output plots (default: plots directory under script directory), e.g. --save_location /path/to/save/plots")
    argparser.add_argument("-ex", "--exclude-runs", nargs="+", type=int, default=[], metavar="RUN", help="Run number(s) to exclude, e.g. --exclude-runs 1005 1010")
    argparser.add_argument("--debug_plot", action="store_true", help="If set, will create debug plots.")
    argparser.add_argument("--sampling_rate", type=str, default= "after_2024", choices=["before_2024", "after_2024"], help="!!! Only needed for method 'monitoring' !!!. Sampling rate to use, choices are 'before_2024' (3.2 GHz) and 'after_2024' (2.4 GHz), default is 'after_2024'.")

    run_selection = argparser.add_mutually_exclusive_group(required=True)
    run_selection.add_argument("--runs", nargs="+", type=int, metavar="RUN_NUMBERS",
                           help="Run number(s) to analyze. Each run number should be given explicitly separated by a space, e.g. --runs 1001 1002 1005")
    run_selection.add_argument("--run_range", nargs=2, type=int, metavar=("START_RUN", "END_RUN"),
                            help="Range of run numbers to analyze (inclusive). Provide start and end run numbers separated by a space, e.g. --run_range 1000 1050")
    run_selection.add_argument("--time_range", nargs=2, type=str, metavar=("START_DATE", "END_DATE"),
                            help="Date range to analyze (inclusive). Provide start and end dates separated by a space in YYYY-MM-DD format, e.g. --time_range 2024-07-15 2024-09-30")


    args = argparser.parse_args()

    use_monitoring = False

    logger.info("Using dataProviderRNOG method to read data !!!! Outdated, please use science_verification_analysis_main.py !!!!")
    rms_label = "vrms"

    station_id = args.station_id
    backend = args.backend
    if backend not in ["pyroot", "uproot"]:
        raise ValueError("Backend should be either 'pyroot' or 'uproot'")
   
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

    sampling_rate_choice = args.sampling_rate
    sampling_rate = {"after_2024": 2.4*units.GHz,
                 "before_2024": 3.2*units.GHz}
    sr = sampling_rate[sampling_rate_choice]

    # Start logging
    setup_logging(station_id, run_label)
    logger.info(f"Starting analysis for station {station_id}, runs: {run_numbers}, backend: {backend}, sampling rate: {sr}")
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

    base_data_path = "/pnfs/ifh.de/acs/radio/diskonly/data/inbox/"

    # Read data 
    spec_arr, trace_arr, times_trace_arr, snr_arr, run_no, times, freqs, event_info, glitch_arr, block_offsets_arr = read_rnog_data(station_id, run_numbers, backend=backend, sampling_rate=sr) 
    
    # Spectral analysis configuration parameters
    spectral_analysis_config_json = os.path.join(CONFIG_DIR, "config_spectral_analysis.json")
    with open(spectral_analysis_config_json, "r") as f:
        spectral_analysis_config_dict = json.load(f)

    spectral_bands = spectral_analysis_config_dict["spectral_bands"]
    alpha_spec = spectral_analysis_config_dict["alpha_spec"]
    ci_threshold_spec = spectral_analysis_config_dict["ci_threshold_spec"]
    normalization_band = spectral_analysis_config_dict["normalization_band"]
    log_ratio_thresholds_spec = spectral_analysis_config_dict["log_ratio_thresholds_spec"]

    # Normalize surface channel spectra
    norm_spec_arr, scale_factors = normalize_channels(spec_arr, freqs, downward_channels, upward_channels, normalization_band=normalization_band)
    logger.debug(f"Event info trigger type: {event_info['triggerType']}")
    logger.debug(f"Spec arr shape: {spec_arr.shape}, Norm spec arr shape: {norm_spec_arr.shape}")

    # Select FORCE trigger events
    force_mask = choose_trigger_type(event_info, "FORCE")

    ###### Spectral analysis for FORCE trigger events only
    spec_arr_force = spec_arr[:, force_mask, :]
    norm_spec_arr_force = norm_spec_arr[:, force_mask, :]
    logger.info(f"Number of FORCE trigger events: {spec_arr_force.shape[1]}")

    if len(spec_arr_force[1]) < 30:
        logger.warning("Less than 30 FORCE-trigger events, results of the sign test may not be reliable.")
    lt_mask = choose_trigger_type(event_info, "LT")
    spec_arr_lt = spec_arr[:, lt_mask, :]

    radiant0_mask = choose_trigger_type(event_info, "RADIANT0")
    spec_arr_radiant0 = spec_arr[:, radiant0_mask, :]

    radiant1_mask = choose_trigger_type(event_info, "RADIANT1")
    spec_arr_radiant1 = spec_arr[:, radiant1_mask, :]  
    run_event_counts = None # Not available when reading with dataProviderRNOG, only with monitoring data, used for spectral plotting
    run_no_force = event_info["run"][force_mask]
    event_number_force = event_info["eventNumber"][force_mask]
    n_events_force = spec_arr_force.shape[1]
    
    failed_run_info = {}
    excluded_runs = args.exclude_runs.copy() if args.exclude_runs else []
    if excluded_runs:
        logger.info(f"Excluding runs {excluded_runs} from the analysis as specified by the user.")
        for excluded_run in excluded_runs:
            failed_run_info[int(excluded_run)] = "Run excluded by user"

    if failed_run_info:
        write_failed_runs_to_csv(station_id, failed_run_info, run_label, results_dir=RESULTS_DIR)

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

    # Write detailed spectral results for all channels in a single write (dCache/pnfs is write-once)
    write_spectral_results(all_excess_info, surface_channels, station_id, run_label, results_dir=RESULTS_DIR)

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

    snr_arr_radiant0 = snr_arr[:, radiant0_mask]
    snr_arr_radiant1 = snr_arr[:, radiant1_mask]
    snr_arr_lt = snr_arr[:, lt_mask]

    times_radiant0 = times[radiant0_mask]
    times_radiant1 = times[radiant1_mask]
    times_lt = times[lt_mask]

    plot_snr_against_time_per_trigger(station_id, times_radiant0, snr_arr_radiant0, all_channels, save_location, run_label, nrows=12, ncols=2, day_interval=day_interval, color = "tab:orange", triggerlabel="RADIANT0")
    plot_snr_against_time_per_trigger(station_id, times_radiant1, snr_arr_radiant1, all_channels, save_location, run_label, nrows=12, ncols=2, day_interval=day_interval, color = "tab:green", triggerlabel="RADIANT1")
    plot_snr_against_time_per_trigger(station_id, times_lt, snr_arr_lt, all_channels, save_location, run_label, nrows=12, ncols=2, day_interval=day_interval, color = "tab:red", triggerlabel="LT")
   
    ##### Vrms analysis 
    rms_config_json = os.path.join(CONFIG_DIR, "config_rms.json") #### !!!! In ADC, wrong for this analysis !!!!
    with open(rms_config_json, "r") as f:
        rms_config_dict = json.load(f)
    
    kde_modality_function_parameters = rms_config_dict["kde_modality_function_parameters"]
    skewness_function_parameters = rms_config_dict["skewness_function_parameters"]
    report_vrms_function_parameters = rms_config_dict["report_vrms_function_parameters"]

    logger.info("Starting Vrms analysis for data read with dataProviderRNOG...")
    vrms_arr, vrms_arr_force, vrms_arr_radiant0, vrms_arr_radiant1, vrms_arr_lt = calculate_vrms(trace_arr, event_info)

    logger.info(f"Number of RADIANT0 trigger events: {len(vrms_arr_radiant0[1])}, Number of RADIANT1 trigger events: {len(vrms_arr_radiant1[1])}, Number of LT trigger events: {len(vrms_arr_lt[1])}")
    
    logger.info(f"Calculating RMS (for monitoring.root) or Vrms (for dataProviderRNOG) modality and tail characteristics for each trigger type...")
    modality_dict_force = kde_modality(vrms_arr_force, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_force = tail_fraction_and_trimmed_skew_two_sided(vrms_arr_force, all_channels, skewness_config=skewness_function_parameters)
    if len(vrms_arr_force[1]) < 100:
        logger.warning(f"FORCE trigger has less than 100 valid RMS (for monitoring.root) or Vrms (for dataProviderRNOG) entries ({len(vrms_arr_force[1])}). Results for the Vrms statistics may be unreliable.")
    modality_force, tail_label_force = report_vrms_characteristics(modality_dict_force, tail_dict_force, all_channels, report_config=report_vrms_function_parameters)

    modality_dict_radiant0 = kde_modality(vrms_arr_radiant0, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_radiant0 = tail_fraction_and_trimmed_skew_two_sided(vrms_arr_radiant0, all_channels, skewness_config=skewness_function_parameters)
    if len(vrms_arr_radiant0[1]) < 100:
        logger.warning(f"RADIANT0 trigger has less than 100 valid RMS (for monitoring.root) or Vrms (for dataProviderRNOG) entries ({len(vrms_arr_radiant0[1])}). Results for the Vrms statistics may be unreliable.")
    modality_radiant0, tail_label_radiant0 = report_vrms_characteristics(modality_dict_radiant0, tail_dict_radiant0, all_channels, report_config=report_vrms_function_parameters)

    modality_dict_radiant1 = kde_modality(vrms_arr_radiant1, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_radiant1 = tail_fraction_and_trimmed_skew_two_sided(vrms_arr_radiant1, all_channels, skewness_config=skewness_function_parameters)
    if len(vrms_arr_radiant1[1]) < 100:
        logger.warning(f"RADIANT1 trigger has less than 100 valid RMS (for monitoring.root) or Vrms (for dataProviderRNOG) entries ({len(vrms_arr_radiant1[1])}). Results for the Vrms statistics may be unreliable.")
    modality_radiant1, tail_label_radiant1 = report_vrms_characteristics(modality_dict_radiant1, tail_dict_radiant1, all_channels, report_config=report_vrms_function_parameters)

    modality_dict_lt = kde_modality(vrms_arr_lt, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_lt = tail_fraction_and_trimmed_skew_two_sided(vrms_arr_lt, all_channels, skewness_config=skewness_function_parameters)
    if len(vrms_arr_lt[1]) < 100:
        logger.warning(f"LT trigger has less than 100 valid RMS (for monitoring.root) or Vrms (for dataProviderRNOG) entries ({len(vrms_arr_lt[1])}). Results for the Vrms statistics may be unreliable.")
    
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
    
    # Vrms stability
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
    
    ##### Glitching analysis - Same for both monitoring and dataProviderRNOG 
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

    ##### Block offsets - dataProviderRNOG
    
    logger.info("Starting block offset analysis (dataProviderRNOG), results are not used to determine channel health, see warnings in the log file for channels with potential block offset issues. The block offsets are then removed.")
    fit_block_offsets_before = get_block_offsets_before_removal(block_offsets_arr, event_info, all_channels)
    fit_block_offsets_after = get_block_offsets_after_removal(trace_arr, event_info, all_channels, sampling_rate=sr)

    block_offset_stats = block_offset_statistics(fit_block_offsets_before, fit_block_offsets_after, all_channels)
    block_offset_results_dict = write_block_offset_results(block_offset_stats, station_id, run_label, ref_block_off_dict = ref_block_offset_results, results_dir = RESULTS_DIR, use_monitoring=use_monitoring)
    plot_block_offsets_violin_before_after_comparison(fit_block_offsets_before, fit_block_offsets_after, all_channels, station_id, run_label, save_location)

    # Debug plots
    if args.debug_plot:   
        debug_plot_ratios(ratio_arr_dict=ratio_arr_dict, channels_order=channels_order, save_location=save_location, station_id=station_id, run_label=run_label, bins=30,)
        debug_plot_snr_distribution(log_snr_arr, channel_list=all_channels, save_location=save_location, station_id=station_id, run_label=run_label, bins=30)
        debug_plot_z_score_snr(z_score_arr_log_snr, channel_list=all_channels, save_location=save_location, station_id=station_id, run_label=run_label, bins=30)   
        
    # Create summary CSV file
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
    

