from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
import rnog_data.runtable as rt
from NuRadioReco.detector import detector
import logging
import os
import datetime
import numpy as np
from matplotlib import pyplot as plt
from NuRadioReco.utilities import units
from tqdm import tqdm
from argparse import ArgumentParser
import warnings
import pandas as pd
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import NuRadioReco.framework.trigger
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
from cycler import cycler
import matplotlib as mpl
#from cmap import Colormap
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from science_verification_analysis import convert_events_information, choose_trigger_type, station_channels, read_rnog_runtable, read_rnog_data, calculate_statistics_log_snr, calculate_z_score_snr, outlier_flag, find_outlier_details, print_outlier_summary, choose_day_interval, plot_snr_against_time
import json
import matplotlib.dates as mdates
from datetime import timezone


'''This module can be used to find expected parameter values for RNO-G stations from a known stable time period.'''

# Channel mapping for the first seven RNO-G stations:
SURFACE_CHANNELS_LIST = [12, 13, 14, 15, 16, 17, 18 , 19 , 20]
DEEP_CHANNELS_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]
UPWARD_CHANNELS_LIST = [13, 16, 19]
DOWNWARD_CHANNELS_LIST = [12, 14, 15, 17, 18, 20]
VPOL_LIST = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
HPOL_LIST = [4, 8, 11, 21]
PHASED_ARRAY_LIST = [0, 1, 2, 3]
ALL_CHANNELS_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 , 19 , 20, 21, 22, 23]

# Channel mapping for Station 14:
STATION_14_SURFACE_CHANNELS_LIST = [12, 13, 14, 15, 16, 17, 18 , 19]
STATION_14_DEEP_CHANNELS_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23]
STATION_14_UPWARD_CHANNELS_LIST = [13, 15, 16, 18]
STATION_14_DOWNWARD_CHANNELS_LIST = [12, 14, 17, 19]
STATION_14_VPOL_LIST = [0, 1, 2, 3, 5, 6, 7, 9, 10, 20, 22, 23]
STATION_14_HPOL_LIST = [4, 8, 11, 21]
STATION_14_PHASED_ARRAY_LIST = [0, 1, 2, 3]
STATION_14_ALL_CHANNELS_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 , 19 , 20, 21, 22, 23]

# Script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Matplotlib settings

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 17,

    'axes.labelsize': 17,
    'axes.titlesize': 18,
    'axes.linewidth': 1.2,
    'axes.grid': False,

    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,

    'lines.linewidth': 1.6,
    'lines.antialiased': True,
    'lines.markersize': 6,

    'legend.fontsize': 14,
    'legend.frameon': False,
    'legend.handlelength': 1,
    'legend.borderpad': 0.3,

    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def find_k_value(z_score_log, channel_list, quantile=0.999):
    '''Find the k-value corresponding to the given reference z-score array and quantile.'''
    k_ch_list = {}
    for ch in channel_list:
        z_score_ch = z_score_log[ch]
        k_ch = np.quantile(np.abs(z_score_ch), quantile)
        k_ch_list[ch] = k_ch
    return k_ch_list

def save_values_json(values_log, filename):
    '''Save the reference values as a JSON file.'''
    if not os.path.isabs(filename):
        filepath = os.path.join(SCRIPT_DIR, filename)
    else:
        filepath = filename

    with open(filepath, "w") as f:
        json.dump(values_log, f, indent=4)

    print(f"Values saved to {filepath}")


if __name__ == "__main__":

    argparser = ArgumentParser(description="RNO-G Science Verification Analysis")
    argparser.add_argument("-st", "--station_id", type=int, required=True, help="Station to analyze, e.g --station_id 14")
    argparser.add_argument("-b", "--backend", type=str, default="pyroot", help="Backend to use for reading data, should be either pyroot or uproot (default: pyroot), e.g. --backend pyroot or --backend uproot")
    argparser.add_argument("-sl", "--save_location", type=str, default=".", help="Location to save the output plots (default: current directory), e.g. --save_location /path/to/save/plots")
    argparser.add_argument("-ex", "--exclude-runs", nargs="+", type=int, default=[], metavar="RUN", help="Run number(s) to exclude, e.g. --exclude-runs 1005 1010")
    argparser.add_argument("--save_values", action="store_true", help="If set, will save the reference values to separate JSON files.")

    run_selection = argparser.add_mutually_exclusive_group(required=True)
    run_selection.add_argument("--runs", nargs="+", type=int, metavar="RUN_NUMBERS",
                           help="Run number(s) to analyze. Each run number should be given explicitly separated by a space, e.g. --runs 1001 1002 1005")
    run_selection.add_argument("--run_range", nargs=2, type=int, metavar=("START_RUN", "END_RUN"),
                            help="Range of run numbers to analyze (inclusive). Provide start and end run numbers separated by a space, e.g. --run_range 1000 1050")
    run_selection.add_argument("--time_range", nargs=2, type=str, metavar=("START_DATE", "END_DATE"),
                            help="Date range to analyze (inclusive). Provide start and end dates separated by a space in YYYY-MM-DD format, e.g. --time_range 2024-07-15 2024-09-30")


    surface_channels, deep_channels, upward_channels, downward_channels, vpol_channels, hpol_channels, phased_array_channels, all_channels = station_channels(argparser.parse_args().station_id)

    args = argparser.parse_args()

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


    # Create save location directory if it doesn't exist
    save_location = os.path.expanduser(args.save_location)
    os.makedirs(save_location, exist_ok=True)

    spec_arr, trace_arr, times_trace_arr, snr_arr, run_no, times, freqs, event_info = read_rnog_data(station_id, run_numbers, backend=backend) 

    # Choose FORCE triggered events only
    force_mask = choose_trigger_type(event_info, "FORCE")
    snr_arr_force = snr_arr[:, force_mask]
    event_info_force = {key: np.array(value)[force_mask] for key, value in event_info.items()}
    times = np.array(times)
    times_force = times[force_mask]

    log_snr_arr, log_mean_list, log_median_list, log_std_list, log_difference_list = calculate_statistics_log_snr(snr_arr_force)

    z_score_arr_log_snr = calculate_z_score_snr(log_snr_arr, log_mean_list, log_std_list, all_channels)
    k_values_log_snr = find_k_value(z_score_arr_log_snr, all_channels, quantile=0.999)

    k_values_filename_snr = f"station_{station_id}_k_ref_values_snr.json"

    if args.save_values:
        save_values_json(k_values_log_snr, k_values_filename_snr)
        save_values_json(log_mean_list, f"station_{station_id}_ref_log_mean_snr.json")
        save_values_json(log_std_list, f"station_{station_id}_ref_log_std_snr.json")

    flag_outliers_snr = outlier_flag(z_score_arr_log_snr, k_values_log_snr, all_channels)

    outlier_details_snr = find_outlier_details(z_score_arr_log_snr, k_values_log_snr, flag_outliers_snr, event_info_force, all_channels)
    print_outlier_summary(outlier_details_snr)

    day_interval = choose_day_interval(times)
    plot_snr_against_time(station_id, times_force, snr_arr_force, flag_outliers_snr, z_score_arr_log_snr, k_values_log_snr, all_channels, nrows=12, ncols=2, day_interval=day_interval)