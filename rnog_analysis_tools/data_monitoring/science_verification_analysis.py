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

import matplotlib as mpl
#from cmap import Colormap
from collections import defaultdict

'''
This module can be used to test if the stations are working as expected.
'''

# Channel mapping for the first seven RNO-G stations:
SURFACE_CHANNELS_LIST = [12, 13, 14, 15, 16, 17, 18 , 19 , 20]
DEEP_CHANNELS_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]
UPWARD_CHANNELS_LIST = [13, 16, 19]
DOWNWARD_CHANNELS_LIST = [12, 14, 15, 17, 18]
ALL_CHANNELS_LIST = SURFACE_CHANNELS_LIST + DEEP_CHANNELS_LIST

# Channel mapping for Station 14:
STATION_14_SURFACE_CHANNELS_LIST = [12, 13, 14, 15, 16, 17, 18 , 19]
STATION_14_DEEP_CHANNELS_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23]
STATION_14_UPWARD_CHANNELS_LIST = [13, 15, 16, 18]
STATION_14_DOWNWARD_CHANNELS_LIST = [12, 14, 17, 19]
STATION_14_ALL_CHANNELS_LIST = STATION_14_SURFACE_CHANNELS_LIST + STATION_14_DEEP_CHANNELS_LIST

# Matplotlib settings
# cm = Colormap('tol:muted')
# colors = cm.colors

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 17,

    'axes.labelsize': 17,
    'axes.titlesize': 18,
    'axes.linewidth': 1.2,
    'axes.grid': False,

 #   'axes.prop_cycle': cycler(colors=colors),

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
    'legend.handlelength': 2.2,
    'legend.borderpad': 0.3,

    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def convert_events_information(event_info, convert_to_arrays=True):

    data = defaultdict(list)

    for ele in event_info.values():
        for k, v in ele.items():
            data[k].append(v)

    if convert_to_arrays:
        for k in data:
            data[k] = np.array(data[k])

    return data

# Find the correct channel mapping for a given station
def station_channels(station_id: int):
    if station_id == 14:
        return STATION_14_SURFACE_CHANNELS_LIST, STATION_14_DEEP_CHANNELS_LIST, STATION_14_UPWARD_CHANNELS_LIST, STATION_14_DOWNWARD_CHANNELS_LIST, STATION_14_ALL_CHANNELS_LIST
    else:
        return SURFACE_CHANNELS_LIST, DEEP_CHANNELS_LIST, UPWARD_CHANNELS_LIST, DOWNWARD_CHANNELS_LIST, ALL_CHANNELS_LIST

# Normalize surface channel spectra to the average of the down channels in the reference frequency band (500-650 MHz)
def normalize_surface_channels_to_down_reference(spec_arr, frequencies, down_channels, up_channels, f_low=500*units.MHz, f_high=650*units.MHz):
    '''Normalize all surface channel spectra to the average of the down channels in the reference frequency band.'''
    # spec_arr shape: (n_channels, n_events, n_freqs)
    freq_mask = (frequencies >= f_low) & (frequencies <= f_high)
    spec_arr = np.copy(spec_arr)

    # Use only down channels to define the reference
    down = spec_arr[down_channels]
    down_band_avg = np.mean(down[:, :, freq_mask], axis=2)  # (n_down, n_events)
    ref_band_avg = np.mean(down_band_avg, axis=0)           # (n_events,)

    all_surface_channels = up_channels + down_channels
    all_surface_spectra = spec_arr[all_surface_channels]

    ch_band_avg = np.mean(all_surface_spectra[:, :, freq_mask], axis=2)  # (n_ch, n_events)
    scale_factors = ref_band_avg[np.newaxis, :] / ch_band_avg    # (n_ch, n_events)

    all_surface_spectra_norm = all_surface_spectra * scale_factors[:, :, np.newaxis]
    spec_arr[all_surface_channels] = all_surface_spectra_norm

    return spec_arr, scale_factors

def read_rnog_runtable(station_id: int, start_time: str, stop_time: str):
    '''Get run numbers from the runtable tool for a given station and time range.'''
    RunTable = rt.RunTable()
    testrt = RunTable.get_table( start_time=start_time, stop_time=stop_time, stations=[station_id], run_types = ['physics'])
    return testrt

def read_rnog_data(station_id: int, run_numbers: list, backend: str = "pyroot"):
    '''Read RNO-G data for a given station and list of run numbers using the specified backend.'''
    file_list = [ f"/pnfs/ifh.de/acs/radio/diskonly/data/inbox/station14/run{run_id}/combined.root" for run_id in run_numbers]
    n_files = len(file_list)
    n_batches = n_files // 100 + 1
    print(f"Reading {n_files} files in {n_batches} batches using {backend} backend.")

    event_info = defaultdict(list)

    n_events_total = 0
    spec_batches = []
    trace_batches = []
    times_trace_batches = []
    snr_batches = []
    run_no_all = []
    times_all = []
    freqs = None

    from NuRadioReco.modules.channelSignalReconstructor import channelSignalReconstructor as csr

    for batch in np.array_split(np.array(file_list), n_batches):
        tableReader = dataProviderRNOG()
        tableReader.begin(files=batch.tolist(), 
                          det=None,
                          reader_kwargs={"overwrite_sampling_rate":2.4*units.GHz, 
                                         "convert_to_voltage":False,
                                         "apply_baseline_correction":"auto",
                                         "mattak_kwargs":{"backend":backend}})
        event_info_tmp = tableReader.reader.get_events_information(
            keys=["triggerType", "triggerTime", "readoutTime", "radiantThrs", "lowTrigThrs"])
        
        event_info_tmp = convert_events_information(event_info_tmp, False)
        for key, value in event_info_tmp.items():
            event_info[key] += value

        n_events = tableReader.reader.get_n_events()
        n_events_total += n_events

        channel_list = [i for i in range(24)]  
        spec_arr = np.zeros((len(channel_list), n_events, 1025))
        trace_arr = np.zeros((len(channel_list), n_events, 2048))
        times_trace_arr = np.zeros((len(channel_list), n_events, 2048))
        snr_arr = np.zeros((len(channel_list), n_events))

        run_no = []
        times = []
        event_ids = []

        csr = csr()
        csr.begin(debug=False)
        
        for idx, event in enumerate(tableReader.reader.run()):
            station = event.get_station()
            time = station.get_station_time().datetime64
            times.append(time)
            run_no.append(event.get_run_number())

            csr.run(evt=event, station=station, det=None, stored_noise=False)
            for i_ch, ch in enumerate(channel_list):
                channel = station.get_channel(ch)

                times_ch = channel.get_times()
                times_trace_arr[i_ch, idx, :] = times_ch

                snr_dict = channel.get_parameter(chp.SNR)
                snr_peak = snr_dict["peak_amplitude"]
                snr_arr[i_ch, idx] = snr_peak
                
                spec = channel.get_frequency_spectrum()
                spec_arr[i_ch, idx, :] = np.abs(spec)

                trace = channel.get_trace()
                trace_arr[i_ch, idx, :] = trace

                if freqs is None and idx == 0 and i_ch == 0:
                    freqs = channel.get_frequencies()
        
        spec_batches.append(spec_arr)
        trace_batches.append(trace_arr)
        times_trace_batches.append(times_trace_arr)
        snr_batches.append(snr_arr)
        run_no_all.extend(run_no)
        times_all.extend(times)

    spec_arr = np.concatenate(spec_batches, axis=1)
    trace_arr = np.concatenate(trace_batches, axis=1)
    times_trace_arr = np.concatenate(times_trace_batches, axis=1)
    snr_arr = np.concatenate(snr_batches, axis=1)

    run_no = np.array(run_no_all)
    times = np.array(times_all)     

    for key, value in event_info.items():
        event_info[key] = np.array(value)

    inf_mask = np.isinf(event_info["triggerTime"])
    event_info["triggerTime"][inf_mask] = event_info["readoutTime"][inf_mask]
    print(f"Found {np.sum(inf_mask)} events with inf trigger time (of {len(inf_mask)} events)")

    #print(f"n_events read: {spec_arr.shape[1]}, n_events_total: {n_events_total}")
    #print(f"freqs shape: {freqs.shape}, spec_arr shape: {spec_arr.shape}, trace_arr shape: {trace_arr.shape}, times_trace_arr shape: {times_trace_arr.shape}, snr_arr shape: {snr_arr.shape}")
    #print(f"freqs {freqs}")
    #print(f"trigger types: {np.unique(event_info['triggerType'])}")

if __name__ == "__main__":

    argparser = ArgumentParser(description="RNO-G Science Verification Analysis")
    argparser.add_argument("-st", "--station_id", type=int, required=True, help="Station to analyze, e.g --station_id 14")
    argparser.add_argument("-b", "--backend", type=str, default="pyroot", help="Backend to use for reading data, should be either pyroot or uproot (default: pyroot), e.g. --backend pyroot or --backend uproot")
    
    run_selection = argparser.add_mutually_exclusive_group(required=True)
    run_selection.add_argument("--runs", nargs="+", type=int, metavar="RUN_NUMBERS",
                           help="Run number(s) to analyze. Each run number should be given explicitly separated by a space, e.g. --runs 1001 1002 1005")
    run_selection.add_argument("--run_range", nargs=2, type=int, metavar=("START_RUN", "END_RUN"),
                            help="Range of run numbers to analyze (inclusive). Provide start and end run numbers separated by a space, e.g. --run_range 1000 1050")
    run_selection.add_argument("--time_range", nargs=2, type=str, metavar=("START_DATE", "END_DATE"),
                            help="Date range to analyze (inclusive). Provide start and end dates separated by a space in YYYY-MM-DD format, e.g. --time_range 2024-07-15 2024-09-30")


    surface_channels, deep_channels, upward_channels, downward_channels, all_channels = station_channels(argparser.parse_args().station_id)

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
        start_time = args.time_range[0]
        stop_time = args.time_range[1]
        runtable = read_rnog_runtable(station_id, start_time, stop_time)
        run_numbers = runtable['run_number'].tolist()

    read_rnog_data(station_id, run_numbers, backend=backend)

    

