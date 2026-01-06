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
from NuRadioReco.modules.channelSignalReconstructor import channelSignalReconstructor as csr
from cycler import cycler
import matplotlib as mpl
from cmap import Colormap

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
cm = Colormap('tol:muted')
colors = cm.colors

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 17,

    'axes.labelsize': 17,
    'axes.titlesize': 18,
    'axes.linewidth': 1.2,
    'axes.grid': False,

    'axes.prop_cycle': cycler(colors=colors),

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

# Find the correct channel mapping for a given station
def station_channels(station_id: int):
    if station_id == 14:
        return STATION_14_SURFACE_CHANNELS_LIST, STATION_14_DEEP_CHANNELS_LIST, STATION_14_UPWARD_CHANNELS_LIST, STATION_14_DOWNWARD_CHANNELS_LIST, STATION_14_ALL_CHANNELS_LIST
    else:
        return SURFACE_CHANNELS_LIST, DEEP_CHANNELS_LIST, UPWARD_CHANNELS_LIST, DOWNWARD_CHANNELS_LIST, ALL_CHANNELS_LIST

def normalize_all_to_down_reference(spec_arr, frequencies, down_channels, up_channels, f_low, f_high):
    freq_mask = (frequencies >= f_low) & (frequencies <= f_high)
    spec_arr = np.copy(spec_arr)

    # Use only down channels to define the reference
    down = spec_arr[down_channels]
    down_band_avg = np.mean(down[:, :, freq_mask], axis=2)  # (n_down, n_events)
    ref_band_avg = np.mean(down_band_avg, axis=0)           # (n_events,)

    all_channels = up_channels + down_channels
    all_spectra = spec_arr[all_channels]

    ch_band_avg = np.mean(all_spectra[:, :, freq_mask], axis=2)  # (n_ch, n_events)
    scale_factors = ref_band_avg[np.newaxis, :] / ch_band_avg    # (n_ch, n_events)

    all_spectra_norm = all_spectra * scale_factors[:, :, np.newaxis]
    spec_arr[all_channels] = all_spectra_norm

    return spec_arr, scale_factors
    

if __name__ == "__main__":

    argparser = ArgumentParser(description="RNO-G Science Verification Analysis")
    argparser.add_argument("-st", "--station_id", type=int, required=True, help="Station to analyze")
    
    run_selection = argparser.add_mutually_exclusive_group(required=True)
    run_selection.add_argument("--runs", nargs="+", type=int, metavar="RUN_NUMBERS", 
                               help="Run number(s) to analyze. Each run number should be given explicitly separated by a space, e.g. --runs 1001 1002 1005")
    run_selection.add_argument("--run_range", nargs=2, type=int, metavar=("START_RUN", "END_RUN"),
                               help="Range of run numbers to analyze (inclusive). Provide start and end run numbers separated by a space, e.g. --run_range 1000 1050")
    run_selection.add_argument("--time_range", nargs=2, type=int, metavar=("START_TIME", "END_TIME"),
                               help="Range of time numbers to analyze (inclusive). Provide start and end time numbers separated by a space, e.g. --time_range 2024-07-15 2024-09-30")

    surface_channels, deep_channels, upward_channels, downward_channels, all_channels = station_channels(argparser.parse_args().station_id)


