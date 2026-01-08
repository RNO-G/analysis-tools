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
from scipy.stats import skew
import json
import matplotlib.dates as mdates
from datetime import timezone


'''
This module can be used to test if the stations are working as expected.
'''

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

# Matplotlib settings

COLORS = [
    "#1F77B4",  # blue
    "#FF4B4B",  # bright red
    "#2CA02C",  # green
    "#9467BD",  # purple
    "#17BECF",  # cyan
    "#D62728",  # deep red
    "#8C564B",  # warm brown
    "#1FA187",  # teal-green
    "#4E79A7",  # steel blue
    "#F76E5C",  # coral
    "#A55194",  # magenta-purple
    "#3D4ED7",  # indigo
]

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 17,

    'axes.labelsize': 17,
    'axes.titlesize': 18,
    'axes.linewidth': 1.2,
    'axes.grid': False,

    'axes.prop_cycle': cycler('color', COLORS),

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

def convert_events_information(event_info, convert_to_arrays=True):

    data = defaultdict(list)

    for ele in event_info.values():
        for k, v in ele.items():
            data[k].append(v)

    if convert_to_arrays:
        for k in data:
            data[k] = np.array(data[k])

    return data

def choose_trigger_type(event_info, trigger_type: str):
    '''Choose events based on trigger type.'''
    mask = event_info["triggerType"] == trigger_type

    return mask
    
# Find the correct channel mapping for a given station
def station_channels(station_id: int):
    if station_id == 14:
        return STATION_14_SURFACE_CHANNELS_LIST, STATION_14_DEEP_CHANNELS_LIST, STATION_14_UPWARD_CHANNELS_LIST, STATION_14_DOWNWARD_CHANNELS_LIST, STATION_14_VPOL_LIST, STATION_14_HPOL_LIST, STATION_14_PHASED_ARRAY_LIST, STATION_14_ALL_CHANNELS_LIST
    else:
        return SURFACE_CHANNELS_LIST, DEEP_CHANNELS_LIST, UPWARD_CHANNELS_LIST, DOWNWARD_CHANNELS_LIST, VPOL_LIST, HPOL_LIST, PHASED_ARRAY_LIST, ALL_CHANNELS_LIST

# Normalize surface channel spectra to the average of the down channels in the reference frequency band (500-650 MHz)
def normalize_channels(spec_arr, frequencies, down_channels, up_channels, f_low=500*units.MHz, f_high=650*units.MHz):
    '''Normalize all surface channel spectra to the average of the down channels in the reference frequency band. The reference band is 500-650 MHz by default. This will be replaced by lab measurements in the future.'''
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
    file_list = []
    valid_run_numbers = []
    missing_runs = []

    for run_id in run_numbers:
        path = f"/pnfs/ifh.de/acs/radio/diskonly/data/inbox/station{station_id}/run{run_id}/combined.root"
        if os.path.isfile(path):
            file_list.append(path)
            valid_run_numbers.append(run_id)
        else:
            missing_runs.append(run_id)

    if missing_runs:
        print(f"!!!! Skipping {len(missing_runs)} missing runs: {missing_runs} !!!!")
    if not file_list:
        raise FileNotFoundError("No combined.root files found for selected runs.")
    
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
    csr = csr()
    csr.begin(debug=False)

    for batch in tqdm(np.array_split(np.array(file_list), n_batches), desc="Reading batches", unit="batch"):
        tableReader = dataProviderRNOG()
        tableReader.begin(files=batch.tolist(), 
                          det=None,
                          reader_kwargs={"overwrite_sampling_rate":2.4, 
                                         "convert_to_voltage":True,
                                         "apply_baseline_correction":"auto",
                                         "mattak_kwargs":{"backend":backend}})
        event_info_tmp = tableReader.reader.get_events_information(
            keys=["triggerType", "triggerTime", "readoutTime", "radiantThrs", "lowTrigThrs", "run", "eventNumber"])
        
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
        
        for idx, event in enumerate(tqdm(tableReader.reader.run(), total=n_events, desc="Events", unit="evt", leave=False)):
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

    return spec_arr, trace_arr, times_trace_arr, snr_arr, run_no, times, freqs, event_info

    #print(f"n_events read: {spec_arr.shape[1]}, n_events_total: {n_events_total}")
    #print(f"freqs shape: {freqs.shape}, spec_arr shape: {spec_arr.shape}, trace_arr shape: {trace_arr.shape}, times_trace_arr shape: {times_trace_arr.shape}, snr_arr shape: {snr_arr.shape}")
    #print(f"freqs {freqs}")
    #print(f"trigger types: {np.unique(event_info['triggerType'])}")

def find_amplitude_ratio_in_band(station_id, freqs, norm_spec_arr, upward_channels, downward_channels, reference_channels, freq_min, freq_max):
    '''Find normalized amplitude ratio of upward vs downward channels in a given frequency band. This will be replaced by lab measurements in the future.'''
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    ratio_list = []
    spectrum_list = []

    ref_specs = np.stack([norm_spec_arr[ch][:, freq_mask] for ch in reference_channels], axis=0)
    ref_spectra_per_ch = np.median(ref_specs, axis=2)   # (n_ref_ch, n_events)
    ref_spec = np.median(ref_spectra_per_ch, axis=0) # (n_events,)

    for ch in upward_channels + downward_channels:
        masked_spec = norm_spec_arr[ch][:, freq_mask]        # (n_events, n_freqs)
        masked_spec_med = np.median(masked_spec, axis=1)       # (n_events,)

        spec_ratio = masked_spec_med / ref_spec
        ratio_list.append(spec_ratio)
        spectrum_list.append(masked_spec_med)

    return ratio_list, spectrum_list


def find_amplitude_ratio_in_band_specific_bkg(station_id, freqs, norm_spec_arr, upward_channels, downward_channels):
    '''Find normalized amplitude ratio of upward vs downward channels in specific frequency bands. Backgrouns were defined using wiki page: https://radio.uchicago.edu/wiki/index.php/Features_observed_in_data'''

    # Galactic excess frequency band
    if station_id == 14:
        reference_channels_gal = [12, 14, 19]
    else:
        reference_channels_gal = downward_channels
    freq_min_gal = 80 * units.MHz
    freq_max_gal = 120 * units.MHz
    ratio_arr_gal, _ = find_amplitude_ratio_in_band(station_id, freqs, norm_spec_arr, upward_channels, downward_channels, reference_channels_gal, freq_min_gal, freq_max_gal)

    # 360-380 MHz frequency band
    reference_channels_360 = downward_channels
    freq_min_360 = 360 * units.MHz
    freq_max_360 = 380 * units.MHz
    ratio_arr_360, _ = find_amplitude_ratio_in_band(station_id, freqs, norm_spec_arr, upward_channels, downward_channels, reference_channels_360, freq_min_360, freq_max_360)

    # 482-485 MHz frequency band
    reference_channels_482 = downward_channels
    freq_min_482 = 482 * units.MHz
    freq_max_485 = 485 * units.MHz
    ratio_arr_482, _ = find_amplitude_ratio_in_band(station_id, freqs, norm_spec_arr, upward_channels, downward_channels, reference_channels_482, freq_min_482, freq_max_485)

    reference_channels_240 = downward_channels
    freq_min_240 = 240 * units.MHz
    freq_max_272 = 272 * units.MHz
    ratio_arr_240, _ = find_amplitude_ratio_in_band(station_id, freqs, norm_spec_arr, upward_channels, downward_channels, reference_channels_240, freq_min_240, freq_max_272)

    return ratio_arr_gal, ratio_arr_360, ratio_arr_482, ratio_arr_240

def excess_info_from_ratio(ratio_arr, alpha=0.01, freq_range_str=""):
    '''Calculate excess information from amplitude ratios in frequency bands.'''
    from scipy.stats import binomtest

    log_ratio = np.log10(np.asarray(ratio_arr))
    median_log_ratio = np.median(log_ratio)
    mean_log_ratio = np.mean(log_ratio)
    frac_pos = np.mean(log_ratio > 0)/np.mean(log_ratio < 0) if np.mean(log_ratio < 0) != 0 else np.inf

    k = np.sum(log_ratio > 0)
    n = int(log_ratio.size)
    result = binomtest(k, n, p=0.5, alternative="greater")
    pval = result.pvalue
    statistic = result.statistic
    confidence_interval = result.proportion_ci(confidence_level=0.95)

    print(f"Binomial test for frequency range {freq_range_str}: k={k}, n={n}, pval={pval:.3e}, statistic={statistic:.3f}, 95% CI={confidence_interval}")

    if pval > alpha:
        validation = "NO EXCESS (sign-test)"
    else:
        if confidence_interval.low > 0.75:
            validation = "STRONG EXCESS (sign-test)"
        elif confidence_interval.low > 0.55:
            validation = "MODERATE EXCESS (sign-test)"
        else: 
            validation = "WEAK EXCESS (sign-test)"       

    return {
        "median_log_ratio": median_log_ratio,
        "mean_log_ratio": mean_log_ratio,
        "frac_pos": frac_pos,
        "pval": pval,
        "validation": validation
    }

def validate_excess_in_bands(ratio_arr_gal, ratio_arr_360, ratio_arr_482, ratio_arr_240, alpha=0.01):
    '''Validate excess information from amplitude ratios in frequency bands.'''
    gal_info = excess_info_from_ratio(ratio_arr_gal, alpha, "80–120 MHz")
    freq360_info = excess_info_from_ratio(ratio_arr_360, alpha, "360–380 MHz")
    freq482_info = excess_info_from_ratio(ratio_arr_482, alpha, "482–485 MHz")
    freq240_info = excess_info_from_ratio(ratio_arr_240, alpha, "240–272 MHz")
    return {
        "galactic_excess": gal_info,
        "freq_360_380MHz": freq360_info,
        "freq_482_485MHz": freq482_info,
        "freq_240_272MHz": freq240_info
    }

def debug_plot_ratios(ratio_arr_gal, ratio_arr_360, ratio_arr_482, ratio_arr_240, channels_order, save_location, station_id, run_label, bins=30):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    bands = [("80–120 MHz  (FORCE Trigger)", ratio_arr_gal),
             ("360–380 MHz  (FORCE Trigger)", ratio_arr_360),
             ("482–485 MHz  (FORCE Trigger)", ratio_arr_482),
             ("240–272 MHz  (FORCE Trigger)", ratio_arr_240)]

    for ax, (title, ratio_list) in zip(axes, bands):
        for ch, r in zip(channels_order, ratio_list):
            r = np.asarray(r)
            ax.hist(np.log10(r), bins=bins, histtype="step", linewidth=1.3, label=f"Ch {ch}", alpha=0.8)

        ax.set_title(title)
        ax.set_xlabel("log10(R)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Counts")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=8, frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(save_location,f"debug_amplitude_ratios_force_trigger_{station_id}_{run_label}.pdf",))
    plt.close(fig)


def plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr, freqs, upward_channels, downward_channels, save_location, run_label):
    '''Plot time-integrated surface channel spectra.'''
    plt.figure(figsize=(10, 6))
    for ch in upward_channels:
        spec_mean = np.mean(spec_arr[ch, :, :], axis=0)
        plt.plot(freqs / units.MHz, spec_mean, label=f'Ch {ch} (up)', linestyle='-')
    for ch in downward_channels:
        spec_mean = np.mean(spec_arr[ch, :, :], axis=0)
        plt.plot(freqs / units.MHz, spec_mean, label=f'Ch {ch} (down)', linestyle='--')

    plt.xlabel('Frequency [MHz]')
    plt.xlim(0, 800)
    plt.ylabel('Amplitude Spectrum [V/GHz]')
    plt.title('Time-Integrated Spectrum of Surface Channels  (FORCE Trigger)')
    plt.legend(loc="upper right", 
               frameon=True,
               fancybox=True,
               framealpha=0.9,
               edgecolor="black")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, f"time_integrated_surface_spectra_unnormalized_force_trigger_{station_id}_{run_label}.pdf"))


def plot_time_integrated_surface_spectra_normalized(station_id, norm_spec_arr, freqs, upward_channels, downward_channels, save_location, run_label):
    '''Plot time-integrated surface channel spectra.'''
    plt.figure(figsize=(10, 6))
    for ch in upward_channels:
        spec_mean = np.mean(norm_spec_arr[ch, :, :], axis=0)
        plt.plot(freqs / units.MHz, spec_mean, label=f'Ch {ch} (up)', linestyle='-')
    for ch in downward_channels:
        spec_mean = np.mean(norm_spec_arr[ch, :, :], axis=0)
        plt.plot(freqs / units.MHz, spec_mean, label=f'Ch {ch} (down)', linestyle='--')

    periodiccolor2 = "mediumseagreen"
    excesscolor = 'grey'
    wb_color = "mediumvioletred"
    normcolor = "steelblue"

    plt.axvspan(80, 120, color=excesscolor, alpha=0.3, label="_nolegend_")
    plt.axvline(x=0.403e3, color=wb_color, linestyle='--', linewidth=1.2, label="_nolegend_", alpha=0.7)
    plt.axvspan(0.278e3, 0.285e3, color=periodiccolor2, alpha=0.3, label="_nolegend_")
    plt.axvspan(0.482e3, 0.485e3, color=periodiccolor2, alpha=0.3, label="_nolegend_")
    plt.axvspan(0.240e3, 0.272e3, color=periodiccolor2, alpha=0.3, label="_nolegend_")
    plt.axvspan(0.360e3, 0.380e3, color=periodiccolor2, alpha=0.3, label="_nolegend_")
    plt.axvspan(0.136e3, 0.139e3, color=periodiccolor2, alpha=0.3, label="_nolegend_")
    plt.axvspan(0.151e3, 0.157e3, color=periodiccolor2, alpha=0.3, label="_nolegend_")
    plt.axvspan(0.125e3, 0.127e3, color=periodiccolor2, alpha=0.3, label="_nolegend_")
    plt.axvspan(500, 650, color=normcolor, alpha=0.3, label="_nolegend_")

    plt.xlabel('Frequency [MHz]')
    plt.xlim(0, 800)
    plt.ylabel('Amplitude Spectrum [V/GHz]')
    plt.title('Time-Integrated Spectrum of Surface Channels (FORCE Trigger)')
    
    ax = plt.gca()
    line_legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black")
    
    annotation_handles = [
        Patch(facecolor=excesscolor, alpha=0.3, label="Galactic Excess"),
        Line2D([0], [0], color=wb_color, linestyle="--", linewidth=1.2, label="Weather Balloon"),
        Patch(facecolor=periodiccolor2, alpha=0.3, label="Periodic Signal"),
        Patch(facecolor=normcolor, alpha=0.3, label="Normalization Region"),]

    annotation_legend = ax.legend(
        handles=annotation_handles,
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black")

    ax.add_artist(line_legend)

    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, f"time_integrated_surface_spectra_normalized_force_trigger_{station_id}_{run_label}.pdf"))

def plot_time_integrated_deep_spectra(station_id, spec_arr, freqs, vpol_channels, hpol_channels, save_location, run_label):
    '''Plot time-integrated deep channel spectra.'''
    plt.figure(figsize=(10, 6))
    for ch in vpol_channels:
        spec_mean = np.mean(spec_arr[ch, :, :], axis=0)
        plt.plot(freqs / units.MHz, spec_mean, label=f'Ch {ch} (VPOL)', linestyle='-')
    for ch in hpol_channels:
        spec_mean = np.mean(spec_arr[ch, :, :], axis=0)
        plt.plot(freqs / units.MHz, spec_mean, label=f'Ch {ch} (HPOL)', linestyle='--')

    plt.xlabel('Frequency [MHz]')
    plt.xlim(0, 800)
    plt.ylabel('Amplitude Spectrum [V/GHz]')
    plt.title('Time-Integrated Spectrum of Deep Channels')
    plt.legend(loc="upper right", 
               frameon=True,
               fancybox=True,
               framealpha=0.9,
               edgecolor="black")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, f"time_integrated_deep_spectra_unnormalized_force_trigger_{station_id}_{run_label}.pdf"))


def calculate_statistics_log_snr(snr_arr):
    '''Calculate log10 statistics (mean, median, std, mean-median) for SNR values.'''
    log_snr_arr = np.zeros((len(snr_arr), len(snr_arr[0])))
    for ch in range(len(snr_arr)):
        snr_arr_ch = snr_arr[ch]
        log_snr_arr_ch = np.log10(snr_arr_ch)
        log_snr_arr[ch, :] = log_snr_arr_ch
    log_mean_list = []
    log_median_list = []
    log_std_list = []
    log_difference_list = []

    for ch in range(len(log_snr_arr)):
        log_mean_list.append(np.mean(log_snr_arr[ch]))
        log_median_list.append(np.median(log_snr_arr[ch]))
        log_std_list.append(np.std(log_snr_arr[ch]))
        log_difference_list.append(np.mean(log_snr_arr[ch]) - np.median(log_snr_arr[ch]))

    return log_snr_arr, log_mean_list, log_median_list, log_std_list, log_difference_list

def calculate_z_score_snr(snr_arr, mean_list, std_list, channel_list):
    '''Calculate the z-score for SNR values given mean and standard deviation lists for each channel.'''
    z_score_arr = np.zeros((len(snr_arr), len(snr_arr[0])))
    for ch in channel_list:
        snr_arr_ch = snr_arr[ch]
        mean_ch = mean_list[ch]
        std_ch = std_list[ch]
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

def load_k_values_json(filepath):
    ''' Load k-values from a JSON file.'''
    with open(filepath, "r") as f:
        data = json.load(f)     
    k_values = {int(ch): float(k) for ch, k in data.items()}

    return k_values

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


def print_outlier_summary(outlier_details):
    '''Print a summary of outlier events for each channel.'''
    for ch in sorted(outlier_details.keys()):
        entries = outlier_details[ch]
        n_outliers = len(entries)

        if n_outliers == 0:
            print(f"Channel {ch}: 0 outliers")
            continue

        k_ch = entries[0]["k"]
        print(f"Channel {ch}: {n_outliers} outliers (k = {k_ch:.2f})")

        for e in entries:
            print(f"  - run {e['run']}, event {e['eventNumber']}, "
                  f"|z| = {e['z_abs']:.2f} (delta = {e['z_minus_k']:.2f} above k)")



if __name__ == "__main__":

    argparser = ArgumentParser(description="RNO-G Science Verification Analysis")
    argparser.add_argument("-st", "--station_id", type=int, required=True, help="Station to analyze, e.g --station_id 14")
    argparser.add_argument("-b", "--backend", type=str, default="pyroot", help="Backend to use for reading data, should be either pyroot or uproot (default: pyroot), e.g. --backend pyroot or --backend uproot")
    argparser.add_argument("-sl", "--save_location", type=str, default=".", help="Location to save the output plots (default: current directory), e.g. --save_location /path/to/save/plots")
    argparser.add_argument("-ex", "--exclude-runs", nargs="+", type=int, default=[], metavar="RUN", help="Run number(s) to exclude, e.g. --exclude-runs 1005 1010")
    argparser.add_argument("--debug_plot", action="store_true", help="If set, will create debug plots for amplitude ratios in frequency bands.")

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

    # Read data
    spec_arr, trace_arr, times_trace_arr, snr_arr, run_no, times, freqs, event_info = read_rnog_data(station_id, run_numbers, backend=backend) 

    # Normalize surface channel spectra
    norm_spec_arr, scale_factors = normalize_channels(spec_arr, freqs, downward_channels, upward_channels)
    print(f"Event info trigger type: {event_info['triggerType']}, event info trigger type length: {len(event_info['triggerType'])}")
    print(f"Spec arr shape: {spec_arr.shape}, Norm spec arr shape: {norm_spec_arr.shape}")

    # Select FORCE trigger events
    force_mask = choose_trigger_type(event_info, "FORCE")

    # Spectral analysis for FORCE trigger events only
    spec_arr_force = spec_arr[:, force_mask, :]
    norm_spec_arr_force = norm_spec_arr[:, force_mask, :]
    print(f"Number of FORCE trigger events: {spec_arr_force.shape[1]}")

    if len(spec_arr_force[1]) < 30:
        print("!!!! Warning: Less than 30 FORCE-trigger events, results of the sign test may not be reliable. !!!!")

    ratio_arr_gal, ratio_arr_360, ratio_arr_482, ratio_arr_240 = find_amplitude_ratio_in_band_specific_bkg(station_id, freqs, norm_spec_arr_force, upward_channels, downward_channels)
    channels_order = upward_channels + downward_channels
    ch_to_idx = {ch: i for i, ch in enumerate(channels_order)}
    print(ch_to_idx)

    for ch in surface_channels:
        print(f"\nAnalyzing channel {ch}")
        i = ch_to_idx[ch]
        print(i)

        validation_results = validate_excess_in_bands(
            ratio_arr_gal[i],
            ratio_arr_360[i],
            ratio_arr_482[i],
            ratio_arr_240[i],
            alpha=0.01
        )

        for band, results in validation_results.items():
            print(f"=== {band} ===")
            for key, value in results.items():
                print(f"{key}: {value}")
            print("==============")

    plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr_force, freqs, upward_channels, downward_channels, save_location, run_label)
    plot_time_integrated_surface_spectra_normalized(station_id, norm_spec_arr_force, freqs, upward_channels, downward_channels, save_location, run_label)

    # SNR analysis
    snr_arr_force = snr_arr[:, force_mask]
    event_info_force = {key: np.array(value)[force_mask] for key, value in event_info.items()}

    log_snr_arr, log_mean_list, log_median_list, log_std_list, log_difference_list = calculate_statistics_log_snr(snr_arr_force)

    z_score_arr_log_snr = calculate_z_score_snr(log_snr_arr, log_mean_list, log_std_list, all_channels)
    k_values_filename_snr = f"station_{station_id}_k_ref_values_snr.json"
    k_values_log_snr = load_k_values_json(k_values_filename_snr)
    flag_outliers_snr = outlier_flag(z_score_arr_log_snr, k_values_log_snr, all_channels)

    outlier_details_snr = find_outlier_details(z_score_arr_log_snr, k_values_log_snr, flag_outliers_snr, event_info_force, all_channels)
    print_outlier_summary(outlier_details_snr)


    if args.debug_plot:   
        debug_plot_ratios(ratio_arr_gal, ratio_arr_360, ratio_arr_482, ratio_arr_240, channels_order=channels_order, save_location=save_location, station_id=station_id, run_label=run_label, bins=30,)