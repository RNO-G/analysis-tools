'''
This module can be used to test if the stations are working as expected.
'''

import rnog_data.runtable as rt
import logging
import os
import datetime
import numpy as np
from matplotlib import pyplot as plt
from NuRadioReco.utilities import units
from argparse import ArgumentParser
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from datetime import timezone
from matplotlib import colors, cm
from NuRadioReco.modules.RNO_G.channelBlockOffsetFitter import fit_block_offsets
from config_files_sva.config_station import get_station_config, sampling_rate
from config_files_sva.config_plotting import set_plot_style
from read_rnog_data_nuradio import convert_events_information, read_rnog_data
from analysis_functions_sva.spectral_analysis_sva import find_amplitude_ratio_in_band, find_amplitude_ratio_in_band_specific_bkg, excess_info_from_ratio, excess_info_from_ratio_specific_bkg, validate_excess_in_bands
from config_files_sva.config_spectral_analysis import SPECTRAL_BANDS, ALPHA_SPEC, CI_THRESHOLDS_SPEC, NORMALIZATION_BAND
import copy
from analysis_functions_sva.snr_analysis_sva import calculate_statistics_log_snr, calculate_z_score_snr, symmetry_metrics_channel_z_score, symmetry_metrics_z_score, load_values_json, outlier_flag, find_outlier_details
from analysis_functions_sva.vrms_analysis_sva import calculate_vrms, kde_modality, tail_fraction_and_trimmed_skew_two_sided, report_vrms_characteristics
from config_files_sva.config_vrms import kde_modality_function_parameters, skewness_function_parameters, report_vrms_function_parameters
from analysis_functions_sva.glitching_analysis_sva import binomtest_glitch_fraction
from config_files_sva.config_glitching import config_glitching_values
from analysis_functions_sva.block_offsets_analysis_sva import get_block_offsets_after_removal, get_block_offsets_before_removal, plot_block_offsets_violin_before_after_comparison, block_offset_statistics
from config_files_sva.config_block_offsets import block_offset_limits

#### Script directory for json files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

#### Output directories for plots, results, and logs
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "detailed_results")
CSV_DIR = os.path.join(SCRIPT_DIR, "channel_health_summary")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

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

#### Choose events based on trigger type
def choose_trigger_type(event_info, trigger_type: str):
    '''Choose events based on trigger type.'''
    mask = event_info["triggerType"] == trigger_type

    return mask

#### Normalize surface channel spectra to the average of the down channels in the reference frequency band (500-650 MHz)
def normalize_channels(spec_arr, frequencies, down_channels, up_channels):
    f_low=NORMALIZATION_BAND["freq_min"]
    f_high=NORMALIZATION_BAND["freq_max"]
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

def plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr, freqs, upward_channels, downward_channels, save_location, run_label, trigger_label):
    '''Plot time-integrated surface channel spectra.'''
    plt.figure(figsize=(10, 6))
    for ch in upward_channels:
        spec_mean = np.mean(spec_arr[ch, :, :], axis=0)
        plt.plot(freqs / units.MHz, spec_mean, label=f'Ch {ch} (up)', linestyle='-')
    for ch in downward_channels:
        spec_mean = np.mean(spec_arr[ch, :, :], axis=0)
        plt.plot(freqs / units.MHz, spec_mean, label=f'Ch {ch} (down)', linestyle='--')

    plt.xlabel('Frequency [MHz]')
    plt.xlim(50, 800)
    plt.ylim(0, 5)
    plt.ylabel('Amplitude Spectrum [V/GHz]')
    plt.title(f'Time-Integrated Spectrum of Surface Channels  ({trigger_label} Trigger)')
    plt.legend(loc="upper right", 
               frameon=True,
               fancybox=True,
               framealpha=0.9,
               edgecolor="black")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, f"{trigger_label}_time_integrated_surface_spectra_unnormalized_{station_id}_{run_label}.pdf"))


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
    plt.xlim(50, 800)
    plt.ylim(0, 5)
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

def plot_time_integrated_deep_spectra(station_id, spec_arr, freqs, vpol_channels, hpol_channels, save_location, run_label, trigger_label):
    '''Plot time-integrated deep channel spectra.'''
    plt.figure(figsize=(10, 6))
    for ch in vpol_channels:
        spec_mean = np.mean(spec_arr[ch, :, :], axis=0)
        plt.plot(freqs / units.MHz, spec_mean, label=f'Ch {ch} (VPOL)', linestyle='-')
    for ch in hpol_channels:
        spec_mean = np.mean(spec_arr[ch, :, :], axis=0)
        plt.plot(freqs / units.MHz, spec_mean, label=f'Ch {ch} (HPOL)', linestyle='--')

    plt.xlabel('Frequency [MHz]')
    plt.xlim(50, 800)
    plt.ylim(0, 5)
    plt.ylabel('Amplitude Spectrum [V/GHz]')
    plt.title(f'Time-Integrated Spectrum of Deep Channels ({trigger_label} Trigger)')
    plt.legend(loc="upper right", 
               frameon=True,
               fancybox=True,
               framealpha=0.9,
               edgecolor="black")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, f"{trigger_label}_time_integrated_deep_spectra_unnormalized_{station_id}_{run_label}.pdf"))

#### SNR analysis functions ####       
def choose_day_interval(times):
    times = pd.to_datetime(times, utc=True)
    total_days = (times.max() - times.min()).days

    if total_days < 10:
        return 1
    elif total_days < 20:
        return 2
    elif total_days < 40:
        return 4
    elif total_days < 80:
        return 7
    elif total_days < 150:
        return 10
    elif total_days < 300:
        return 15
    elif total_days < 600:
        return 30
    else:
        return 60

def plot_snr_against_time(station_id,times,snr_arr,flag,z_log,k_list,channels,save_location,run_label,nrows=12,ncols=2,day_interval=None):
    times = pd.to_datetime(times,utc=True)
    channels = list(channels)
    n_channels = len(channels)

    if day_interval is None:
        day_interval = choose_day_interval(times) 

    fig, axs = plt.subplots(nrows,ncols,figsize=(15,24),sharex=True)
    axs = np.array(axs)

    for idx, ch in enumerate(channels):
        r = idx//ncols
        c = idx%ncols
        ax = axs[r,c]
        good_mask = ~flag[ch]
        ax.scatter(times[good_mask], np.log10(snr_arr[ch][good_mask]), s=8,alpha=0.25, color="gray")
        zex = np.abs(z_log[ch]) - k_list[ch]
        zex = np.clip(zex,0,None)
        sc = ax.scatter(times[flag[ch]], np.log10(snr_arr[ch][flag[ch]]), s=8,c=zex[flag[ch]], cmap="Reds")
        cax = ax.inset_axes([1.02,0.1,0.05,0.8])
        plt.colorbar(sc,cax=cax)
        ax.grid(alpha=0.4)
        ax.text(0.85, 0.95, f"Ch {ch}", transform = ax.transAxes, ha = "left",va = "top", bbox = dict(boxstyle = "round, pad = 0.25", facecolor = "white", alpha = 0.8))
    
    for idx in range(n_channels, nrows*ncols):
        r = idx//ncols
        c = idx%ncols
        axs[r, c].set_visible(False)

    red = plt.cm.Reds(0.6)
    legend_handles = [Line2D([0],[0],marker="o",color="none",markeredgecolor="gray",markerfacecolor="gray",markersize=6,label=r"$|z|\leq k$"),
                    Line2D([0],[0],marker="o",color="none",markeredgecolor=red,markerfacecolor=red,markersize=6,label=r"$|z|>k$")]                 
    axs[0, 0].legend(handles = legend_handles, loc = "upper left")

    ticks_ax = axs[-1,0]
    time_span = (times.max() - times.min()).total_seconds() / 86400.0  # in days

    if time_span < 1:
        # Use 2h ticks if less than 1 day
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    elif time_span < 3:
        # Use 6h ticks if less than 3 days
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    else:
        # Use day ticks otherwise
        ticks_ax.xaxis.set_major_locator(mdates.DayLocator(interval = day_interval))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz = timezone.utc))

    fig.autofmt_xdate()

    plt.subplots_adjust(bottom = 0.11, wspace = 0.38, left=0.08)
    fig.supxlabel("Date [UTC]", x = 0.5, y = 0.06)
    fig.supylabel(r"$\log_{10}(\mathrm{SNR})$", x = 0.02)
    plt.savefig(os.path.join(save_location,f"snr_against_time_{station_id}_{run_label}.pdf"))
            

#### Vrms analysis functions ####
def plot_vrms_values_against_time(times, vrms_arr_all, channel_list, station_id, run_label, save_location, n_rows = 12, n_cols = 2, day_interval=5):
    '''Plot Vrms distributions for different trigger types.'''

    force_mask = event_info["triggerType"] == "FORCE"
    radiant0_mask = event_info["triggerType"] == "RADIANT0"
    radiant1_mask = event_info["triggerType"] == "RADIANT1"
    lt_mask = event_info["triggerType"] == "LT"

    trigger_masks = {"FORCE": force_mask, 
                     "RADIANT0": radiant0_mask,
                     "RADIANT1": radiant1_mask,
                     "LT": lt_mask,}

    trigger_colors = {"FORCE": "tab:blue", 
                      "RADIANT0": "tab:orange",
                      "RADIANT1": "tab:green",
                      "LT": "tab:red",}

    times = np.asarray(times)
    vrms_arr_all = np.asarray(vrms_arr_all)

    n_channels = len(channel_list)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 24), sharex=True, squeeze=False)
    axs = axs.ravel()

    legend_handles = {}

    for idx, ch in enumerate(channel_list):
        ax = axs[idx]
        vrms_ch = vrms_arr_all[ch]

        for trig_name, trig_mask in trigger_masks.items():
            times_trig = times[trig_mask]
            vrms_trig = vrms_ch[trig_mask]

            scatter = ax.scatter(times_trig, vrms_trig, s=8, alpha=0.5, label=trig_name, color=trigger_colors[trig_name])
            if trig_name not in legend_handles:
                legend_handles[trig_name] = scatter
                

        ax.set_title(f"Channel {ch}")
        ax.grid(alpha=0.4)
    
    for j in range(len(channel_list), len(axs)):
        axs[j].set_visible(False)

    ticks_ax = axs[-1]   
    time_span = times.max() - times.min()     
    time_span_days = time_span / np.timedelta64(1, "D")

    if time_span_days < 1:
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    elif time_span_days < 3:
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    else:
        ticks_ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=timezone.utc))

    fig.legend(handles=[legend_handles[k] for k in trigger_masks.keys()], labels=list(trigger_masks.keys()), 
               loc="lower center", ncol=4, frameon=True, markerscale=2, bbox_to_anchor=(0.5, 0.005))
    
    fig.supylabel(r"$V_\mathrm{rms}$ [V]", x=0.02)
    plt.subplots_adjust(bottom = 0.05, wspace = 0.15, left=0.1)
    fig.supxlabel("Date [UTC]", x = 0.5, y = 0.06)
    plt.savefig(os.path.join(save_location, f"vrms_against_time_{station_id}_{run_label}.pdf"))
    plt.close(fig)

#### Trigger analysis (adapted from the plot_trigger() function from analyze_run.py) ####
def compute_radiant_thresholds(event_info, down_channels, up_channels):
    radiant = event_info["radiantThrs"]

    downward = radiant[:, down_channels].mean(axis=1)
    upward   = radiant[:, up_channels].mean(axis=1)
    low_trig = event_info["lowTrigThrs"].mean(axis=1)

    return upward, downward, low_trig

def plot_trigger_rate_with_thresholds(station_id, event_info, down_channels, up_channels, run_label, day_interval, bin_width_initial=300, max_bins=800, save_location=None):

    trigger_times = np.asarray(event_info["triggerTime"])
    readout_times = np.asarray(event_info["readoutTime"])

    run_duration = trigger_times.max() - trigger_times.min()
    run_duration_readout = readout_times.max() - readout_times.min()

    bin_width = bin_width_initial
    nbins = int(run_duration // bin_width)
    if nbins > max_bins:
        bin_width = 3600  # 1 hour
        nbins = int(run_duration // bin_width)

    times = np.array([datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc) for ts in trigger_times])
    time_span = times.max() - times.min()
    time_span_days = time_span.total_seconds() / 86400.0  # convert to days

    fig, ax_rate = plt.subplots(figsize=(12, 6))

    ax_rate.grid(True, which="both", ls="--", lw=0.35, alpha=0.5)

    weights_total = np.full(times.shape[0], 1.0 / bin_width)
    _, bin_edges, _ = ax_rate.hist(times, bins=nbins, weights=weights_total, histtype="step", color="k", label="Total Rate",)

    triggers = np.unique(event_info["triggerType"])
    trigger_colors = {
        "FORCE": "tab:blue",
        "RADIANT0": "tab:orange",
        "RADIANT1": "tab:green",
        "LT": "tab:red",}

    for trigger in triggers:
        mask = event_info["triggerType"] == trigger
        n_mask = mask.sum()
        if n_mask == 0:
            continue

        color = trigger_colors.get(trigger)  

        ax_rate.hist(times[mask], bins=bin_edges, weights=np.full(n_mask, 1.0 / bin_width), histtype="step", lw=1.1, label=str(trigger), color=color,)

    ax_rate.set_ylabel("Trigger Rate [Hz]")
    ax_rate.set_yscale("log")

    if time_span_days < 1:
        # Use 6h ticks if less than 1 day
        ax_rate.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax_rate.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    elif time_span_days < 3:
        # Use 12h ticks if less than 3 days
        ax_rate.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax_rate.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    else:
        # Use day ticks otherwise
        ax_rate.xaxis.set_major_locator(mdates.DayLocator(interval = day_interval))
        ax_rate.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz = timezone.utc))

    ax_rate.tick_params(axis="x", rotation=25)
    ax_rate.set_xlabel("Time (UTC)")

    upward, downward, lt = compute_radiant_thresholds(event_info, down_channels, up_channels)
    scale = 2.5 / 16777215.0  # convert register to Volts

    ax_thr = ax_rate.twinx()

    ax_thr.plot(times, upward * scale, ls="--", lw=2, color="darkmagenta", label="RADIANT Up (avg)",)
    ax_thr.plot(times, downward * scale, ls="--", lw=2, color="darkgreen", label="RADIANT Down (avg)",)
    ax_thr.plot(times, lt * scale, ls="--", lw=2, color="mediumblue", label="LT (avg)",)

    ax_thr.set_ylabel("Threshold [V]")

    h1, l1 = ax_rate.get_legend_handles_labels()
    h2, l2 = ax_thr.get_legend_handles_labels()
    ax_rate.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.1, 1), borderaxespad=0., frameon=True, framealpha=1.0,)

    fig.tight_layout()
    fig.savefig(os.path.join(save_location, f"trigger_rate_with_thresholds_{station_id}_{run_label}.pdf"))

    return fig, ax_rate, ax_thr

#### Glitching analysis functions ####
def glitching_violin_plot(glitch_arr, channel_list, station_id, run_label, save_location):
    data = [glitch_arr[ch] for ch in channel_list]
    means = np.array([np.mean(glitch_arr[ch]) for ch in channel_list])

    fig, ax = plt.subplots(figsize=(12, 10))

    parts = ax.violinplot(data, positions=channel_list, showextrema=True, showmedians=True, vert=False, side="high", widths=1.8,)

    norm = colors.Normalize(vmin=np.min(means), vmax=np.max(means), clip=False)
    sm = cm.ScalarMappable(norm=norm, cmap="Blues")

    for idx, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(sm.to_rgba(means[idx]))
        pc.set_alpha(0.5)
        pc.set_edgecolor("k")

    parts["cmins"].set_linewidth(0.2)
    parts["cmaxes"].set_linewidth(0.2)
    parts["cbars"].set_linewidth(0.5)
    parts["cmins"].set_color("k")
    parts["cmaxes"].set_color("k")
    parts["cbars"].set_color("k")
    parts["cmedians"].set_color("k")
    parts["cmedians"].set_linewidth(1)

    cb = plt.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("Mean test statistics")

    ax.set_xlabel("Glitching test statistics")
    ax.set_ylabel("Channel")
    ax.set_yticks(channel_list)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min(channel_list) - 1, max(channel_list) + 1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_location, f"glitching_violin_plot_{station_id}_{run_label}.pdf"))
    plt.close(fig)

def choose_bin_size(times):
    total_hours = (times.max() - times.min()).total_seconds() / 3600.0
    if total_hours < 12:
        return "30min"
    elif total_hours < 24:
        return "1h"
    elif total_hours < 3 * 24:
        return "2h"
    elif total_hours < 7 * 24:
        return "6h"
    elif total_hours < 30 * 24:
        return "12h"
    else:
        return "24h"

def plot_glitch_q99_over_time(times, glitch_arr, channels, station_id, run_label, save_location):
    times = pd.to_datetime(times)
    df = pd.DataFrame(glitch_arr.T, index=times, columns=channels)

    bin_rule = choose_bin_size(times)
    q99 = df.resample(bin_rule).quantile(0.99)

    fig, ax = plt.subplots(figsize=(12, 10))

    for ch in channels:
        ax.plot(q99.index, q99[ch], marker=".", linestyle="-", label=f"ch {ch}")

    ax.set_xlabel("Date [UTC]")
    ax.set_ylabel(f"99% quantile glitching ts ({bin_rule} bins)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, frameon=True, framealpha=0.9, edgecolor="black")

    plt.tight_layout()
    plt.savefig(os.path.join(save_location, f"glitch_q99_{station_id}_{run_label}.pdf"))
    plt.close(fig)

#### Debug plots ####
def debug_plot_ratios(ratio_arr_dict, channels_order, save_location, station_id, run_label, bins=30):
    n_bands = len(ratio_arr_dict)
    
    fig, axes = plt.subplots(1, n_bands, figsize=(5*n_bands, 5), sharey=True)
    
    if n_bands == 1:
        axes = [axes]

    for ax, (band_name, ratio_list) in zip(axes, ratio_arr_dict.items()):
        for ch, r in zip(channels_order, ratio_list):
            r = np.asarray(r)
            ax.hist(np.log10(r), bins=bins, histtype="step", linewidth=1.3, label=f"Ch {ch}", alpha=0.8)

        ax.set_title(band_name)
        ax.set_xlabel("log10(R)")
        ax.grid(True, alpha=0.3)

        ax.set_title(f"{band_name} (FORCE Trigger)")
        ax.set_xlabel("log10(R)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Counts")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=8, frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(save_location,f"debug_amplitude_ratios_force_trigger_{station_id}_{run_label}.pdf",))
    plt.close(fig)

def debug_plot_snr_distribution(log_snr_arr, channel_list, save_location, station_id, run_label, bins=30):
    fig, ax = plt.subplots(figsize=(10, 6))
    for ch in channel_list:
        log_snr_ch = log_snr_arr[ch]
        ax.hist(log_snr_ch, bins=bins, histtype="step", linewidth=1.3, label=f"Ch {ch}", alpha=0.8)

    ax.set_title("Log10 SNR Distribution (FORCE Trigger)")
    ax.set_xlabel("log10(SNR)")
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=8, frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(save_location,f"debug_snr_distribution_force_trigger_{station_id}_{run_label}.pdf",))
    plt.close(fig)

def debug_plot_z_score_snr(z_score_arr, channel_list, save_location, station_id, run_label, bins=30):
    fig, ax = plt.subplots(figsize=(10, 6))
    for ch in channel_list:
        z_score_ch = z_score_arr[ch]
        ax.hist(z_score_ch, bins=bins, histtype="step", linewidth=1.3, label=f"Ch {ch}", alpha=0.8)

    ax.set_title("Z-Score SNR Distribution (FORCE Trigger)")
    ax.set_xlabel("Z-Score(SNR)")
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=8, frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(save_location,f"debug_z_score_snr_force_trigger_{station_id}_{run_label}.pdf",))
    plt.close(fig)

def debug_plot_vrms_distribution(vrms_arr, modality_dict, channel_list, station_id, run_label, trigger_label, save_location, n_rows=12, n_cols=2):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 36))
    axes = axes.flatten()

    for idx, ch in enumerate(channel_list):
        ax = axes[idx]
        info = modality_dict[ch]
        vrms = vrms_arr[ch]
        vrms = vrms[np.isfinite(vrms)] 
        vrms_grid = info["vrms_grid"]
        kde_values = info["kde_values"]
        peaks = info["peaks"]
        n_peaks = info["n_peaks"]

        if n_peaks == 0:
            modality = "flat/noisy"
        elif n_peaks == 1:
            modality = "unimodal"
        elif n_peaks == 2:
            modality = "bimodal"
        else:
            modality = f"multimodal ({n_peaks})"

        # histogram
        ax.hist(vrms, bins=30, density=True, alpha=0.3, color="gray")
        # kde curve
        ax.plot(vrms_grid, kde_values, color="blue", lw=1.5)
        # peaks
        if len(peaks) > 0:
            ax.plot(vrms_grid[peaks], kde_values[peaks], "ro", markersize=5)

        ax.set_title(f"Ch {ch}: {modality}")
        ax.set_xlabel("Vrms Values [V]")
        ax.set_ylabel("KDE Density")

    for i in range(len(channel_list), n_rows * n_cols):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_location,f"debug_vrms_hist_kde_density_peaks_{station_id}_{run_label}_{trigger_label}.pdf",))

def channel_health(row):
    severity = {"OK": 0, "!!": 1, "X": 2}
    vals = [severity[v] for v in row if v in severity]
    if not vals:
        return "-"
    inv = {0: "OK", 1: "!!", 2: "X"}
    return inv[max(vals)]


def create_result_csv_file(station_id, run_label, n_events_force, surface_channels, downward_channels, upward_channels, all_channels, validation_results, glitch_info, modality_dict_force, modality_dict_lt, 
                           modality_dict_radiant0, modality_dict_radiant1, outlier_details, save_location):
    out_csv_file = os.path.join(CSV_DIR, f"validation_summary_station{station_id}_{run_label}.csv")
    ch_list = list(all_channels)

    spectral_col = []
    glitch_col = []
    modality_force_col = []
    modality_lt_col = []
    modality_radiant0_col = []
    modality_radiant1_col = []
    snr_col = []

    for ch in ch_list:
        df_spec_val = ""
        if ch in surface_channels:
            spectral_validation = None
            vr = validation_results.get(ch, {})
            spectral_validation = vr.get("galactic_excess", {})

            if spectral_validation is None:
                df_spec_val = "?"
            else:
                if ch in downward_channels:
                    if spectral_validation == "NO EXCESS":
                        df_spec_val = "OK"
                    elif spectral_validation == "WEAK EXCESS":
                        df_spec_val = "!!"
                    elif spectral_validation in ["MODERATE EXCESS", "STRONG EXCESS"]:
                        df_spec_val = "X"
                    else:
                        df_spec_val = "?"
                elif ch in upward_channels:
                    if spectral_validation in ["STRONG EXCESS", "MODERATE EXCESS"]:
                        df_spec_val = "OK"
                    elif spectral_validation == "WEAK EXCESS":
                        df_spec_val = "!!"
                    elif spectral_validation == "NO EXCESS":
                        df_spec_val = "X"
                    else:
                        df_spec_val = "?"
                else:
                    df_spec_val = "?"
        else:
            df_spec_val = "-"

        spectral_col.append(df_spec_val)

        # Glitching column
        if glitch_info is None:
            glitch_val = "-"
            glitch_col.append(glitch_val)
        else:
            info = glitch_info.get(ch, None)
            glitch_val_raw = info.get("validation") if info is not None else "-"
            if glitch_val_raw == "NO EXCESSIVE GLITCHING":
                glitch_val = "OK"
            elif glitch_val_raw == "WEAK EXCESSIVE GLITCHING":
                glitch_val = "!!"
            elif glitch_val_raw in ["MODERATE EXCESSIVE GLITCHING", "STRONG EXCESSIVE GLITCHING"]:
                glitch_val = "X"
            else:
                glitch_val = "-"
            glitch_col.append(glitch_val)

        # Vrms analysis column
        if modality_dict_force is None:
            modality_value = "-"
            modality_force_col.append(modality_value)
        else:
            n_peaks = modality_dict_force[ch]["n_peaks"]
            if n_peaks == 0:
                modality_value = "!!"
            elif n_peaks == 1:
                modality_value = "OK"
            elif n_peaks == 2:
                modality_value = "X"
            else:
                modality_value = f"X"
            modality_force_col.append(modality_value)

        if modality_dict_lt is None:
            modality_value = "-"
            modality_lt_col.append(modality_value)
        else:
            n_peaks = modality_dict_lt[ch]["n_peaks"]
            if n_peaks == 0:
                modality_value = "!!"
            elif n_peaks == 1:
                modality_value = "OK"
            elif n_peaks == 2:
                modality_value = "X"
            else:
                modality_value = f"X"
            modality_lt_col.append(modality_value)

        if modality_dict_radiant0 is None:
            modality_value = "-"
            modality_radiant0_col.append(modality_value)
        else:
            n_peaks = modality_dict_radiant0[ch]["n_peaks"]
            if n_peaks == 0:
                modality_value = "!!"
            elif n_peaks == 1:
                modality_value = "OK"
            elif n_peaks == 2:
                modality_value = "X"
            else:
                modality_value = "X"
            modality_radiant0_col.append(modality_value)
        if modality_dict_radiant1 is None:
            modality_value = "-"
            modality_radiant1_col.append(modality_value)
        else:
            n_peaks = modality_dict_radiant1[ch]["n_peaks"]
            if n_peaks == 0:
                modality_value = "!!"
            elif n_peaks == 1:
                modality_value = "OK"
            elif n_peaks == 2:
                modality_value = "X"
            else:
                modality_value = "X"
            modality_radiant1_col.append(modality_value)

        # SNR validation column
        outlier_ch_info = outlier_details.get(ch, [])
        n_out = len(outlier_ch_info)
        if n_out == 0:
            snr_value = "OK"
        else:
            max_delta = max(abs(o.get("z_minus_k", 0.0)) for o in outlier_ch_info)
            frac_out = n_out / n_events_force if n_events_force > 0 else 0.0
            if max_delta < 3.0:
                snr_value = "OK"

            elif max_delta < 5.0:
                snr_value = "OK" if frac_out < 0.01 else "!!"

            else:  # max_delta >= 5
                if n_out == 1 and frac_out < 0.001:
                    snr_value = "OK"
                elif frac_out < 0.01:
                    snr_value = "!!"
                else:
                    snr_value = "X"
        snr_col.append(snr_value)

    df = pd.DataFrame({
        "Channel": ch_list,
        "SNR": snr_col,
        "Galaxy (FORCE)": spectral_col,
        "Vrms (FORCE)": modality_force_col,
        "Vrms (LT)": modality_lt_col,
        "Vrms (RADIANT0)": modality_radiant0_col,
        "Vrms (RADIANT1)": modality_radiant1_col,
        "Glitching": glitch_col,
    })

    health_cols =["SNR", "Galaxy (FORCE)", "Vrms (FORCE)", "Vrms (LT)", "Vrms (RADIANT0)", "Vrms (RADIANT1)", "Glitching"]
    df["Channel Health"] = df[health_cols].apply(channel_health, axis=1)
    df.to_csv(out_csv_file, index=False)
    logger.info(f"Validation summary saved to {out_csv_file}")

def write_spectral_results(ch, excess_info_results, station_id, run_label, log_once = False, reset_file = False):
    spectral_results_file = os.path.join(RESULTS_DIR, f"spectral_analysis_results_{station_id}_{run_label}.txt")
    if reset_file:
        open(spectral_results_file, "w").close()  # Clear the file if reset is requested
    
    with open(spectral_results_file, "a") as f:
        f.write(f"Channel {ch:02d}:\n")
        for band, results in excess_info_results.items():
            f.write(f"\n=== {band} ===\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        f.write("\n")
    if log_once:
        logger.info(f"Spectral analysis results written to {spectral_results_file}")

    
def write_snr_outlier_details(outlier_details, station_id, run_label):
    outlier_results_file = os.path.join(RESULTS_DIR, f"snr_details_{station_id}_{run_label}.txt")
    with open(outlier_results_file, "w") as f:
        for ch in sorted(outlier_details.keys()):
            entries = outlier_details[ch]
            n_outliers = len(entries)

            if n_outliers == 0:
                f.write(f"\nChannel {ch:02d}:\n 0 outliers\n")
                continue

            k_ch = entries[0]["k"]
            f.write(f"\nChannel {ch:02d}:\n {n_outliers} outliers (k = {k_ch:.2f})\n")
            
            for e in entries:
                f.write(f"  - run {e['run']}, event {e['eventNumber']}, |z| = {e['z_abs']:.2f} (delta = {e['z_minus_k']:.2f} above k)\n")
            
    logger.info(f"SNR outlier details written to {outlier_results_file}")

def write_vrms_modality_results(modality, tail_label, trigger_label, station_id, run_label):
    modality_results_file = os.path.join(RESULTS_DIR, f"vrms_modality_{station_id}_{run_label}_{trigger_label}.txt")
    with open(modality_results_file, "w") as f:
        for ch in sorted(modality.keys()):
            modality_result = modality[ch]
            tail_label_result = tail_label[ch]
            f.write(f"Channel {ch} ({trigger_label} events): {modality_result} ({tail_label_result})\n")
    logger.info(f"Vrms modality results for {trigger_label} events written to {modality_results_file}")

def write_glitching_results(glitch_info, station_id, run_label):
    glitch_results_file = os.path.join(RESULTS_DIR, f"glitching_analysis_results_{station_id}_{run_label}.txt")
    lines = [
        (
            f"Channel {ch:2d} | "
            f"n_glitches: {glitch_info[ch]['n_glitches']:4d} | "
            f"n_events: {glitch_info[ch]['n_events']:<4d} | "
            f"(frac={glitch_info[ch]['glitch_fraction']:.3f}) | "
            f"p={glitch_info[ch]['pval']:.2e} | "
            f"CI99%={glitch_info[ch]['confidence_interval']} | "
            f"{glitch_info[ch]['validation']}"
        )
        for ch in all_channels]
    with open(glitch_results_file, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Glitching analysis results written to {glitch_results_file}")

def write_block_offset_results(block_offset_stats, station_id, run_label):
    block_offset_results_file = os.path.join(RESULTS_DIR, f"block_offset_analysis_results_{station_id}_{run_label}.txt")
    with open(block_offset_results_file, "w") as f:
        for ch in sorted(block_offset_stats.keys()):
            stats = block_offset_stats[ch]
            f.write(f"Channel {ch:02d}:\n")
            f.write(f"  Before removal - mean: {stats['before_mean']} V, median: {stats['before_median']} V, std: {stats['before_std']} V, IQR: {stats['iqr_before']} V\n")
            f.write(f"  After removal - mean: {stats['after_mean']} V, median: {stats['after_median']} V, std: {stats['after_std']} V, IQR: {stats['iqr_after']} V\n")
            f.write(f"  Removal fraction (based on median): {stats['removal_fraction']*100:.1f}%\n")
            f.write(f"  IQR reduction fraction: {stats['iqr_reduction_fraction']*100:.1f}%\n")

            if stats["before_median"] > block_offset_limits["median"]:
                logger.warning(f"Channel {ch:02d} has a high median block offset of {stats['before_median']} V before removal, which may indicate a potential issue with the channel.")
            elif stats["iqr_before"] > block_offset_limits["iqr"]:
                logger.warning(f"Channel {ch:02d} has a high IQR of block offsets ({stats['iqr_before']} V) before removal, indicating significant variability that may need further investigation.")
            elif stats["after_median"] > block_offset_limits["median"]:
                logger.warning(f"Channel {ch:02d} has a relatively high median block offset of {stats['after_median']} V after removal, removal was not fully effective.")
            elif stats["iqr_after"] > block_offset_limits["iqr"]:
                logger.warning(f"Channel {ch:02d} has a relatively high IQR of block offsets ({stats['iqr_after']} V) after removal, indicating that there may still be significant variability in block offsets.")
    logger.info(f"Block offset analysis results written to {block_offset_results_file}")

if __name__ == "__main__":

    argparser = ArgumentParser(description="RNO-G Science Verification Analysis")
    argparser.add_argument("-st", "--station_id", type=int, required=True, help="Station to analyze, e.g --station_id 14")
    argparser.add_argument("-b", "--backend", type=str, default="pyroot", help="Backend to use for reading data, should be either pyroot or uproot (default: pyroot), e.g. --backend pyroot or --backend uproot")
    argparser.add_argument("-sl", "--save_location", type=str, default=PLOTS_DIR, help="Location to save the output plots (default: plots directory under script directory), e.g. --save_location /path/to/save/plots")
    argparser.add_argument("-ex", "--exclude-runs", nargs="+", type=int, default=[], metavar="RUN", help="Run number(s) to exclude, e.g. --exclude-runs 1005 1010")
    argparser.add_argument("--debug_plot", action="store_true", help="If set, will create debug plots.")
    argparser.add_argument("--sampling_rate", type=str, default= "after_2024", choices=["before_2024", "after_2024"], help="Sampling rate to use, choices are 'before_2024' (3.2 GHz) and 'after_2024' (2.4 GHz), default is 'after_2024'.")

    run_selection = argparser.add_mutually_exclusive_group(required=True)
    run_selection.add_argument("--runs", nargs="+", type=int, metavar="RUN_NUMBERS",
                           help="Run number(s) to analyze. Each run number should be given explicitly separated by a space, e.g. --runs 1001 1002 1005")
    run_selection.add_argument("--run_range", nargs=2, type=int, metavar=("START_RUN", "END_RUN"),
                            help="Range of run numbers to analyze (inclusive). Provide start and end run numbers separated by a space, e.g. --run_range 1000 1050")
    run_selection.add_argument("--time_range", nargs=2, type=str, metavar=("START_DATE", "END_DATE"),
                            help="Date range to analyze (inclusive). Provide start and end dates separated by a space in YYYY-MM-DD format, e.g. --time_range 2024-07-15 2024-09-30")


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

    sampling_rate_choice = args.sampling_rate
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
    config = get_station_config(station_id)
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

    # Read data 
    ###### !!!!! Implement monitoring data !!!!! ######
    spec_arr, trace_arr, times_trace_arr, snr_arr, run_no, times, freqs, event_info, glitch_arr, block_offsets_arr = read_rnog_data(station_id, run_numbers, backend=backend, sampling_rate=sr) 

    # Normalize surface channel spectra
    norm_spec_arr, scale_factors = normalize_channels(spec_arr, freqs, downward_channels, upward_channels)
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

    band_config = copy.deepcopy(SPECTRAL_BANDS)

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

    logger.info("Starting spectral analysis for FORCE trigger events...")
    for ch in surface_channels:
        i = ch_to_idx[ch]
        ratio_arr_dict_ch = {}

        for band_name, ratio_arr in ratio_arr_dict.items():
            ratio_arr_dict_ch[band_name] = ratio_arr[i]

        excess_info_results = excess_info_from_ratio_specific_bkg(ratio_arr_dict_ch, ALPHA_SPEC, CI_THRESHOLDS_SPEC)
        validation_results = validate_excess_in_bands(excess_info_results)

        all_excess_info[ch] = excess_info_results
        all_validation_results[ch] = validation_results

        # Write detailed spectral results to text file for each channel
        write_spectral_results(ch, excess_info_results, station_id, run_label, log_once=(ch==surface_channels[-1]), reset_file=(ch==surface_channels[0]))
    

    # Select other trigger types and plot them
    lt_mask = choose_trigger_type(event_info, "LT")
    spec_arr_lt = spec_arr[:, lt_mask, :]

    radiant0_mask = choose_trigger_type(event_info, "RADIANT0")
    spec_arr_radiant0 = spec_arr[:, radiant0_mask, :]

    radiant1_mask = choose_trigger_type(event_info, "RADIANT1")
    spec_arr_radiant1 = spec_arr[:, radiant1_mask, :]       

    # Surface spectrum
    plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr_force, freqs, upward_channels, downward_channels, save_location, run_label, trigger_label="force")
    plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr_lt, freqs, upward_channels, downward_channels, save_location, run_label, trigger_label="lt")
    plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr_radiant0, freqs, upward_channels, downward_channels, save_location, run_label, trigger_label="radiant0")
    plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr_radiant1, freqs, upward_channels, downward_channels, save_location, run_label, trigger_label="radiant1")
    
    # Normalized surface spectrum - only force
    plot_time_integrated_surface_spectra_normalized(station_id, norm_spec_arr_force, freqs, upward_channels, downward_channels, save_location, run_label)

    # Deep spectrum (unnormalized)
    plot_time_integrated_deep_spectra(station_id, spec_arr_force, freqs, vpol_channels, hpol_channels, save_location, run_label, trigger_label="force")
    plot_time_integrated_deep_spectra(station_id, spec_arr_lt, freqs, vpol_channels, hpol_channels, save_location, run_label, trigger_label="lt")
    plot_time_integrated_deep_spectra(station_id, spec_arr_radiant0, freqs, vpol_channels, hpol_channels, save_location, run_label, trigger_label="radiant0")
    plot_time_integrated_deep_spectra(station_id, spec_arr_radiant1, freqs, vpol_channels, hpol_channels, save_location, run_label, trigger_label="radiant1")
    
    ###### SNR analysis
    logger.info("Starting SNR analysis for FORCE trigger events...")
    snr_arr_force = snr_arr[:, force_mask]
    event_info_force = {key: np.array(value)[force_mask] for key, value in event_info.items()}
    times = np.array(times)
    times_force = times[force_mask]

    log_snr_arr, log_mean_list, log_median_list, log_std_list, log_difference_list = calculate_statistics_log_snr(snr_arr_force)
    ref_log_mean_filename = f"expected_values/station_{station_id}_ref_log_mean_snr.json"
    ref_log_std_filename = f"expected_values/station_{station_id}_ref_log_std_snr.json"
    ref_log_mean_list = load_values_json(SCRIPT_DIR, ref_log_mean_filename)
    ref_log_std_list = load_values_json(SCRIPT_DIR, ref_log_std_filename)
    z_score_arr_log_snr = calculate_z_score_snr(log_snr_arr, ref_log_mean_list, ref_log_std_list, all_channels)
    k_values_filename_snr = f"expected_values/station_{station_id}_k_ref_values_snr.json"
    k_values_log_snr = load_values_json(SCRIPT_DIR, k_values_filename_snr)
    flag_outliers_snr = outlier_flag(z_score_arr_log_snr, k_values_log_snr, all_channels)

    outlier_details_snr = find_outlier_details(z_score_arr_log_snr, k_values_log_snr, flag_outliers_snr, event_info_force, all_channels)
    write_snr_outlier_details(outlier_details_snr, station_id, run_label)

    day_interval = choose_day_interval(times)
    plot_snr_against_time(station_id, times_force, snr_arr_force, flag_outliers_snr, z_score_arr_log_snr, k_values_log_snr, all_channels, save_location, run_label, nrows=12, ncols=2, day_interval=day_interval)

    ##### Vrms analysis
    vrms_arr, vrms_arr_force, vrms_arr_radiant0, vrms_arr_radiant1, vrms_arr_lt = calculate_vrms(trace_arr, event_info)
    logger.info(f"Number of RADIANT0 trigger events: {len(vrms_arr_radiant0[1])}, Number of RADIANT1 trigger events: {len(vrms_arr_radiant1[1])}, Number of LT trigger events: {len(vrms_arr_lt[1])}")
    
    logger.info(f"Calculating Vrms modality and tail characteristics for each trigger type...")
    modality_dict_force = kde_modality(vrms_arr_force, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_force = tail_fraction_and_trimmed_skew_two_sided(vrms_arr_force, all_channels, skewness_config=skewness_function_parameters)
    if len(vrms_arr_force[1]) < 100:
            logger.warning(f"FORCE trigger has less than 100 valid Vrms entries ({len(vrms_arr_force)}). Results for the Vrms statistics may be unreliable.")
    modality_force, tail_label_force = report_vrms_characteristics(modality_dict_force, tail_dict_force, all_channels, report_config=report_vrms_function_parameters)

    modality_dict_radiant0 = kde_modality(vrms_arr_radiant0, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_radiant0 = tail_fraction_and_trimmed_skew_two_sided(vrms_arr_radiant0, all_channels, skewness_config=skewness_function_parameters)
    if len(vrms_arr_radiant0[1]) < 100:
            logger.warning(f"RADIANT0 trigger has less than 100 valid Vrms entries ({len(vrms_arr_radiant0)}). Results for the Vrms statistics may be unreliable.")
    modality_radiant0, tail_label_radiant0 = report_vrms_characteristics(modality_dict_radiant0, tail_dict_radiant0, all_channels, report_config=report_vrms_function_parameters)

    modality_dict_radiant1 = kde_modality(vrms_arr_radiant1, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_radiant1 = tail_fraction_and_trimmed_skew_two_sided(vrms_arr_radiant1, all_channels, skewness_config=skewness_function_parameters)
    if len(vrms_arr_radiant1[1]) < 100:
            logger.warning(f"RADIANT1 trigger has less than 100 valid Vrms entries ({len(vrms_arr_radiant1)}). Results for the Vrms statistics may be unreliable.")
    modality_radiant1, tail_label_radiant1 = report_vrms_characteristics(modality_dict_radiant1, tail_dict_radiant1, all_channels, report_config=report_vrms_function_parameters)

    modality_dict_lt = kde_modality(vrms_arr_lt, all_channels, kde_modality_config=kde_modality_function_parameters)
    tail_dict_lt = tail_fraction_and_trimmed_skew_two_sided(vrms_arr_lt, all_channels, skewness_config=skewness_function_parameters)
    if len(vrms_arr_lt[1]) < 100:
            logger.warning(f"LT trigger has less than 100 valid Vrms entries ({len(vrms_arr_lt)}). Results for the Vrms statistics may be unreliable.")
    modality_lt, tail_label_lt = report_vrms_characteristics(modality_dict_lt, tail_dict_lt, all_channels, report_config=report_vrms_function_parameters)
    plot_vrms_values_against_time(times, vrms_arr, all_channels, station_id, run_label, save_location, n_rows=12, n_cols=2, day_interval=day_interval)

    # Write detailed Vrms modality results to text files for each trigger type
    write_vrms_modality_results(modality_force, tail_label_force, trigger_label="FORCE", station_id=station_id, run_label=run_label)
    write_vrms_modality_results(modality_radiant0, tail_label_radiant0, trigger_label="RADIANT0", station_id=station_id, run_label=run_label)
    write_vrms_modality_results(modality_radiant1, tail_label_radiant1, trigger_label="RADIANT1", station_id=station_id, run_label=run_label)
    write_vrms_modality_results(modality_lt, tail_label_lt, trigger_label="LT", station_id=station_id, run_label=run_label)


    # The Vrms statistics can be misleading (especially for low event number) so the debugging plots are always generated
    debug_plot_vrms_distribution(vrms_arr_force, modality_dict_force, channel_list=all_channels, station_id=station_id, run_label=run_label, trigger_label="FORCE", save_location=save_location, n_rows=12, n_cols=2)
    debug_plot_vrms_distribution(vrms_arr_radiant0, modality_dict_radiant0, channel_list=all_channels, station_id=station_id, run_label=run_label, trigger_label="RADIANT0", save_location=save_location, n_rows=12, n_cols=2)
    debug_plot_vrms_distribution(vrms_arr_radiant1, modality_dict_radiant1, channel_list=all_channels, station_id=station_id, run_label=run_label, trigger_label="RADIANT1", save_location=save_location, n_rows=12, n_cols=2)
    debug_plot_vrms_distribution(vrms_arr_lt, modality_dict_lt, channel_list=all_channels, station_id=station_id, run_label=run_label, trigger_label="LT", save_location=save_location, n_rows=12, n_cols=2)
    
    ##### Glitching analysis
    logger.info("Starting glitching analysis...")
    config_glitching = copy.deepcopy(config_glitching_values)
    glitch_info = binomtest_glitch_fraction(glitch_arr, all_channels, config_glitching=config_glitching)
    write_glitching_results(glitch_info, station_id, run_label)
    
    glitching_violin_plot(glitch_arr, all_channels, station_id, run_label, save_location)
    plot_glitch_q99_over_time(np.array(times), glitch_arr, all_channels, station_id, run_label, save_location)

    # Trigger rate with thresholds plot
    fig_trigger, ax_rate, ax_thr = plot_trigger_rate_with_thresholds(station_id, event_info, downward_channels, upward_channels, run_label, day_interval, bin_width_initial=300, max_bins=800, save_location=save_location)
    
    ##### Block offsets
    logger.info("Starting block offset analysis, results are not used to determine channel health, see warnings in the log file for channels with potential block offset issues. The block offsets are then removed.")
    fit_block_offsets_before = get_block_offsets_before_removal(block_offsets_arr, event_info, all_channels)
    fit_block_offsets_after = get_block_offsets_after_removal(trace_arr, event_info, all_channels, sampling_rate=sr)

    block_offset_stats = block_offset_statistics(fit_block_offsets_before, fit_block_offsets_after, all_channels)
    write_block_offset_results(block_offset_stats, station_id, run_label)
    plot_block_offsets_violin_before_after_comparison(fit_block_offsets_before, fit_block_offsets_after, all_channels, station_id, run_label, save_location)

    # Debug plots
    if args.debug_plot:   
        debug_plot_ratios(ratio_arr_dict=ratio_arr_dict, channels_order=channels_order, save_location=save_location, station_id=station_id, run_label=run_label, bins=30,)
        debug_plot_snr_distribution(log_snr_arr, channel_list=all_channels, save_location=save_location, station_id=station_id, run_label=run_label, bins=30)
        debug_plot_z_score_snr(z_score_arr_log_snr, channel_list=all_channels, save_location=save_location, station_id=station_id, run_label=run_label, bins=30)   
        

    # Create summary CSV file
    n_events_force = spec_arr_force.shape[1]
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
        modality_dict_force,
        modality_dict_lt,
        modality_dict_radiant0,
        modality_dict_radiant1,
        outlier_details_snr,
        save_location,
    )
    