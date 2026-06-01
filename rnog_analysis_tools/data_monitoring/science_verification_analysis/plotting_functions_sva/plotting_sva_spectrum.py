import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import numpy as np
from NuRadioReco.utilities import units
import os
import logging

logger = logging.getLogger(__name__)

TRIGGER_MAP = {
    "force" : "n_forced_triggers",
    "lt" : "n_lt_triggers",
    "radiant0" : "n_rf0_triggers",
    "radiant1" : "n_rf1_triggers",
}

def get_weights_if_monitoring(trigger_label, use_monitoring=False, run_event_counts=None):
    '''Helper function to get weights for averaging spectra if using monitoring data, otherwise return None.'''
    if use_monitoring and run_event_counts is not None:
        weight_key = TRIGGER_MAP[trigger_label.lower()]
        n_events_per_run_trigger = np.array([run_event_counts[run_no][weight_key] for run_no in run_event_counts])
        unit_label = "ADC Counts"
        return n_events_per_run_trigger, unit_label
    else:
        unit_label = "V/GHz"
        return None, unit_label

#### Spectrum Plots ####
def plot_time_integrated_surface_spectra_unnormalized(station_id, spec_arr, freqs, upward_channels, downward_channels, save_location, run_label, trigger_label, use_monitoring = False, run_event_counts = None):
    '''Plot time-integrated surface channel spectra. Use weighted average if use_monitoring is True and run_event_counts is provided, otherwise use simple average.'''
    plt.figure(figsize=(10, 6))
    weights, unit_label = get_weights_if_monitoring(trigger_label, use_monitoring, run_event_counts)
    for ch in upward_channels:
        if weights is not None:
            spec_mean = np.average(spec_arr[ch, :, :], axis=0, weights=weights)
        else:
            spec_mean = np.mean(spec_arr[ch, :, :], axis=0)
        plt.plot(freqs[1:] / units.MHz, spec_mean[1:], label=f'Ch {ch} (up)', linestyle='-')
    for ch in downward_channels:
        if weights is not None:
             spec_mean = np.average(spec_arr[ch, :, :], axis=0, weights=weights)
        else:
            spec_mean = np.mean(spec_arr[ch, :, :], axis=0)
        plt.plot(freqs[1:] / units.MHz, spec_mean[1:], label=f'Ch {ch} (down)', linestyle='--')

    plt.xlabel('Frequency [MHz]')
    plt.xlim(50, 800)
    #plt.ylim(0, 5)
    plt.ylabel(f'Amplitude Spectrum [{unit_label}]')
    plt.title(f'Time-Integrated Spectrum of Surface Channels  ({trigger_label} Trigger)')
    plt.legend(loc="upper right", 
               frameon=True,
               fancybox=True,
               framealpha=0.9,
               edgecolor="black")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, f"{trigger_label}_time_integrated_surface_spectra_unnormalized_{station_id}_{run_label}.pdf"))
    plt.close()

def plot_time_integrated_surface_spectra_normalized(station_id, norm_spec_arr, freqs, upward_channels, downward_channels, save_location, run_label, use_monitoring = False, run_event_counts = None):
    '''Plot time-integrated normalized surface channel spectra for FORCE trigger events. Use weighted average if use_monitoring is True and run_event_counts is provided, otherwise use simple average.'''
    plt.figure(figsize=(10, 6))
    trigger_label = "force"
    weights, unit_label = get_weights_if_monitoring(trigger_label, use_monitoring, run_event_counts)
    for ch in upward_channels:
        if weights is not None:
            spec_mean = np.average(norm_spec_arr[ch, :, :], axis=0, weights=weights)
        else:
            spec_mean = np.mean(norm_spec_arr[ch, :, :], axis=0)
        plt.plot(freqs[1:] / units.MHz, spec_mean[1:], label=f'Ch {ch} (up)', linestyle='-')
    for ch in downward_channels:
        if weights is not None:
            spec_mean = np.average(norm_spec_arr[ch, :, :], axis=0, weights=weights)
        else:
            spec_mean = np.mean(norm_spec_arr[ch, :, :], axis=0)
        plt.plot(freqs[1:] / units.MHz, spec_mean[1:], label=f'Ch {ch} (down)', linestyle='--')

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
    plt.ylim(0, )
    plt.ylabel(f'Amplitude Spectrum [{unit_label}]')
    plt.title(f'Time-Integrated Spectrum of Surface Channels (FORCE Trigger)')
    
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
    plt.close()

def plot_time_integrated_deep_spectra(station_id, spec_arr, freqs, vpol_channels, hpol_channels, save_location, run_label, trigger_label, use_monitoring = False, run_event_counts = None):
    '''Plot time-integrated deep channel spectra. Use weighted average if use_monitoring is True and run_event_counts is provided, otherwise use simple average.'''
    plt.figure(figsize=(10, 6))
    weights, unit_label = get_weights_if_monitoring(trigger_label, use_monitoring, run_event_counts)
    for ch in vpol_channels:
        if weights is not None:
            spec_mean = np.average(spec_arr[ch, :, :], axis=0, weights=weights)
        else:
            spec_mean = np.mean(spec_arr[ch, :, :], axis=0)
        plt.plot(freqs[1:] / units.MHz, spec_mean[1:], label=f'Ch {ch} (VPOL)', linestyle='-')
    for ch in hpol_channels:
        if weights is not None:
            spec_mean = np.average(spec_arr[ch, :, :], axis=0, weights=weights)
        else:
            spec_mean = np.mean(spec_arr[ch, :, :], axis=0)
        plt.plot(freqs[1:] / units.MHz, spec_mean[1:], label=f'Ch {ch} (HPOL)', linestyle='--')

    plt.xlabel('Frequency [MHz]')
    plt.xlim(50, 800)
    #plt.ylim(0, 5)
    plt.ylabel(f'Amplitude Spectrum [{unit_label}]')
    plt.title(f'Time-Integrated Spectrum of Deep Channels ({trigger_label} Trigger)')
    plt.legend(loc="upper right", 
               frameon=True,
               fancybox=True,
               framealpha=0.9,
               edgecolor="black")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, f"{trigger_label}_time_integrated_deep_spectra_unnormalized_{station_id}_{run_label}.pdf"))
    plt.close()