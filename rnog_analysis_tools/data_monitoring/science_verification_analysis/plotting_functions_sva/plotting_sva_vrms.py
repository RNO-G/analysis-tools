import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import timezone
import logging
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


logger = logging.getLogger(__name__)

#### Vrms Plots ####
def plot_vrms_values_against_time(times, vrms_arr_all, channel_list, station_id, run_label, save_location, force_mask, radiant0_mask, radiant1_mask, lt_mask, n_rows = 12, n_cols = 2, day_interval=5, use_monitoring=False):
    '''Plot RMS (for monitoring.root) or Vrms (for dataProviderRNOG) distributions for different trigger types.'''
    if use_monitoring:
        unit_label = "RMS [ADC]"
        plot_label = "RMS"
    else:
        unit_label = r"$V_\mathrm{rms}$ [V]"
        plot_label = "Vrms"

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

            scatter = ax.scatter(times_trig, vrms_trig, s=8, alpha=0.5, label=trig_name, color=trigger_colors[trig_name], rasterized=True)
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
    
    fig.supylabel(unit_label, x=0.02)
    plt.subplots_adjust(bottom = 0.05, wspace = 0.15, left=0.1)
    fig.supxlabel("Date [UTC]", x = 0.5, y = 0.06)
    plt.savefig(os.path.join(save_location, f"{plot_label.lower()}_against_time_{station_id}_{run_label}.pdf"))
    plt.close(fig)

def plot_vrms_values_against_time_per_trigger(times, vrms_arr_all, channel_list, station_id, run_label, save_location,force_mask, radiant0_mask, radiant1_mask, lt_mask, n_rows=12, n_cols=2, day_interval=5, use_monitoring=False):
    '''Plot RMS/Vrms against time separately for each trigger type.'''

    if use_monitoring:
        unit_label = "RMS [ADC]"
        plot_label = "rms"
    else:
        unit_label = r"$V_\mathrm{rms}$ [V]"
        plot_label = "vrms"

    trigger_masks = {
        "FORCE": force_mask,
        "RADIANT0": radiant0_mask,
        "RADIANT1": radiant1_mask,
        "LT": lt_mask,
    }

    trigger_colors = {
        "FORCE": "tab:blue",
        "RADIANT0": "tab:orange",
        "RADIANT1": "tab:green",
        "LT": "tab:red",
    }

    times = np.asarray(times)
    vrms_arr_all = np.asarray(vrms_arr_all)

    for trig_name, trig_mask in trigger_masks.items():

        times_trig = times[trig_mask]

        if len(times_trig) == 0:
            logger.warning(f"No {trig_name} events found. Skipping plot.")
            continue

        fig, axs = plt.subplots(
            n_rows, n_cols,
            figsize=(15, 24),
            sharex=True,
            squeeze=False
        )
        axs = axs.ravel()

        for idx, ch in enumerate(channel_list):
            ax = axs[idx]

            vrms_trig = vrms_arr_all[ch, trig_mask]

            ax.scatter(
                times_trig,
                vrms_trig,
                s=8,
                alpha=0.5,
                color=trigger_colors[trig_name],
                rasterized=True
            )

            ax.set_title(f"Channel {ch}")
            ax.grid(alpha=0.4)

        for j in range(len(channel_list), len(axs)):
            axs[j].set_visible(False)

        ticks_ax = axs[len(channel_list) - 1]

        time_span = times_trig.max() - times_trig.min()
        time_span_days = time_span / np.timedelta64(1, "D")

        if time_span_days < 1:
            ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ticks_ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc)
            )
        elif time_span_days < 3:
            ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            ticks_ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc)
            )
        else:
            ticks_ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
            ticks_ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%m-%d", tz=timezone.utc)
            )

        fig.suptitle(f"{unit_label} vs Time — {trig_name}", y=0.995)
        fig.supylabel(unit_label, x=0.02)
        fig.supxlabel("Date [UTC]", x=0.5, y=0.06)

        plt.subplots_adjust(bottom=0.08, wspace=0.15, left=0.1)

        filename = f"{plot_label}_against_time_{trig_name.lower()}_{station_id}_{run_label}.pdf"
        plt.savefig(os.path.join(save_location, filename))
        plt.close(fig)

def plot_vrms_values_against_time_single_trigger_zscore(times, vrms_arr, flag, z_score, k_values, trigger_name, channel_list, station_id, run_label, save_location, 
                                                 n_rows=12, n_cols=2, day_interval=5, use_monitoring=False):
    '''Plot RMS/Vrms against time for a single trigger type with z-score outlier highlighting.'''

    if use_monitoring:
        unit_label = "RMS [ADC]"
        plot_label = "rms"
    else:
        unit_label = r"$V_\mathrm{rms}$ [V]"
        plot_label = "vrms"

    times = pd.to_datetime(times, utc=True)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 24),
        sharex=True,
        squeeze=False
    )

    axs = axs.ravel()

    for idx, ch in enumerate(channel_list):

        ax = axs[idx]

        vrms_ch = vrms_arr[ch]
        flag_ch = flag[ch]

        good_mask = ~flag_ch

        ax.scatter(
            times[good_mask],
            vrms_ch[good_mask],
            s=8,
            alpha=0.25,
            color="gray",
            rasterized=True
        )

        zex = np.abs(z_score[ch]) - k_values[ch]
        zex = np.clip(zex, 0, None)

        sc = ax.scatter(
            times[flag_ch],
            vrms_ch[flag_ch],
            s=8,
            c=zex[flag_ch],
            cmap="Reds",
            rasterized=True
        )

        if np.any(flag_ch):
            cax = ax.inset_axes([1.02, 0.1, 0.05, 0.8])
            plt.colorbar(sc, cax=cax, label=r"$|z|-k$")

        ax.grid(alpha=0.4)

        ax.text(
            0.85,
            0.95,
            f"Ch {ch}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round, pad=0.25",
                facecolor="white",
                alpha=0.8
            )
        )

    for j in range(len(channel_list), len(axs)):
        axs[j].set_visible(False)

    red = plt.cm.Reds(0.6)

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markeredgecolor="gray",
            markerfacecolor="gray",
            alpha=0.4,
            markersize=6,
            label=r"$|z|\leq k$"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markeredgecolor=red,
            markerfacecolor=red,
            markersize=6,
            label=r"$|z|>k$"
        )
    ]

    axs[0].legend(handles=legend_handles, loc="upper left")

    ticks_ax = axs[len(channel_list) - 1]

    time_span = (times.max() - times.min()).total_seconds() / 86400.0

    if time_span < 1:
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ticks_ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc)
        )

    elif time_span < 3:
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ticks_ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc)
        )

    else:
        ticks_ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=day_interval)
        )

        ticks_ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d", tz=timezone.utc)
        )

    fig.autofmt_xdate()

    fig.suptitle(f"{unit_label} vs Time — {trigger_name}", y=0.995)

    fig.supylabel(unit_label, x=0.02)

    fig.supxlabel("Date [UTC]", x=0.5, y=0.06)

    plt.subplots_adjust(
        bottom=0.08,
        wspace=0.38,
        left=0.1
    )

    filename = (
        f"{plot_label}_against_time_"
        f"{trigger_name.lower()}_"
        f"{station_id}_{run_label}.pdf"
    )

    plt.savefig(os.path.join(save_location, filename))

    plt.close(fig)