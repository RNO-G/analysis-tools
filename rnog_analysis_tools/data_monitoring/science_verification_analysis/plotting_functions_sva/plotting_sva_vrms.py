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
    
def plot_vrms_values_against_time(times, vrms_arr_all, channel_list, station_id, run_label, save_location, force_mask, radiant0_mask, radiant1_mask, lt_mask, n_rows = 12, n_cols = 2, day_interval=None, use_monitoring=False):
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

    if day_interval is None:
        day_interval = choose_day_interval(times) 

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
    plt.subplots_adjust(bottom = 0.11, wspace = 0.38, left=0.08)
    fig.supxlabel("Date [UTC]", x = 0.5, y = 0.06)
    plt.savefig(os.path.join(save_location, f"{plot_label.lower()}_against_time_{station_id}_{run_label}.pdf"))
    plt.close(fig)

def plot_vrms_values_against_time_per_trigger(times, vrms_arr_all, channel_list, station_id, run_label, save_location,force_mask, radiant0_mask, radiant1_mask, lt_mask, n_rows=12, n_cols=2, day_interval=None, use_monitoring=False):
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

    if day_interval is None:
        day_interval = choose_day_interval(times) 

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
            ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ticks_ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc)
            )
        elif time_span_days < 3:
            ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
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

        plt.subplots_adjust(bottom = 0.11, wspace = 0.38, left=0.08)
        fig.supxlabel("Date [UTC]", x = 0.5, y = 0.06)

        filename = f"{plot_label}_against_time_{trig_name.lower()}_{station_id}_{run_label}.pdf"
        plt.savefig(os.path.join(save_location, filename))
        plt.close(fig)

def plot_vrms_values_against_time_single_trigger_zscore(times, vrms_arr, flag, z_score, k_values, trigger_name, channel_list, station_id, run_label, save_location, 
                                                 n_rows=12, n_cols=2, day_interval=None, use_monitoring=False):
    '''Plot RMS/Vrms against time for a single trigger type with z-score outlier highlighting.'''

    if use_monitoring:
        unit_label = "RMS [ADC]"
        plot_label = "rms"
    else:
        unit_label = r"$V_\mathrm{rms}$ [V]"
        plot_label = "vrms"

    times = pd.to_datetime(times, utc=True)
    if day_interval is None:
        day_interval = choose_day_interval(times) 

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
            plt.colorbar(sc, cax=cax, label=r"$
            -k$")

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
            label=r"$
            >k$"
        )
    ]

    axs[0].legend(handles=legend_handles, loc="upper left")

    ticks_ax = axs[len(channel_list) - 1]

    time_span = (times.max() - times.min()).total_seconds() / 86400.0

    if time_span < 1:
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ticks_ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc)
        )

    elif time_span < 3:
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
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

    plt.subplots_adjust(bottom = 0.11, wspace = 0.38, left=0.08)
    fig.supxlabel("Date [UTC]", x = 0.5, y = 0.06)

    filename = (
        f"{plot_label}_against_time_"
        f"{trigger_name.lower()}_"
        f"{station_id}_{run_label}.pdf"
    )

    plt.savefig(os.path.join(save_location, filename))

    plt.close(fig)

def plot_rolling_mean_std(times, rolling_mean_arr, rolling_std_arr, channel_list, station_id, run_label, trigger_name, save_location, n_rows=12, n_cols=2, day_interval=None, use_monitoring=False):
    '''Plot rolling mean and std for Vrms values against time for each channel.'''
    times = pd.to_datetime(times, utc=True)
    if use_monitoring:
        unit_label = "RMS [ADC]"
        plot_label = "rms"
    else:
        unit_label = r"$V_\mathrm{rms}$ [V]"
        plot_label = "vrms"

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 24), sharex=True, squeeze=False)
    axs = axs.ravel()

    for idx, ch in enumerate(channel_list):
        ax = axs[idx]

        ax.plot(times, rolling_mean_arr[ch], label="Rolling Mean", color="blue")
        ax.plot(times, rolling_std_arr[ch], label="Rolling Std", color="orange")

        ax.set_title(f"Channel {ch}")
        ax.grid(alpha=0.4)
        ax.legend()

    for j in range(len(channel_list), len(axs)):
        axs[j].set_visible(False)

    ticks_ax = axs[len(channel_list) - 1]
    time_span = (times.max() - times.min()).total_seconds() / 86400.0
    
    if day_interval is None:
        day_interval = choose_day_interval(times) 

    if time_span < 1:
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    elif time_span < 3:
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    else:
        ticks_ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=timezone.utc))

    fig.autofmt_xdate()
    fig.supylabel(f"Rolling Value ({unit_label})", x=0.02)
    
    plt.subplots_adjust(bottom = 0.11, wspace = 0.38, left=0.08)
    fig.supxlabel("Date [UTC]", x = 0.5, y = 0.06)

    plt.savefig(os.path.join(save_location, f"{plot_label}_rolling_mean_std_against_time_{trigger_name}_{station_id}_{run_label}.pdf"))
    plt.close(fig)

def plot_rolling_mean_linregress(times, rolling_mean_arr, channel_list, slope_dict, intercept_dict, station_id, run_label, trigger_name, save_location, n_rows=12, n_cols=2, day_interval=None, use_monitoring=False):
    '''Plot rolling mean for Vrms values against time for each channel with linear regression line.'''
    times = pd.to_datetime(times, utc=True)

    if day_interval is None:
        day_interval = choose_day_interval(times) 

    if use_monitoring:
        unit_label = "RMS [ADC]"
        plot_label = "rms"
    else:
        unit_label = r"$V_\mathrm{rms}$ [V]"
        plot_label = "vrms"

    times_rel_hour = (times - times.min()).total_seconds() / 3600.0

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 24), sharex=True, squeeze=False)
    axs = axs.ravel()

    for idx, ch in enumerate(channel_list):
        ax = axs[idx]

        ax.plot(times, rolling_mean_arr[ch], label="Rolling Mean", color="blue")

        slope = slope_dict[ch]
        intercept = intercept_dict[ch]
        reg_line = slope * times_rel_hour + intercept
        ax.plot(times, reg_line, label="Linear Fit", color="red", linestyle="--")

        ax.set_title(f"Channel {ch}")
        ax.grid(alpha=0.4)
        ax.legend()

    for j in range(len(channel_list), len(axs)):
        axs[j].set_visible(False)

    ticks_ax = axs[len(channel_list) - 1]
    time_span = (times.max() - times.min()).total_seconds() / 86400.0

    if time_span < 1:
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    elif time_span < 3:
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    else:
        ticks_ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=timezone.utc))

    fig.autofmt_xdate()
    fig.supylabel(f"Rolling Mean ({unit_label})", x=0.02)
    
    plt.subplots_adjust(bottom = 0.11, wspace = 0.38, left=0.08)
    fig.supxlabel("Date [UTC]", x = 0.5, y = 0.06)

    plt.savefig(os.path.join(save_location, f"{plot_label}_rolling_mean_linregress_against_time_{trigger_name}_{station_id}_{run_label}.pdf"))
    plt.close(fig)

def create_heatmap_plot(results_dict, label, save_dir, channel_list, station_id,
                        matrix_key, run_label, cmap="Reds", vmin=None, vmax=None):
    '''Create and save heatmap plots for each channel.'''

    for ch in channel_list:

        matrix = np.array(results_dict[ch][matrix_key])
        runs = list(range(matrix.shape[0]))

        title = f"{label} Heatmap for Channel {ch}"
        save_path = os.path.join(
            save_dir,
            f"force_trigger_station_{station_id}_{label.lower().replace(' ', '_')}_heatmap_channel{ch}_{run_label}.pdf"
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(
            matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            aspect="equal",
            rasterized=True
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"{label}")

        ax.set_xticks(np.arange(len(runs)))
        ax.set_yticks(np.arange(len(runs)))

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xlabel("Run Number")
        ax.set_ylabel("Run Number")
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()