import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import timezone

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
    
    fig.supylabel(unit_label, x=0.02)
    plt.subplots_adjust(bottom = 0.05, wspace = 0.15, left=0.1)
    fig.supxlabel("Date [UTC]", x = 0.5, y = 0.06)
    plt.savefig(os.path.join(save_location, f"{plot_label.lower()}_against_time_{station_id}_{run_label}.pdf"))
    plt.close(fig)