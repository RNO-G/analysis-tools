import os
import datetime
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import timezone


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