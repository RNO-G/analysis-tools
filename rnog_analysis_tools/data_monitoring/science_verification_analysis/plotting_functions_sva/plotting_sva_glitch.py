import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np
import os
import pandas as pd

#### Glitching Plots ####
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
