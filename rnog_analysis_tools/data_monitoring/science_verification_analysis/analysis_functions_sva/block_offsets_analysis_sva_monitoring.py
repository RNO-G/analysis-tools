import os
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import numpy as np

def get_force_block_offsets_monitoring(block_offset_arr, force_mask):
    # block_offset_arr has the shape (n_ch, n_events)
    block_offset_arr_force = block_offset_arr[:, force_mask]
    return block_offset_arr_force

def block_offset_statistics_monitoring(block_offset_arr_force, channel_list):
    stats = {}
    for ch in channel_list:
        offsets_ch = block_offset_arr_force[ch] # No need to flatten since already 2D with shape (n_events, n_blocks)
        stats[ch] = {
            "mean": np.nanmean(offsets_ch),
            "median": np.nanmedian(offsets_ch),
            "std": np.nanstd(offsets_ch),
            "iqr": np.nanpercentile(offsets_ch, 75) - np.nanpercentile(offsets_ch, 25),
        }
    return stats

def plot_block_offsets_violin_monitoring(block_offset_arr_force, channel_list, station_id, run_label, save_location):
    # block_offset_arr_force has the shape (n_ch, n_events)
    fig, ax = plt.subplots(figsize=(12, 6))

    positions = np.array(channel_list)
    violin_plot = ax.violinplot(block_offset_arr_force.T, positions=positions, showextrema=True, showmedians=True, vert=False, side="high", widths=1.8)

    for pc in violin_plot['bodies']:
        pc.set_facecolor("lightblue")
        pc.set_edgecolor("blue")
        pc.set_alpha(0.7)

    violin_plot["cmedians"].set_color("k")
    violin_plot["cmedians"].set_linewidth(1.8)

    ax.set_xlabel("Block Offsets")
    ax.set_ylabel("Channel")
    ax.grid(True, alpha=0.3)

    n_force_events = block_offset_arr_force.shape[1]
    ax.plot(np.nan, np.nan, label=f"{n_force_events} FORCE triggers", color="k")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(save_location, f"block_offset_violin_monitoring_{station_id}_{run_label}.pdf"))
    plt.close(fig)