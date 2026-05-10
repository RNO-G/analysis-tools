import os

import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.modules.RNO_G.channelBlockOffsetFitter import fit_block_offsets
from NuRadioReco.utilities import units

def get_block_offsets_after_removal(trace_arr, event_info, channel_list, sampling_rate=2.4*units.GHz):
    force_mask = event_info["triggerType"] == "FORCE"
    trace_arr_force = trace_arr[:, force_mask, :]
    n_force_events = trace_arr_force.shape[1]

    fit_blocks = np.zeros((len(channel_list), n_force_events, 16))

    for ch in channel_list:
        traces_force_ch = trace_arr_force[ch] 
        offsets_ch = np.array([fit_block_offsets(trace, sampling_rate=sampling_rate) for trace in traces_force_ch])
        fit_blocks[ch] = offsets_ch

    return fit_blocks

def get_block_offsets_before_removal(block_offset_arr, event_info, channel_list):
    force_mask = event_info["triggerType"] == "FORCE"
    block_offset_arr_force = block_offset_arr[:, force_mask, :]
    n_force_events = block_offset_arr_force.shape[1]

    fit_blocks = np.zeros((len(channel_list), n_force_events, 16))

    for ch in channel_list:
        offsets_ch = block_offset_arr_force[ch] 
        fit_blocks[ch] = offsets_ch

    return -fit_blocks # - because in chp the negative is stored

def block_offset_statistics(fit_blocks_before, fit_blocks_after, channel_list):
    stats = {}
    for ch in channel_list:
        before_ch = fit_blocks_before[ch].flatten()
        after_ch = fit_blocks_after[ch].flatten()
        stats[ch] = {
            "before_mean": np.nanmean(before_ch),
            "before_median": np.nanmedian(before_ch),
            "before_std": np.nanstd(before_ch),
            "after_mean": np.nanmean(after_ch),
            "after_median": np.nanmedian(after_ch),
            "after_std": np.nanstd(after_ch),
            "iqr_before": np.nanpercentile(before_ch, 75) - np.nanpercentile(before_ch, 25),
            "iqr_after": np.nanpercentile(after_ch, 75) - np.nanpercentile(after_ch, 25),
        }

        removal_fraction = 1 - (np.median((after_ch)) / np.median((before_ch))) if np.median((before_ch)) != 0 else np.nan
        stats[ch]["removal_fraction"] = removal_fraction
        iqr_reduction_fraction = 1 - (stats[ch]["iqr_after"] / stats[ch]["iqr_before"]) if stats[ch]["iqr_before"] != 0 else np.nan
        stats[ch]["iqr_reduction_fraction"] = iqr_reduction_fraction
    return stats
    

def plot_block_offsets_violin_before_after_comparison(fit_blocks_before, fit_blocks_after, channel_list, station_id, run_label, save_location):
    fit_blocks_before_flat = fit_blocks_before.reshape(len(channel_list), -1)
    fit_blocks_after_flat = fit_blocks_after.reshape(len(channel_list), -1)

    fig, ax = plt.subplots(figsize=(12, 6))
    positions_before = np.array(channel_list)
    positions_after = np.array(channel_list) + 0.2

    parts_before = ax.violinplot(fit_blocks_before_flat.T, positions=positions_before, showextrema=True, showmedians=True, vert=False, side="high", widths=1.8)
    parts_after = ax.violinplot(fit_blocks_after_flat.T, positions=positions_after, showextrema=True, showmedians=True, vert=False, side="high", widths=1.8)

    for pc in parts_before["bodies"]:
         pc.set_facecolor("lightblue")
         pc.set_edgecolor("blue")
         pc.set_alpha(0.7)

    for pc in parts_after["bodies"]:
        pc.set_facecolor("crimson")
        pc.set_edgecolor("darkred")
        pc.set_alpha(0.7)

    parts_before["cmedians"].set_color("black")
    parts_before["cmedians"].set_linewidth(1.8)
    parts_after["cmedians"].set_color("black")
    parts_after["cmedians"].set_linewidth(1.8)

    ax.set_xlabel("Fitted block offset [V]")
    ax.set_ylabel("Channel")
    ax.grid(True, alpha=0.3)

    n_force_events = fit_blocks_before.shape[1]
    ax.plot(np.nan, np.nan, label=f"{n_force_events} FORCE triggers", color="k")
    ax.plot(np.nan, np.nan, label="removed block offsets", color="C0")
    ax.plot(np.nan, np.nan, label="after removal", color="C1")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(save_location, f"block_offset_violin_comparison_{station_id}_{run_label}.pdf"))
    plt.close(fig)


