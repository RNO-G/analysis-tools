import matplotlib.pyplot as plt
import numpy as np
import os

#### Debug Plots ####
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

    #ax.set_title("Log10 SNR Distribution (FORCE Trigger)")
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

    #ax.set_title("Z-Score SNR Distribution (FORCE Trigger)")
    ax.set_xlabel("Z-Score(SNR)")
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=8, frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(save_location,f"debug_z_score_snr_force_trigger_{station_id}_{run_label}.pdf",))
    plt.close(fig)

def debug_plot_vrms_distribution(vrms_arr, modality_dict, channel_list, station_id, run_label, trigger_label, save_location, n_rows=12, n_cols=2, use_monitoring=False):
    if use_monitoring:
        unit_label = "RMS [ADC]"
        plot_label = "RMS"
    else: 
        unit_label = "Vrms Values [V]"
        plot_label = "Vrms"
    
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
        ax.set_xlabel(unit_label)
        ax.set_ylabel("KDE Density")

    for i in range(len(channel_list), n_rows * n_cols):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_location,f"debug_{plot_label.lower()}_hist_kde_density_peaks_{station_id}_{run_label}_{trigger_label}.pdf",))

def debug_plot_ratios_just_galaxy(
    ratio_arr_dict,
    channels,
    save_location,
    station_id,
    run_label,
    bins=30,
):
    """
    Plot galactic-excess ratio distributions for selected surface channels.

    The channel axis in ratio_arr_dict["galactic_excess"] is assumed to follow
    the order given in surface_channels, rather than the numerical channel index.
    """
    band_name = "galactic_excess"
    surface_channels = [13, 15, 16, 18, 12, 14, 17, 19]

    if band_name not in ratio_arr_dict:
        raise KeyError(
            f"'{band_name}' not found in ratio_arr_dict. "
            f"Available keys: {list(ratio_arr_dict.keys())}"
        )

    ratio_list = ratio_arr_dict[band_name]

    if len(ratio_list) != len(surface_channels):
        raise ValueError(
            f"Expected {len(surface_channels)} channel entries in "
            f"ratio_arr_dict['{band_name}'], but found {len(ratio_list)}."
        )

    channel_to_index = {
        ch: index for index, ch in enumerate(surface_channels)
    }

    fig, ax = plt.subplots(figsize=(6, 5))

    for ch in channels:
        if ch not in channel_to_index:
            print(
                f"Warning: channel {ch} is not available. "
                f"Available channels: {surface_channels}"
            )
            continue

        if ch in [13, 15, 16, 18]:
            up_label = "Up"
        elif ch in [12, 14, 17, 19]:
            up_label = "Down"
        else:
            up_label = "unknown"

        channel_index = channel_to_index[ch]
        ratios = np.asarray(ratio_list[channel_index], dtype=float)

        # log10 is only defined for finite, positive values.
        valid_mask = np.isfinite(ratios) & (ratios > 0)
        ratios = ratios[valid_mask]

        if ratios.size == 0:
            print(
                f"Warning: channel {ch} contains no finite, positive ratios."
            )
            continue

        ax.hist(
            np.log10(ratios),
            bins=bins,
            histtype="step",
            linewidth=1.3,
            alpha=0.8,
            label=f"Ch {ch} ({up_label})",
        )

    #ax.set_title("Galactic Excess (FORCE Trigger)")
    ax.set_xlabel(r"$\log_{10}(R)$")
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.3)

    if ax.has_data():
        ax.legend(frameon=True)

    fig.tight_layout()

    output_path = os.path.join(
        save_location,
        (
            f"debug_amplitude_ratios_force_trigger_"
            f"{station_id}_{run_label}_just_galaxy.pdf"
        ),
    )

    fig.savefig(output_path)
    plt.close(fig)

def debug_plot_vrms_distribution_single_channel(
    vrms_arr,
    modality_dict,
    channel,
    station_id,
    run_label,
    trigger_label,
    save_location,
    use_monitoring=False
):
    if use_monitoring:
        unit_label = "RMS [ADC]"
        plot_label = "RMS"
    else:
        unit_label = "Vrms Values [V]"
        plot_label = "Vrms"

    fig, ax = plt.subplots(figsize=(8,6))

    info = modality_dict[channel]

    vrms = vrms_arr[channel]
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

    ax.hist(
        vrms,
        bins=30,
        density=True,
        alpha=0.3,
        color="gray",
        label=f"{plot_label} distribution",
    )

    ax.plot(
        vrms_grid,
        kde_values,
        color="blue",
        lw=1.5,
        label="Gaussian KDE",
    )

    if len(peaks) > 0:
        ax.plot(
            vrms_grid[peaks],
            kde_values[peaks],
            "ro",
            markersize=5,
            label="Peak",
        )

    #ax.set_title(f"Ch {channel}: {modality}")
    ax.set_xlabel(unit_label)
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_location,
            f"debug_{plot_label.lower()}_hist_kde_density_peaks_ch{channel}_{station_id}_{run_label}_{trigger_label}.pdf"
        )
    )

    plt.close(fig)