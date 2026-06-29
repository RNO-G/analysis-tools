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
