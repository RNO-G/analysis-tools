import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import numpy as np
from NuRadioReco.utilities import units
import os
import logging

logger = logging.getLogger(__name__)

trigger_label_mapping = {
    "force_trigger_rate": "FORCE",
    "lt_trigger_rate": "LT",
    "rf0_trigger_rate": "RADIANT0",
    "rf1_trigger_rate": "RADIANT1"
}

def plot_trigger_rates_over_time(run_trigger_rates, save_location, station_id, run_label):
    '''Plot trigger rates over time for different trigger types.'''

    fig, ax = plt.subplots(figsize=(10, 6))

    trigger_types = [trigger for trigger in trigger_label_mapping.keys()]
    trigger_labels = [trigger_label_mapping[trigger] for trigger in trigger_types]

    run_numbers = sorted(run_trigger_rates.keys())
    times = [run_trigger_rates[run_no]["run_start_time_utc"] for run_no in run_numbers]

    for trigger_type, trigger_label in zip(trigger_types, trigger_labels):
        rates = [run_trigger_rates[run_no].get(trigger_type, np.nan) for run_no in run_numbers]
        ax.plot(times, rates, marker="o", label=trigger_label)

    ax.set_xlabel("Run Start Time (UTC)")
    ax.set_ylabel("Trigger Rate (Hz)")
    ax.set_title(f"Trigger Rates Over Time ({run_label})")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.autofmt_xdate()

    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black"
    )

    ax.grid()
    fig.tight_layout()

    fig.savefig(os.path.join(save_location, f"trigger_rates_over_time_{station_id}_{run_label}.pdf"))
    plt.close(fig)

def plot_trigger_rate_heatmap(run_trigger_rates, save_location, station_id, run_label):
    '''Plot trigger rates as heatmap (run vs trigger type).'''
    trigger_types = [trigger for trigger in trigger_label_mapping.keys()]
    trigger_labels = [trigger_label_mapping[trigger] for trigger in trigger_types]

    run_numbers = sorted(run_trigger_rates.keys())
    n_runs = len(run_numbers)

    rate_matrix = []

    for run_no in run_numbers:
        rate_matrix.append([
            run_trigger_rates[run_no].get(trigger, np.nan)
            for trigger in trigger_types
        ])
        

    rate_matrix = np.asarray(rate_matrix)

    times = [
        run_trigger_rates[run_no]["run_start_time_utc"]
        for run_no in run_numbers
    ]

    fig, ax = plt.subplots(figsize=(10, max(5, len(run_numbers) * 0.08)))

    im = ax.imshow(
        rate_matrix,
        aspect="auto",
        origin="lower"
    )

    cbar = fig.colorbar(im)
    cbar.set_label("Trigger Rate (Hz)")

    ax.set_xticks(np.arange(len(trigger_types)))
    ax.set_xticklabels(trigger_labels)

    if n_runs <= 15:
        ax.set_yticks(np.arange(len(run_numbers)))
        ax.set_yticklabels(run_numbers)
    else:
        ax.set_yticks(np.arange(0, len(run_numbers), max(1, len(run_numbers) // 15)))
        ax.set_yticklabels([run_numbers[i] for i in range(0, len(run_numbers), max(1, len(run_numbers) // 15))])
    

    ax.set_xlabel("Trigger Type")
    ax.set_ylabel("Run Number")
    ax.set_title(
        f"Trigger Rate Heatmap ({run_label})"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            save_location,
            f"trigger_rate_heatmap_{station_id}_{run_label}.pdf"
        )
    )

    plt.close(fig)