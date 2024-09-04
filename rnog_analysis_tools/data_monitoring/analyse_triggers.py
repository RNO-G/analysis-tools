from rnog_analysis_tools.data_monitoring.datasets import Datasets, convert_events_information

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md

import sys
import datetime as dt


def plot_triggers(data, runs, station, duration=None):

    if station != 14:
        downwardfacing_radiantThrs = np.mean(data["radiantThrs"][:, [12, 14, 15, 17, 18, 20]], axis=1)
        upwardfacing_radiantThrs = np.mean(data["radiantThrs"][:, [13, 16, 19]], axis=1)
    else:
        downwardfacing_radiantThrs = np.mean(data["radiantThrs"][:, [12, 14, 16, 18]], axis=1)
        upwardfacing_radiantThrs = np.mean(data["radiantThrs"][:, [13, 15, 17, 19]], axis=1)

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots(figsize=(16, 5))

    
    run_duration = duration or np.amax(data["triggerTime"]) - np.amin(data["triggerTime"])
    run_duration_readout = np.amax(data["readoutTime"]) - np.amin(data["readoutTime"])

    bin_width = 300 #  secs
    nbins = int(run_duration // bin_width)
    if nbins > 1000:
        bin_width = 3600 #  secs
        nbins = int(run_duration // bin_width)

    if nbins > 1000:
        bin_width = 5 * 3600 #  secs
        nbins = int(run_duration // bin_width)

    print(f"Run duration {run_duration / 3600:.2f} h")
    print(f"Run duration (readout) {run_duration_readout / 3600:.2f} h")
    triggers = np.unique(data["triggerType"])

    times = np.array([dt.datetime.fromtimestamp(ts).astimezone(dt.UTC) for ts in data["triggerTime"]])

    n, bins, _ = ax2.hist(times, nbins, weights=np.full(len(times), 1 / bin_width),
                    histtype="step", color="k", label="total")

    for idx, trigger in enumerate(triggers):
        mask = data["triggerType"] == trigger

        rate = np.sum(mask) / run_duration_readout

        ax.bar(idx, np.sum(mask), label=f"{trigger}: {rate:.3f}")

        ax2.hist(times[mask], bins, histtype="step", label=trigger, weights=np.full(np.sum(mask), 1 / bin_width))


    if 0:
        ax3 = ax2.twinx()
        ax3.plot(times, upwardfacing_radiantThrs / 16777215 * 2.5, "C2--", lw=1, label="RADIANT0 (up)")
        ax3.plot(times, downwardfacing_radiantThrs / 16777215 * 2.5, "C3--", lw=1, label="RADIANT1 (down)")

        ax3.set_ylabel("thresholds")
        # ax3.plot(times, lowTrigThrs / 30, "C1--", lw=1, label="LT / 30")
        # ax3.set_ylim(None, 1.2)
        ax3.legend(loc="upper left", title="Trigger thresholds")

    ax.set_xticks(np.arange(len(triggers)))
    ax.set_xticklabels(triggers, rotation=20)

    ax.legend(title=f"Trigger rates. Total: {len(data['triggerType']) / run_duration_readout:.3f}")

    ax2.set_ylabel("rate / s")
    xfmt = md.DateFormatter('%m-%d %H:%M')
    ax2.xaxis.set_major_formatter(xfmt)
    ax2.xaxis.set_tick_params(rotation=20)
    ax2.set_yscale("log")
    ax2.legend()

    if len(runs) == 1:
        fname = f"station{station}_run{runs[0]}"
    else:
        fname = f"station{station}_run{runs[0]}-{runs[-1]}"


    fig.tight_layout()
    fig.savefig(f"{fname}_trigger_hist.png")

    fig2.tight_layout()
    fig2.savefig(f"{fname}_trigger_vs_time.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyse_triggers.py <dataset_paths>")
        sys.exit(1)

    dataset_paths = sys.argv[1:]

    datasets = Datasets(dataset_paths, skip_incomplete=False)

    data = datasets.eventInfo()
    data = convert_events_information(data)
    
    inf_mask = np.isinf(data["triggerTime"])
    data["triggerTime"][inf_mask] = data["readoutTime"][inf_mask]
    print(f"Found {np.sum(inf_mask)} events with inf trigger time (of {len(inf_mask)} events)")


    plot_triggers(data, datasets.runs, datasets.station, datasets.duration)


