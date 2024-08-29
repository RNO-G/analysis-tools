import numpy as np
import sys
from matplotlib import pyplot as plt
from collections import defaultdict
import matplotlib.dates as md
import datetime as dt
from matplotlib import cm, colors

from rnog_analysis_tools.glitch_unscrambler import glitch_detection_per_event

from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.RNO_G.channelBlockOffsetFitter import fit_block_offsets


"""
This script can be used to analyze RNO-G data runs. Examples are plotting the trigger rate. RMS of the waveforms per channel type and trigger type and other things.

Pass a run directory with rootified files as argument to the script.
"""


def convert_events_information(event_info, convert_to_arrays=True):

    data = defaultdict(list)

    for ele in event_info.values():
        for k, v in ele.items():
            data[k].append(v)

    if convert_to_arrays:
        for k in data:
            data[k] = np.array(data[k])

    return data


def plot_blockoffset(reader, event_info, wfs):

    mask = event_info["triggerType"] == "FORCE"
    wfs = np.copy(wfs)[mask]

    # # Manual block splitting and median calculation
    # wfs_blocks = np.split(wfs, 16, axis=-1)
    # wfs_blocks = np.swapaxes(np.array(wfs_blocks), 0, 2)
    # block_medians = np.median(wfs_blocks, axis=-1).reshape(24, -1)

    fit_blocks = []
    for wfs_channel in np.swapaxes(wfs, 0, 1):
        fit_blocks.append(np.array([fit_block_offsets(wf, sampling_rate=2.4) for wf in wfs_channel]))

    fit_blocks = np.array(fit_blocks)
    fit_blocks = fit_blocks.reshape(24, -1  )

    fig, ax = plt.subplots()
    # parts = ax.violinplot(block_medians.T, np.arange(24), showextrema=True, showmedians=True,
                        #   vert=False, side="high", widths=1.8)
    parts2 = ax.violinplot(
        fit_blocks.T, np.arange(24), showextrema=True, showmedians=True,
        vert=False, side="high", widths=1.8)

    ax.plot(np.nan, np.nan, label=f"{np.sum(mask)} forced triggers", color="k")
    # ax.plot(np.nan, np.nan, label="median", color="C0")
    ax.plot(np.nan, np.nan, label="fit", color="C0")

    # norm = colors.Normalize(vmin=np.amin(means), vmax=np.amax(means), clip=False)
    # sm = cm.ScalarMappable(norm, cmap="Oranges")

    # for idx, pc in enumerate(parts['bodies']):
    #     # pc.set_facecolor(sm.to_rgba(means[idx]))
    #     pc.set_alpha(0.5)

    for p in [parts2]:
        p["cmins"].set_linewidth(0.2)
        p["cmaxes"].set_linewidth(0.2)
        p["cbars"].set_linewidth(0.5)
        p["cmins"].set_color("k")
        p["cmaxes"].set_color("k")
        p["cbars"].set_color("k")
        p["cmedians"].set_color("k")
        p["cmedians"].set_linewidth(1)

    ax.set_xlabel("median ADC per block")
    ax.set_ylabel("Channel id")

    ax.legend()
    ax.grid()

    if len(reader._datasets) == 1:
        dset = reader._datasets[0]
        fname = f"station{dset.station}_run{dset.run}"
    else:
        assert len(np.unique([dset.station for dset in reader._datasets]))
        station = reader._datasets[0].station
        fname = f"station{station}_run{reader._datasets[0].run}-{reader._datasets[-1].run}"

    fig.tight_layout()
    fig.savefig(f"{fname}_offsets.png")


def plot_glitching(reader, event_info, wfs):

    apply_norm = True

    ts = np.array([glitch_detection_per_event.is_channel_scrambled(wf) for wf in wfs.reshape(-1, 2048)]).reshape(wfs.shape[:2])
    std = np.std(wfs, axis=-1)

    if apply_norm:
        ts /= std

    fig, ax = plt.subplots()

    means = np.mean(ts, axis=0)
    parts = ax.violinplot(ts, np.arange(24), showextrema=True, showmedians=True,
                          vert=False, side="high", widths=1.8)

    norm = colors.Normalize(vmin=np.amin(means), vmax=np.amax(means), clip=False)
    sm = cm.ScalarMappable(norm, cmap="Oranges")

    for idx, pc in enumerate(parts['bodies']):
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
    cb.set_label("mean value ts")

    # for ch in range(24):
    #     mean = np.mean(ts[:, ch])
    #     c = "k" if mean < 100 else "C3"
    #     pretty_trans_hist(ax, ts[:, ch], bins, color=c, label=ch)
    # ax.legend(ncols=2)
    # ax.set_yscale("log")

    if apply_norm:
        ax.set_xlabel("norm. glitching test statistics")
    else:
        ax.set_xlabel("glitching test statistics")

    ax.set_ylabel("channel")
    if not apply_norm:
        x1, x2 = ax.get_xlim()
        ax.set_xlim(max(-50000, x1), min(50000, x2))
    else:
        x1, x2 = ax.get_xlim()
        ax.set_xlim(max(-2000, x1), min(2000, x2))

    ax.grid()

    if len(reader._datasets) == 1:
        dset = reader._datasets[0]
        fname = f"station{dset.station}_run{dset.run}"
    else:
        assert len(np.unique([dset.station for dset in reader._datasets]))
        station = reader._datasets[0].station
        fname = f"station{station}_run{reader._datasets[0].run}-{reader._datasets[-1].run}"

    if apply_norm:
        fname += "_norm"

    fig.tight_layout()
    fig.savefig(f"{fname}_glitching.png")


def plot_rms(reader, event_info, wfs):

    times = np.array([dt.datetime.fromtimestamp(ts) for ts in event_info["readoutTime"]])
    # t_mask = times > dt.datetime.fromisoformat('2024-06-26T23:00:00')
    # wfs = wfs[t_mask]
    # times = times[t_mask]

    fig, ax = plt.subplots(figsize=(9, 5))

    std = np.std(wfs, axis=-1)

    # ax.errorbar(np.arange(24), np.mean(std, axis=0), np.std(std, axis=0), ls="", marker="o", color="k", label="all")

    for idx, trigger in enumerate(np.unique(event_info["triggerType"])):
        mask = event_info["triggerType"] == trigger
        # ax.errorbar(np.arange(24) + idx / 8, np.mean(std[mask], axis=0), np.std(std[mask], axis=0), ls="", marker="o", label=trigger)

        parts = ax.violinplot(std[mask], np.arange(24) + idx / 6, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor(f"C{idx}")
            pc.set_alpha(0.5)

        parts["cmins"].set_linewidth(0.2)
        parts["cmaxes"].set_linewidth(0.2)
        parts["cbars"].set_linewidth(0.5)

        ax.plot(np.nan, np.nan, label=f"{trigger}: {len(std[mask])}", color=f"C{idx}")


    ax.axvspan(11.8, 20.8, color="grey", alpha=0.3, label="LPDAS")
    ax.axvspan(-0.2, 3.8, color="C5", alpha=0.3, label="PA")
    ax.axvspan(4.8, 7.8, color="C6", alpha=0.3, label="Upper VPols")

    ax.axvspan(8.8, 10.8, color="C4", alpha=0.3, label="Helper VPols")
    ax.axvspan(21.8, 23.8, color="C4", alpha=0.3)

    ax.axvspan(3.8, 4.8, color="C10", alpha=0.3, label="HPols")
    ax.axvspan(7.8, 8.8, color="C10", alpha=0.3)
    ax.axvspan(10.8, 11.8, color="C10", alpha=0.3)
    ax.axvspan(20.8, 21.8, color="C10", alpha=0.3)

    # ax.axvline(11.8, color="k", lw=0.5, ls="--")
    # ax.axvline(19.8)
    ax.legend(loc="upper left", bbox_to_anchor=[1, 1.02], fontsize="small")

    ax.set_xlabel("channels")
    ax.set_ylabel("std of waveforms / ADC")

    if len(reader._datasets) == 1:
        dset = reader._datasets[0]
        fname = f"station{dset.station}_run{dset.run}"
    else:
        assert len(np.unique([dset.station for dset in reader._datasets]))
        station = reader._datasets[0].station
        fname = f"station{station}_run{reader._datasets[0].run}-{reader._datasets[-1].run}"

    fig.tight_layout()
    fig.savefig(f"{fname}_rms_hist.png")

    fig, axs = plt.subplots(5, 1, figsize=(12, 6), sharex=True, sharey=False,
                            gridspec_kw=dict(hspace=0, wspace=0, left=0.06, bottom=0.09, right=0.99,
                                             top=0.85))

    channel_groups = {
        "PA": [0, 1, 2, 3],
        "HPols": [4, 8, 11, 21],
        "Upper VPols": [5, 6, 7],
        "Helper VPols": [9, 10, 22, 23],
        "LPDAs": list(range(12, 21),)
    }

    for cg, ax in zip(channel_groups, axs.flatten()):
        for idx, trigger in enumerate(np.unique(event_info["triggerType"])):
            mask = event_info["triggerType"] == trigger
            ax.plot(times[mask], std[mask][:, channel_groups[cg]], f"C{idx}.", markersize=1)
        ax.plot(np.nan, np.nan, "k.", label=cg)
        ax.legend()

    for idx, trigger in enumerate(np.unique(event_info["triggerType"])):
        axs[0].plot(np.nan, np.nan, f"C{idx}.", label=trigger)
    axs[0].legend(ncols=5)


    for ax in axs[:-1]:
        ax.spines['bottom'].set_linewidth(0.1)

    xfmt = md.DateFormatter('%m-%d %H:%M')
    axs[-1].xaxis.set_major_formatter(xfmt)
    axs[-1].xaxis.set_tick_params(rotation=20)
    fig.supylabel("RMS")

    # ax2, _, bins, _ = add_histogram_on_axis(fig, axs[0], times[event_info["triggerType"] == "FORCE"], 50)
    # for idx, trigger in enumerate(np.unique(event_info["triggerType"])):
    #     if trigger == "FORCE":
    #         continue
    #     mask = event_info["triggerType"] == trigger
    #     pretty_trans_hist(ax2, times[mask], bins)


    plt.savefig(f"{fname}_rms_vs_time.png")


def plot_triggers(reader, data):

    dset = reader._datasets[0]
    if dset.station != 14:
        downwardfacing_radiantThrs = np.mean(data["radiantThrs"][:, [12, 14, 15, 17, 18, 20]], axis=1)
        upwardfacing_radiantThrs = np.mean(data["radiantThrs"][:, [13, 16, 19]], axis=1)
    else:
        downwardfacing_radiantThrs = np.mean(data["radiantThrs"][:, [12, 14, 16, 18]], axis=1)
        upwardfacing_radiantThrs = np.mean(data["radiantThrs"][:, [13, 15, 17, 19]], axis=1)

    lowTrigThrs = np.mean(data["lowTrigThrs"], axis=1)

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    run_duration = np.amax(data["triggerTime"]) - np.amin(data["triggerTime"])
    run_duration_readout = np.amax(data["readoutTime"]) - np.amin(data["readoutTime"])
    bin_width = 300 #  secs
    nbins = int(run_duration // bin_width)
    if nbins > 1000:
        bin_width = 3600 #  secs
        nbins = int(run_duration // bin_width)

    print(f"Run duration {run_duration / 3600:.2f} h")
    print(f"Run duration (readout) {run_duration_readout / 3600:.2f} h")
    triggers = np.unique(data["triggerType"])

    times = np.array([dt.datetime.fromtimestamp(ts).astimezone(dt.UTC) for ts in data["triggerTime"]])
    # times_num = md.date2num(times)

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

    if len(reader._datasets) == 1:
        dset = reader._datasets[0]
        fname = f"station{dset.station}_run{dset.run}"
    else:
        assert len(np.unique([dset.station for dset in reader._datasets]))
        station = reader._datasets[0].station
        fname = f"station{station}_run{reader._datasets[0].run}-{reader._datasets[-1].run}"


    fig.tight_layout()
    fig.savefig(f"{fname}_trigger_hist.png")

    fig2.tight_layout()
    fig2.savefig(f"{fname}_trigger_vs_time.png")


if __name__ == "__main__":

    n_files = len(sys.argv[1:])    
    n_batches = n_files // 100 + 1

    wfs = []
    event_info = defaultdict(list)

    for batch in np.array_split(np.array(sys.argv[1:]), n_batches):
        reader = readRNOGData(load_run_table=False)
        reader.begin(
            batch, convert_to_voltage=False, overwrite_sampling_rate=2.4, check_trigger_time=False, mattak_kwargs=dict(skip_incomplete=False))

        event_info_tmp = reader.get_events_information(
            keys=["triggerType", "triggerTime", "readoutTime", "radiantThrs", "lowTrigThrs", "hasWaveforms"])

        wfs_tmp = reader.get_waveforms(max_events=None, override_skip_incomplete=True)
        print(f"Found {len(wfs_tmp)} waveforms")
        event_info_tmp = convert_events_information(event_info_tmp, False)

        wfs += list(wfs_tmp)
        
        for key, value in event_info_tmp.items():
            event_info[key] += value

        
    for key, value in event_info.items():
        event_info[key] = np.array(value)
    wfs = np.array(wfs)
    print(wfs.shape)

    inf_mask = np.isinf(event_info["triggerTime"])
    event_info["triggerTime"][inf_mask] = event_info["readoutTime"][inf_mask]
    print(f"Found {np.sum(inf_mask)} events with inf trigger time (of {len(inf_mask)} events)")

    plot_triggers(reader, event_info)

    for key, value in event_info.items():
        event_info[key] = value[event_info["hasWaveforms"]]

    plot_glitching(reader, event_info, wfs)
    plot_rms(reader, event_info, wfs)
    plot_blockoffset(reader, event_info, wfs)
