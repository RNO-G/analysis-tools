import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
from datetime import timezone

#### SNR Plots ####

def choose_day_interval(times):
    times = pd.to_datetime(times, utc=True)
    total_days = (times.max() - times.min()).days

    if total_days < 10:
        return 1
    elif total_days < 20:
        return 2
    elif total_days < 40:
        return 4
    elif total_days < 80:
        return 7
    elif total_days < 150:
        return 10
    elif total_days < 300:
        return 15
    elif total_days < 600:
        return 30
    else:
        return 60
    
def plot_snr_against_time(station_id,times,snr_arr,flag,z_log,k_list,channels,save_location,run_label,nrows=12,ncols=2,day_interval=None):
    times = pd.to_datetime(times,utc=True)
    times = times.tz_convert(None)

    channels = list(channels)
    n_channels = len(channels)

    if day_interval is None:
        day_interval = choose_day_interval(times) 

    fig, axs = plt.subplots(nrows,ncols,figsize=(15,24),sharex=True)
    axs = np.array(axs)

    time_span = (times.max() - times.min()).total_seconds() / 86400.0

    for idx, ch in enumerate(channels):
        r = idx//ncols
        c = idx%ncols
        ax = axs[r,c]

        good_mask = ~flag[ch]
        plot_mask = good_mask & ~pd.isna(times)

        ax.scatter(times[plot_mask], np.log10(snr_arr[ch][plot_mask]), s=8,alpha=0.25, color="gray", rasterized=True)

        zex = np.abs(z_log[ch]) - k_list[ch]
        zex = np.clip(zex,0,None)

        sc = ax.scatter(times[flag[ch]], np.log10(snr_arr[ch][flag[ch]]), s=8,c=zex[flag[ch]], cmap="Reds", rasterized=True)

        cax = ax.inset_axes([1.02,0.1,0.05,0.8])
        plt.colorbar(sc,cax=cax, label=r"$|z|-k$")

        if time_span < 1:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
        elif time_span < 3:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=timezone.utc))

        ax.set_xlabel("Date [UTC]")
        ax.set_ylabel(r"$\log_{10}(\mathrm{SNR})$")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=0.4)
        ax.text(0.85, 0.95, f"Ch {ch}", transform=ax.transAxes, ha="left", va="top", bbox=dict(boxstyle="round, pad=0.25", facecolor="white", alpha=0.8))
    
    for idx in range(n_channels, nrows*ncols):
        r = idx//ncols
        c = idx%ncols
        axs[r,c].set_visible(False)

    red = plt.cm.Reds(0.6)

    legend_handles = [
        Line2D([0],[0],marker="o",color="none",markeredgecolor="gray",markerfacecolor="gray",markersize=6,label=r"$|z|\leq k$"),
        Line2D([0],[0],marker="o",color="none",markeredgecolor=red,markerfacecolor=red,markersize=6,label=r"$|z|>k$")
    ]                 

    axs[0,0].legend(handles=legend_handles, loc="upper left")

    fig.autofmt_xdate()

    plt.subplots_adjust(bottom=0.07, wspace=0.38, hspace=0.45, left=0.08)
    plt.savefig(os.path.join(save_location,f"snr_against_time_{station_id}_{run_label}.pdf"))
    plt.close(fig)


def plot_snr_against_time_per_trigger(station_id,times,snr_arr,channels,save_location,run_label,nrows=12,ncols=2,day_interval=None,color="blue",triggerlabel=""):
    times = pd.to_datetime(times,utc=True)
    times = times.tz_convert(None)

    channels = list(channels)
    n_channels = len(channels)

    if day_interval is None:
        day_interval = choose_day_interval(times) 

    fig, axs = plt.subplots(nrows,ncols,figsize=(15,24),sharex=True)
    axs = np.array(axs)

    time_span = (times.max() - times.min()).total_seconds() / 86400.0

    for idx, ch in enumerate(channels):
        r = idx//ncols
        c = idx%ncols
        ax = axs[r,c]

        ax.scatter(times, np.log10(snr_arr[ch]), s=8,alpha=0.25, color=color, rasterized=True)

        if time_span < 1:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
        elif time_span < 3:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=timezone.utc))

        ax.set_xlabel("Date [UTC]")
        ax.set_ylabel(r"$\log_{10}(\mathrm{SNR})$")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=0.4)
        ax.text(0.85, 0.95, f"Ch {ch}", transform=ax.transAxes, ha="left", va="top", bbox=dict(boxstyle="round, pad=0.25", facecolor="white", alpha=0.8))
    
    for idx in range(n_channels, nrows*ncols):
        r = idx//ncols
        c = idx%ncols
        axs[r,c].set_visible(False)

    legend_handles = [
        Line2D([0],[0],marker="o",color="none",markeredgecolor=color,markerfacecolor=color,markersize=6,label=r"SNR values")
    ]                 

    axs[0,0].legend(handles=legend_handles, loc="upper left")

    fig.autofmt_xdate()

    plt.subplots_adjust(bottom=0.07, wspace=0.38, hspace=0.45, left=0.08)
    plt.savefig(os.path.join(save_location,f"snr_against_time_{station_id}_{run_label}_{triggerlabel}.pdf"))
    plt.close(fig)

def plot_snr_against_time_single_channel(
    station_id,
    times,
    snr_arr,
    channel,
    save_location,
    run_label,
    flag=None,
    z_log=None,
    k_list=None,
    day_interval=None,
    triggerlabel=""
):
    times = pd.to_datetime(times, utc=True)
    times = times.tz_convert(None)

    if day_interval is None:
        day_interval = choose_day_interval(times)

    fig, ax = plt.subplots(figsize=(10,5))

    if flag is None:
        ax.scatter(times, np.log10(snr_arr[channel]), s=8, alpha=0.25, rasterized=True)

        legend_handles = [
            Line2D([0],[0],marker="o",color="none",markersize=6,label="SNR values")
        ]
    else:
        good_mask = ~flag[channel]
        plot_mask = good_mask & ~pd.isna(times)

        ax.scatter(
            times[plot_mask],
            np.log10(snr_arr[channel][plot_mask]),
            s=8,
            alpha=0.25,
            color="gray",
            rasterized=True
        )

        zex = np.abs(z_log[channel]) - k_list[channel]
        zex = np.clip(zex,0,None)

        sc = ax.scatter(
            times[flag[channel]],
            np.log10(snr_arr[channel][flag[channel]]),
            s=8,
            c=zex[flag[channel]],
            cmap="Reds",
            rasterized=True
        )

        plt.colorbar(sc, ax=ax, label=r"$|z|-k$")

        red = plt.cm.Reds(0.6)

        legend_handles = [
            Line2D([0],[0],marker="o",color="none",markeredgecolor="gray",markerfacecolor="gray",markersize=6,label=r"$|z|\leq k$"),
            Line2D([0],[0],marker="o",color="none",markeredgecolor=red,markerfacecolor=red,markersize=6,label=r"$|z|>k$")
        ]

    time_span = (times.max() - times.min()).total_seconds() / 86400.0

    if time_span < 1:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    elif time_span < 3:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=timezone.utc))

    ax.set_xlabel("Date [UTC]")
    ax.set_ylabel(r"$\log_{10}(\mathrm{SNR})$")
    ax.grid(alpha=0.4)

    ax.text(
        0.85,
        0.95,
        f"Ch {channel}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round, pad=0.25", facecolor="white", alpha=0.8)
    )

    ax.legend(handles=legend_handles, loc="upper left")

    fig.autofmt_xdate()

    plt.tight_layout()

    suffix = f"_{triggerlabel}" if triggerlabel else ""

    plt.savefig(
        os.path.join(
            save_location,
            f"snr_against_time_ch{channel}_{station_id}_{run_label}{suffix}.pdf"
        )
    )

    plt.close(fig)