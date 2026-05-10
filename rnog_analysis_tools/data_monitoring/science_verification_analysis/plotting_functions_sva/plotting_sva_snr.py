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
    channels = list(channels)
    n_channels = len(channels)

    if day_interval is None:
        day_interval = choose_day_interval(times) 

    fig, axs = plt.subplots(nrows,ncols,figsize=(15,24),sharex=True)
    axs = np.array(axs)

    for idx, ch in enumerate(channels):
        r = idx//ncols
        c = idx%ncols
        ax = axs[r,c]
        good_mask = ~flag[ch]
        ax.scatter(times[good_mask], np.log10(snr_arr[ch][good_mask]), s=8,alpha=0.25, color="gray")
        zex = np.abs(z_log[ch]) - k_list[ch]
        zex = np.clip(zex,0,None)
        sc = ax.scatter(times[flag[ch]], np.log10(snr_arr[ch][flag[ch]]), s=8,c=zex[flag[ch]], cmap="Reds")
        cax = ax.inset_axes([1.02,0.1,0.05,0.8])
        plt.colorbar(sc,cax=cax)
        ax.grid(alpha=0.4)
        ax.text(0.85, 0.95, f"Ch {ch}", transform = ax.transAxes, ha = "left",va = "top", bbox = dict(boxstyle = "round, pad = 0.25", facecolor = "white", alpha = 0.8))
    
    for idx in range(n_channels, nrows*ncols):
        r = idx//ncols
        c = idx%ncols
        axs[r, c].set_visible(False)

    red = plt.cm.Reds(0.6)
    legend_handles = [Line2D([0],[0],marker="o",color="none",markeredgecolor="gray",markerfacecolor="gray",markersize=6,label=r"$|z|\leq k$"),
                    Line2D([0],[0],marker="o",color="none",markeredgecolor=red,markerfacecolor=red,markersize=6,label=r"$|z|>k$")]                 
    axs[0, 0].legend(handles = legend_handles, loc = "upper left")

    ticks_ax = axs[-1,0]
    time_span = (times.max() - times.min()).total_seconds() / 86400.0  # in days

    if time_span < 1:
        # Use 2h ticks if less than 1 day
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    elif time_span < 3:
        # Use 6h ticks if less than 3 days
        ticks_ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    else:
        # Use day ticks otherwise
        ticks_ax.xaxis.set_major_locator(mdates.DayLocator(interval = day_interval))
        ticks_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz = timezone.utc))

    fig.autofmt_xdate()

    plt.subplots_adjust(bottom = 0.11, wspace = 0.38, left=0.08)
    fig.supxlabel("Date [UTC]", x = 0.5, y = 0.06)
    fig.supylabel(r"$\log_{10}(\mathrm{SNR})$", x = 0.02)
    plt.savefig(os.path.join(save_location,f"snr_against_time_{station_id}_{run_label}.pdf"))
            