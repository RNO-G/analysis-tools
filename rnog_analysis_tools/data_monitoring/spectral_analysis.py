#!/afs/ifh.de/group/radio/software/radiovenv/bin/python

'''
This script can be used to perform a spectral analysis using force triggered RNO-G data on . 
'''

import os
import logging
import warnings
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import datetime
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
import rnog_data.runtable as rt
from NuRadioReco.utilities import units

def reconstruct_data(station_id, channels, overwrite_sampling_rate=2.4, start_time="", stop_time="",bad_runs_file=None):
    '''
    Obtain the FFT of the force triggered RNO-G data for a specific station and channels, as well as the trace array and frequencies.
    '''

    print(f"Station: {station_id} ; Channels: {channels}")

    # Use the RunTable tool to get the runs for a specific station and time range
    RunTable = rt.RunTable()
    
    testrt = RunTable.get_table(start_time=start_time, 
                                stop_time=stop_time, 
                                stations=[station_id], 
                                run_types=['physics']
                                )
    
    # Load bad runs from a CSV file and ensure the file exists before reading
    if os.path.exists(bad_runs_file):
        bad_runs = pd.read_csv(bad_runs_file, header=None)
        bad_runs = bad_runs[0].values
        testrt = testrt[~testrt['run'].isin(bad_runs)]  
    else:
        logging.warning(f"Bad runs file {bad_runs_file} does not exist. Skipping bad run filtering.")

    tableReader = readRNOGData(log_level=logging.CRITICAL)
    tableReader.begin([f"/pnfs/ifh.de/acs/radio/diskonly/data/inbox/station{station_id}/run{run_id}/combined.root" for run_id in testrt['run']],
                      select_triggers=["FORCE"],
                      overwrite_sampling_rate=overwrite_sampling_rate*units.GHz, # Change the sampling rate if needed 
                      convert_to_voltage=True,
                      apply_baseline_correction="approximate")
    
    n_events = len(tableReader.get_events_information())
    print(f"{n_events} were found.")

    spec_arr = np.zeros((len(channels), n_events, 1025))
    norm_arr = np.zeros((len(channels), n_events, 1025))
    trace_arr = np.zeros((len(channels), n_events, 2048))

    run_no = []
    times = []
    event_ids = []

    for i, event in enumerate(tableReader.run()):
        event_id = event.get_id()
        event_ids.append(event_id)
        station = event.get_station()
        time = station.get_station_time()
        times.append(time.to_datetime())
        run_no.append(event.get_run_number())

        for i_ch, ch in enumerate(channels):
            channel = station.get_channel(ch)
            trace = channel.get_trace()
            spec = channel.get_frequency_spectrum()
            norm = np.abs(spec/np.max(abs(spec)))

            if i == 0:
                freqs = channel.get_frequency_axis()

            spec_arr[i_ch, i, :] = np.abs(spec)
            norm_arr[i_ch, i, :] = np.abs(norm)
            trace_arr[i_ch, i, :] = trace

    run_no = np.array(run_no)
    times = np.array(times, dtype="datetime64[s]")
    event_ids = np.array(event_ids)

    # Save the results in a dictionary
    result = {
        "spec_arr": spec_arr,
        "norm_arr": norm_arr,
        "trace_arr": trace_arr,
        "run_no": run_no,
        "times": times,
        "event_ids": event_ids,
        "freqs": freqs,
        "channels": channels,
        "start_time": start_time,
        "stop_time": stop_time,

    }

    return result


def plot_time_averaged_spectrum_surface_channels_unnormalized(result, station_id):
    """
    Plot the time-averaged spectrum for surface channels.
    """
    channels = result['channels']
    spec_arr = result['spec_arr'] #(n_channels, n_events, n_samples)
    freqs = result['freqs']
    start_time = result['start_time']
    stop_time = result['stop_time']
    start_date = start_time.split('T')[0]
    stop_date = stop_time.split('T')[0]

    avg_spectra = spec_arr.mean(axis=1) #(n_channels, n_samples)

    # Style setup
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Arial'],
        'font.size': 17,
        'axes.labelsize': 17,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'lines.linewidth': 1.3,
        'lines.antialiased': True
    })

    plt.figure(figsize=(10, 6))

    # Define known background regions
    excesscolor = 'grey'
    plt.axvspan(0.08, 0.12, color=excesscolor, alpha=0.3, label='Excess')

    wb_color = "mediumvioletred"
    plt.axvline(x=0.402, color=wb_color, linestyle='--', linewidth=1.2, label='Weather Balloon')

    other_bkg = "mediumseagreen"
    plt.axvspan(0.278, 0.285, color=other_bkg, alpha=0.3, label='periodic signal')
    plt.axvspan(0.482, 0.485, color=other_bkg, alpha=0.3, label='periodic signal')
    plt.axvspan(0.240, 0.272, color=other_bkg, alpha=0.3, label='periodic signal')
    plt.axvspan(0.358, 0.378, color=other_bkg, alpha=0.3, label='periodic signal')
    plt.axvspan(0.136, 0.139, color=other_bkg, alpha=0.3, label='Walkie Talkie')
    plt.axvspan(0.151, 0.157, color=other_bkg, alpha=0.3, label='Walkie Talkie')
    plt.axvspan(0.125, 0.127, color=other_bkg, alpha=0.3, label='Aircraft')

    region_legend_lines = [
            Line2D([0], [0], color=excesscolor, lw=6, alpha=0.3, label='Excess Region'),
            Line2D([0], [0], color=wb_color, linestyle='--', label='Weather Balloon'),
            Line2D([0], [0], color=other_bkg, lw=6, alpha=0.3, label='Other Background')]
    legend_regions = plt.legend(handles=region_legend_lines,
                                loc='upper left', 
                                bbox_to_anchor=(0.818, 0.65), 
                                frameon=True)
    plt.gca().add_artist(legend_regions)

    # Station 14 has a different surface channel configuration with 4 up and 4 down channels.
    if station_id == 14:
        # Define surface channels for station 14
        surface_channels = [12,13,14,15,16,17,18,19]
        # Define up and down channels for station 14
        up_channels = [13, 15, 16, 18]
        down_channels = [12, 14, 17, 19]

        # Define colors for each channel (oranges for down, blues for up)
        colors = {
            12: "#e6550d",
            13: "#1f77b4",
            14: "#f16913",
            15: "#4292c6",
            16: "#6baed6",
            17: "#fd8d3c",
            18: "#9ecae1",
            19: "#fdae6b",
        }

        # Plot the average spectrum for each channel
        for i, ch in enumerate(channels):
            if ch in surface_channels:
                label = f"Ch {ch}"
                plt.plot(freqs, avg_spectra[i], color=colors[ch], alpha=0.8, label=label)

                # Up and Down channel legends (only those present)
                handles_down = [Line2D([0], [0], color=colors[ch], lw=2) for ch in down_channels if ch in channels]
                labels_down = [f"Ch {ch}" for ch in down_channels if ch in channels]

                handles_up = [Line2D([0], [0], color=colors[ch], lw=2) for ch in up_channels if ch in channels]
                labels_up = [f"Ch {ch}" for ch in up_channels if ch in channels]

                legend_channels = plt.legend(handles=handles_up + handles_down,
                                            labels=labels_up + labels_down,
                                            title="   Upward             Downward",
                                            ncol=2, 
                                            loc='upper right', 
                                            frameon=True, 
                                            columnspacing=1.5)
                plt.gca().add_artist(legend_channels)

                plt.text(0.02, 0.96,f'St {station_id} | {start_date} to {stop_date}',
                        transform=plt.gca().transAxes,fontsize=14,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(facecolor='white',alpha=0.8, edgecolor='lightgray'))

                plt.xlabel('Frequency [GHz]')
                plt.ylabel('Time-Averaged Spectrum [V]')
                plt.grid(True, which="both", axis="both", linestyle=':', linewidth=0.3, alpha=0.7)
                #plt.xlim(0, 0.8)
                plt.tight_layout()
                plt.show()

            else:
                raise ValueError(f"Channel {ch} is not a surface channel for station {station_id}. Please check the channel list.")

                

                

                
                
                

                







            else:
                





    

    


    




    


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--station", type=int, default=14)
    parser.add_argument("--channels", type=int, nargs="+", default=list(range(24)))
    parser.add_argument("--overwrite_sampling_rate", type=float, default=2.4)  # Default to 2.4 GHz
    parser.add_argument("--bad_runs_file", type=str, default=None)
    parser.add_argument("--start_time", type=str, default="2024-06-27T00:00:00")
    parser.add_argument("--stop_time", type=str, default="2024-08-02T00:00:00")
    args = parser.parse_args()

    print(f"Station: {args.station} ; Channels: {args.channels}")
    
    results = reconstruct_data(station_id=args.station, 
                    channels=args.channels,
                    start_time=args.start_time,
                    stop_time=args.stop_time,
                    bad_runs_file=args.bad_runs_file)

