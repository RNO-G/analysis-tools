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

from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
import rnog_data.runtable as rt
from NuRadioReco.utilities import units

def reconstruct_data(station_id, channels, overwrite_sampling_rate=2.4, bad_runs_file=None):

    print(f"Station: {station_id} ; Channels: {channels}")

    # Use the RunTable tool to get the runs for a specific station and time range
    RunTable = rt.RunTable()
    testrt = RunTable.get_table(start_time="2024-06-27T00:00:00", 
                                stop_time="2024-08-02T00:00:00", 
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
        "channels": channels

    }

    return result




    


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--station", type=int, default=14)
    parser.add_argument("--channels", type=int, nargs="+", default=list(range(24)))
    parser.add_argument("--overwrite_sampling_rate", type=float, default=2.4)  # Default to 2.4 GHz
    parser.add_argument("--bad_runs_file", type=str, default="bad_runs.csv")
    args = parser.parse_args()

    print(f"Station: {args.station} ; Channels: {args.channels}")
    
    reconstruct_data(station_id=args.station, channels=args.channels, bad_runs_file=args.bad_runs_file)

