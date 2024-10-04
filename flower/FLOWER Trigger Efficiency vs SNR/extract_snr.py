import os
import argparse
import math
import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt
import scipy
import scipy.stats as ss
import scipy.optimize as opt
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq, rfft, irfft
from array import array

#path to the flower traces
main_path = "/data/condor_builds/users/avijai/RNO/tutorials-rnog/get_daqstatus/flower_traces/traces"

#mapping each run to the right attenuation
run_att_map = {}
count = 0
with open('/data/condor_builds/users/avijai/RNO/tutorials-rnog/get_daqstatus/flower_traces/flower_st11_runAtt.csv', newline='') as csvfile:
    for row in csvfile:
        if (count > 0):
            run_att_map[row.split(",")[0]] = row.split(",")[1].split()[0]
        count += 1


for file in os.listdir(main_path):
    path = main_path + "/" + f"{file}" + "/aux/flower_end.json.gz"
    json_data = pd.read_json(path)
    waveforms_all = {}
    for ch in range(4): #get waveforms associated with all channels for each event
        waveforms_all[f"ch{ch}"] = np.array([json_data["events"][i][f"ch{ch}"] for i in range(len(json_data["events"]))])

    trigger_type = []
    timestamps = []
    pps_flag = []
    for ele in json_data["events"]: 
        trigger_type.append(ele["metadata"]["trigger_type"])
        timestamps.append(ele["metadata"]["timestamp"])
        pps_flag.append(ele["metadata"]["pps_flag"])

    snr_arr = []
    for i in range(len(pps_flag)):
        if (pps_flag[i] == True): #cal pulser cut
            snrs = []
            for ch in range(4):
                waveform = json_data["events"][i][f"ch{ch}"]
                noise = np.sqrt(np.mean((np.array(waveform[200:]))**2)) #noise is the in the latter part of wf 
                p2p = np.max(waveform) - np.min(waveform)
                snr = p2p/(2*noise) #calculate snr
                snrs.append(snr)
            sorted_snr = sorted(snrs)
            snr_arr.append(sorted_snr[-2])

    run = json_data["run"][0]
    att = float(run_att_map[f'{run}'])

    np.save(f'/data/condor_builds/users/avijai/RNO/tutorials-rnog/get_daqstatus/flower_traces/snr_flower_st11/snr_{att}.npy', snr_arr)

