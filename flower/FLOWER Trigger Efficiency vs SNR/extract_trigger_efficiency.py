import os
import argparse
import math
import numpy as np
import ROOT
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
import json
import pandas as pd
import csv
import glob 

run_att_map = {}
count = 0
with open('flower_st11_runAtt.csv', newline='') as csvfile:
    for row in csvfile:
        if (count > 0):
            run_att_map[row.split(",")[0]] = row.split(",")[1].split()[0]
        count += 1 

print(run_att_map)

# make sure we have enough arguments to proceed
all_trig_eff = {}
main_path = "/data/condor_builds/users/avijai/RNO/tutorials-rnog/get_daqstatus/flower_traces/traces"

for file in os.listdir(main_path):
    path = main_path + "/" + f"{file}" + "/aux/flower_end.json.gz"
    json_data = pd.read_json(path)

    waveforms_all = {}
    for ch in range(4):
        waveforms_all[f"ch{ch}"] = np.array([json_data["events"][i][f"ch{ch}"] for i in range(len(json_data["events"]))])

    trigger_type = []
    timestamps = []
    pps_flag = []
    for ele in json_data["events"]:
        trigger_type.append(ele["metadata"]["trigger_type"])
        timestamps.append(ele["metadata"]["timestamp"])
        pps_flag.append(ele["metadata"]["pps_flag"])
    

    pps_flag = np.array(pps_flag)
    soft_mask = np.array(trigger_type) == "SOFT"

    run = json_data["run"][0]
    station = json_data["hostname"][0]
    label = f"{run}"

    evts = {}
    for trig, trig_mask in zip(["pps", "no_pps"], [pps_flag, ~pps_flag]):
        evts[trig] = np.sum(trig_mask)
        evts["total"] = len(pps_flag) #get total and triggered number of events
    
    pps_timestamps = []
    for i in range(len(pps_flag)):
        if pps_flag[i] == True:
            pps_timestamps.append(timestamps[i]) #timestamp of cal pulser

    pps_ts = []
    for i in range(len(pps_timestamps)):
        pps_ts.append((pps_timestamps[i] - min(pps_timestamps))/(118*10**(6))) #time spent at each attenuation
    
    #plotting the timestamps
    """
    plt.figure()
    plt.scatter([np.arange(len(pps_ts))], pps_ts)
    plt.title(f'Timestamps Att {run_att_map[label]}')
    #plt.ylim(0, 7.98*10**(13))
    plt.savefig(f'/data/condor_builds/users/avijai/RNO/tutorials-rnog/get_daqstatus/flower_traces/timestamps/ts_run_{run_att_map[label]}.png')
    plt.close()
    """
    
    #get 199 from 1Hz rate of cal pulses multiplied by time spent at each attenuation (~199)

    trig_eff = evts["pps"]/199
    all_trig_eff[float(run_att_map[label])] = trig_eff 

#path to snr files for each attenuation 
indir = "/data/condor_builds/users/avijai/RNO/tutorials-rnog/get_daqstatus/flower_traces/snr_flower_st11"

#plot trigger efficiency vs attenuation
atts = []
trig_eff = []
for att in all_trig_eff:
    atts.append(att)
    trig_eff.append(all_trig_eff[att])

plt.figure()
plt.scatter(atts, trig_eff)
plt.xlabel("Attenuation")
plt.ylabel("Trigger Efficiency")
plt.title("Trigger Efficiency vs Attenuation Station 11 Flower")
plt.savefig("trig_vs_att_11_fl.png")
plt.close()





files = sorted(glob.glob(os.path.join(indir, "*")))


#calculating mean snr value in each attenuation bin 

snr_means = []
snr_std = []
atts = []
atts_12 = []
snr_means_12 = []
for f in files:
    att_one = f.split("/")[-1].split("_")[-1].split(".")[0]
    att_two = f.split("/")[-1].split("_")[-1].split(".")[1]
    att = int(att_one) + 0.1*int(att_two)

    snr_arr = np.load(f)
  
    if (len(snr_arr) != 0):
        atts.append(att)
        snr_means.append(np.average(snr_arr))
        snr_std.append(np.std(snr_arr))
        if (att >= 6 and att <= 13):
            atts_12.append(att)
            snr_means_12.append(np.average(snr_arr))


#exponential fit for attenuation between 6-15 dB


#fit = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  atts_12, snr_means_12,  p0=(17.5, -0.2))
fit = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  atts_12, snr_means_12,  p0=(30, -0.2))

x_vals = np.arange(0,32, 2)
y_vals = []

for x in x_vals:
    y_vals.append(fit[0][0]*np.exp(fit[0][1]*x))
trig_effs = []
for val in x_vals:
    trig_effs.append(all_trig_eff[float(val)])

#plot trigger efficiency vs snr
plt.figure()
plt.scatter(y_vals, trig_effs)
plt.xlabel("SNR")
plt.ylabel("Trigger Efficiency")
plt.title("Trigger Efficiency vs SNR Station 11 Flower")
plt.savefig("trig_vs_snr_11_fl.png")
plt.close()



