import numpy as np
import matplotlib.pyplot as plt
import NuRadioReco.modules.io.rno_g.readRNOGData
from NuRadioReco.utilities import units, fft
import glob
import os
import argparse
import radiotools.helper
import json
import subprocess
import tqdm
import requests
import scipy.optimize

parser = argparse.ArgumentParser()
parser.add_argument('station', type=int)
parser.add_argument('path', type=str)
parser.add_argument('--n_events', type=int, default=1000000)
args = parser.parse_args()
sampling_rate = 3.2
channel_ids = np.arange(24, dtype=int)
# channel_ids = np.array([0, 1, 7, 8, 10, 16, 17, 18, 22, 23])
n_cols = 4
n_rows = channel_ids.shape[0] // n_cols
if channel_ids.shape[0] % n_cols > 0:
    n_rows += 1
connection_points = np.array([32, 64, 128])
linestyles = ['-', '--', ':']

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / sigma**2) / 2
filenames = glob.glob(args.path + '*.root')
distribution_means = np.zeros((3, channel_ids.shape[0], len(filenames)))
distribution_sigmas = np.zeros((3, channel_ids.shape[0], len(filenames)))
distributions_std = np.zeros((3, channel_ids.shape[0], len(filenames)))
run_numbers = np.zeros(len(filenames))
run_times = np.zeros(len(filenames))
rms_trace = np.zeros(2048)
binsize = .005
jump_bins = np.arange(-.1, .1, binsize)
data_reader = NuRadioReco.modules.io.rno_g.readRNOGData.readRNOGData()
for i_file, filename in enumerate(filenames):
    if i_file % 10 == 9:
        print('File {} of {}'.format(i_file + 1, len(filenames)))
    try:
        data_reader.begin(filename)
    except:
        continue
    jumps = np.zeros((3, channel_ids.shape[0], min(data_reader.get_n_events(), args.n_events), 2048 // 128 - 1))
    jumps[:] = np.nan
    for i_event, event in enumerate(data_reader.run()):
        station = event.get_station(args.station)
        if i_event >= args.n_events:
            break
        if i_event == 0:
            run_numbers[i_file] = event.get_run_number()
        for i_channel, channel_id in enumerate(channel_ids):
            rms_trace[:] = 0
            channel = station.get_channel(channel_id)
            trace = channel.get_trace()
            trace_max = np.max(np.abs(trace))
            for i_block in range(2048 // 128 - 1):
                rms_trace[i_block * 128:i_block * 128 + 128] = trace[i_block * 128:i_block * 128 + 128] - np.mean(trace[i_block * 128:i_block * 128 + 128])
            noise_rms = np.sqrt(np.mean(rms_trace**2))
            for i_block in range(2048 // 128 - 1):
                for i_connection, connection_point in enumerate(connection_points):
                    jumps[i_connection, i_channel, i_event, i_block] = (trace[i_block * 128 + connection_point - 1] - trace[i_block * 128 + connection_point]) # / noise_rms
    for i_channel, channel_id in enumerate(channel_ids):
        for i_connection, connection_point in enumerate(connection_points):
            hist = np.histogram(
                (jumps[i_connection, i_channel].flatten()),
                bins=jump_bins,
            )
            try:
                normal_fit = scipy.optimize.curve_fit(
                    gaussian,
                    hist[1][:-1] + .5 * binsize,
                    hist[0]
                )
                distribution_means[i_connection, i_channel, i_file] = normal_fit[0][1]
                distribution_sigmas[i_connection, i_channel, i_file] = normal_fit[0][2]
            except:
                pass
            distributions_std[i_connection, i_channel, i_file] = np.sqrt(np.mean((jumps[i_connection, i_channel] - np.mean(jumps[i_connection, i_channel]))**2))

fig1 = plt.figure(figsize=(8 * n_cols, 4 * n_rows))
for i_channel, channel_id in enumerate(channel_ids):
    ax1_1 = fig1.add_subplot(n_rows, n_cols, i_channel + 1)
    for i_connection, connection_point in enumerate(connection_points):
        ax1_1.scatter(
            run_numbers,
            np.abs(distribution_sigmas[i_connection, i_channel]),
            label='$\Delta$N={}'.format(connection_point),
            c='C{}'.format(i_connection)
        )
        ax1_1.set_xlabel('run #')
        ax1_1.set_ylabel(r'$\sigma_{\Delta U}$ / RMS(U)')
    ax1_1.legend()
    ax1_1.set_title('Channel {}'.format(channel_id))
    ax1_1.grid()
    ax1_1.set_xlabel('run number')
    ax1_1.set_ylabel(r'$\delta$U [a.u.]')
    ax1_1.set_yscale('log')
fig1.tight_layout()
fig1.savefig('plots/connection_points_test_station_{}.png'.format(args.station))
np.savez(
    open('voltage_jumps_station{}.np'.format(args.station), 'wb'),
    run_numbers=run_numbers,
    distribution_means=distribution_means,
    distribution_sigmas=distribution_sigmas,
    distributions_std=distributions_std
)
