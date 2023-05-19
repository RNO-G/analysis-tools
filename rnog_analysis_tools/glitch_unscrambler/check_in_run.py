import numpy as np
import matplotlib.pyplot as plt
#import NuRadioReco.modules.io.rno_g.readRNOGData
import NuRadioReco.modules.io.rno_g.rnogDataReader
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
parser.add_argument('run', type=int)
parser.add_argument('--n_events', type=int, default=1000000)
parser.add_argument('--events_per_block', type=int, default=10)
args = parser.parse_args()
sampling_rate = 3.2
# channel_ids = np.arange(24, dtype=int)
channel_ids = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
n_cols = 4
n_rows = channel_ids.shape[0] // n_cols
#data_reader = NuRadioReco.modules.io.rno_g.readRNOGData.readRNOGData()
filename = 'station{}/run{}/combined.root'.format(args.station, args.run)
# filename = '/home/henrichs/radiant_data/glitch_measurement_31032021/run{}/combined.root'.format(args.run)
# filename = '/home/henrichs/radiant_data/test_radiant/run{}/combined.root'.format(args.run)
#filename = []
#for fn in range(1):
# #    filename.append('/home/henrichs/radiant_data/long_run_scrambled_data/run{}/combined.root'.format(args.run + fn + 44))
    # filename.append('/home/henrichs/radiant_data/glitch_measurement_10Hz_05042023/combined_run{}.root'.format(args.run + fn))
    # filename.append('/home/henrichs/radiant_data/glitch_measurement_external_power_source/run{}/combined.root'.format(args.run + fn))
    # filename.append('/home/henrichs/radiant_data/glitch_measurement_external_power_source/combined_run{}.root'.format(args.run + fn))
#    filename.append('/home/henrichs/radiant_data/glitch_measurement_external_power_source_10Hz/combined_run{}.root'.format(args.run + fn))
#     print(args.run + fn)
#data_reader.begin(filename)
data_reader = NuRadioReco.modules.io.rno_g.rnogDataReader.RNOGDataReader([filename])
print(filename)
#data_reader = NuRadioReco.modules.io.rno_g.rnogDataReader.RNOGDataReader(filename)

if channel_ids.shape[0] % n_cols > 0:
    n_rows += 1

n_events = min(args.n_events, data_reader.get_n_events())
block_size = args.events_per_block
n_blocks = int(np.ceil(n_events / block_size))
connection_points = np.array([32, 64, 128])

connection_jumps = np.zeros((connection_points.shape[0], channel_ids.shape[0], n_blocks, block_size * (2048 // 128 - 1)))
print(connection_jumps.shape)
connection_jumps[:] = np.nan


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / sigma**2) / 2

i_block = 0

#for i_event, event in enumerate(data_reader.run()):
for i_event, event in enumerate(data_reader.get_events()):
    print(i_event, end='\r')
    station = event.get_station(args.station)
    if i_event >= n_events:
        break
    j_event = i_event - i_block * block_size
    if j_event == block_size:
        i_block += 1
        j_event = 0
    for i_channel, channel_id in enumerate(channel_ids):
        channel = station.get_channel(channel_id)
        trace = channel.get_trace()
        for i_chunk in range(2048 // 128 - 1):
            for i_connection, connection_point in enumerate(connection_points):
                connection_jumps[i_connection, i_channel, i_block, j_event * (2048 // 128 - 1) + i_chunk] = (trace[i_chunk * 128 + connection_point - 1] - trace[i_chunk * 128 + connection_point])

distribution_widths = np.zeros((connection_points.shape[0], channel_ids.shape[0], n_blocks))
for i_connection in range(connection_points.shape[0]):
    for i_channel in range(channel_ids.shape[0]):
        for i_block in range(n_blocks):
            distribution_widths[i_connection, i_channel, i_block] = np.sqrt(np.nanmean((connection_jumps[i_connection, i_channel, i_block] - np.nanmean(connection_jumps[i_connection, i_channel, i_block]))**2))
fig1 = plt.figure(figsize=(6 * n_cols, 3 * n_rows))
for i_channel, channel_id in enumerate(channel_ids):
    ax1_1 = fig1.add_subplot(n_rows, n_cols, i_channel + 1)
    for i_connection, connection_point in enumerate(connection_points):
        ax1_1.plot(
            np.arange(n_blocks) * block_size + .5 * block_size,
            distribution_widths[i_connection, i_channel],
            label='$\Delta$n={}'.format(connection_point)
        )
    ax1_1.set_title('Channel {}'.format(channel_id))
    ax1_1.grid()
    ax1_1.legend()
    ax1_1.set_xlabel('N block')
    ax1_1.set_ylabel('$\delta$U [a.u.]')
fig1.tight_layout()
#plt.show()
plt.savefig("glitches.png")
