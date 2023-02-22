import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.signal
import json
import os.path
build_instructions = json.load(open('build_info.json', 'r'))
station_ids = range(1, 11)
channel_time_delays = np.zeros((len(station_ids), len(build_instructions['general']['fiber_mappings'].keys())))
fiber_delay_dict = {}
for i_station, station_id in enumerate(station_ids):
    plt.close('all')
    n_plots = len(build_instructions['general']['fiber_mappings'].keys())
    n_rows = 5
    n_cols = n_plots // n_rows
    if n_plots % n_cols > 0:
        n_cols += 1
    fig1 = plt.figure(figsize=(n_rows * 2, 16))
    fiber_delay_dict[str(station_id)] = {}
    for i_channel, channel in enumerate(build_instructions['general']['fiber_mappings'].keys()):
        filename = '/home/welling/RadioNeutrino/data/FIBER_CALIB_RNO_G-20220923T164319Z-001/FIBER_CALIB_RNO_G/FinalData/{}{}_FULL_P.csv'.format(
            station_id,
            build_instructions['general']['fiber_mappings'][channel]
        )
        if not os.path.exists(filename):
            filename = '/home/welling/RadioNeutrino/data/FIBER_CALIB_RNO_G-20220923T164319Z-001/FIBER_CALIB_RNO_G/FinalData/{}{}_FULL_P.csv'.format(
                station_id,
                build_instructions['general']['fiber_mappings'][channel].upper()
            )
        if not os.path.exists(filename):
            filename = '/home/welling/RadioNeutrino/data/FIBER_CALIB_RNO_G-20220923T164319Z-001/FIBER_CALIB_RNO_G/FinalData/{}{}_P.csv'.format(
                station_id,
                build_instructions['general']['fiber_mappings'][channel].upper()
            )
        if not os.path.exists(filename):
            filename = '/home/welling/RadioNeutrino/data/FIBER_CALIB_RNO_G-20220923T164319Z-001/FIBER_CALIB_RNO_G/FinalData/{}{}_FULL_P.csv'.format(
                station_id,
                build_instructions['general']['fiber_mappings'][channel].lower()
            )
        if not os.path.exists(filename):
            print('Warning: No fiber measurement found for station {}, channel {}!'.format(station_id, channel))
            channel_time_delays[i_station, i_channel] = np.nan
            continue
        data = np.genfromtxt(
            filename,
            delimiter=',',
            skip_header=17,
            skip_footer=1
        )
        frequency_filter = (data[:, 0] > 100.e6) & (data[:, 0] < 500.e6)
        phase_fit = np.polyfit(
            data[:, 0],
            np.unwrap(data[:, 1] * np.pi / 180.),
            1
        )
        time_delay = -phase_fit[0] / 2. / np.pi
        fiber_delay_dict[str(station_id)][str(channel)] = time_delay * 1.e9
        channel_time_delays[i_station, i_channel] = time_delay
        ax1_1 = fig1.add_subplot(2 * n_rows, n_cols, i_channel + 1)
        ax1_2 = fig1.add_subplot(2 * n_rows, n_cols, i_channel + 1 + n_rows * n_cols)
        ax1_1.plot(
            data[:, 0],
            np.unwrap(data[:, 1] * np.pi / 180.) / np.pi
        )
        ax1_2.plot(
            data[:, 0],
            (np.unwrap(data[:, 1] * np.pi / 180.) - phase_fit[0] * data[:, 0] - phase_fit[1]) / np.pi
        )
        ax1_1.grid()
        ax1_2.grid()
    fig1.tight_layout()
    fig1.savefig('plots/fiber_delays/phases_{}.png'.format(station_id))

fig2 = plt.figure(figsize=(8, 16))
time_diff_bins = np.arange(-2, 2, .2)
for i_channel in range(channel_time_delays.shape[1]):
    ax2_1 = fig2.add_subplot(channel_time_delays.shape[1] //2 + channel_time_delays.shape[1] % 2, 2, i_channel + 1)
    ax2_1.hist(
        np.clip(channel_time_delays[:, i_channel] - np.nanmean(channel_time_delays[:, i_channel]), a_min=time_diff_bins[0], a_max=time_diff_bins[-1]) * 1.e9,
        bins=time_diff_bins
    )
    ax2_1.set_title('Channel {}'.format(list(build_instructions['general']['fiber_mappings'].keys())[i_channel]))
    ax2_1.set_xlabel('fiber delay deviation from mean [ns]')
    ax2_1.grid()
fig2.tight_layout()
fig2.savefig('plots/fiber_delays/fiber_delay_hist.png')

fig3 = plt.figure(figsize=(6, 6))
ax3_1 = fig3.add_subplot(111)
ax3_1.hist(
    np.clip(channel_time_delays - np.nanmean(channel_time_delays, axis=0), a_min=time_diff_bins[0], a_max=time_diff_bins[-1]).flatten() * 1.e9,
    bins=time_diff_bins,
    edgecolor='k'
)
ax3_1.grid()
ax3_1.set_xlabel('fiber delay deviation from mean [ns]')
fig3.tight_layout()
fig3.savefig('plots/fiber_delays/fiber_delay_differences.png')

fiber_delay_dict['4']['1'] = 710.
json.dump(fiber_delay_dict, open('fiber_delays.json', 'w'), indent=2)