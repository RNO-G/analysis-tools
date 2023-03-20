import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, fft
import scipy.interpolate
import scipy.signal
import scipy.ndimage
import json

"""
This scipt calculates the cable delays for the RNO-G surface channels from pre-deployment measurements done by Dan Southall and
Kaeli Hughes. Before running it, download the measurement data from https://drive.google.com/drive/folders/1Cqk2t0Lu2i6V74GxBrIoJyHYvPE6Krd5?usp=sharing
and place it in the same folder as this file.
"""

station_ids = np.arange(1, 11, dtype=int)
cable_ids = np.arange(1, 10, dtype=int)
n_convolve = 30
filter_degree = 5

sampling_rate = 4.8
plot_times = np.arange(0, 2000) / sampling_rate
plot_freqs = np.fft.rfftfreq(plot_times.shape[0], 1. / sampling_rate)
cable_delay_dict = {}
for i_station, station_id in enumerate(station_ids):

    plt.close('all')
    cable_delay_dict[str(station_id)] = {}
    fig1 = plt.figure(figsize=(18, 12))
    fig2 = plt.figure(figsize=(18, 12))
    fig3 = plt.figure(figsize=(18, 12))
    # ax2_1.set_title('Cable {}'.format(cable_id))
    # ax1_1.set_title('Cable {}'.format(cable_id))
    # ax2_1.set_title('Cable {}'.format(cable_id))
    for i_cable, cable_id in enumerate(cable_ids):
        cable_log = np.genfromtxt(
            'RNO-G Cable Data/s21 data/station {}/S21_SRX{}_{}_LOG.csv'.format(station_id, station_id, cable_id),
            skip_header=17,
            skip_footer=1,
            delimiter=','
        )
        cable_phase = np.genfromtxt(
            'RNO-G Cable Data/s21 data/station {}/S21_SRX{}_{}_PHASE.csv'.format(station_id, station_id, cable_id),
            skip_header=17,
            skip_footer=1,
            delimiter=','
        )
        ax1_1 = fig1.add_subplot(3, 3, i_cable + 1)
        ax2_1 = fig2.add_subplot(3, 3, i_cable + 1)
        ax3_1 = fig3.add_subplot(3, 3, i_cable + 1)
        freqs = cable_log[:, 0] * units.Hz
        amp = np.power(10., .05 * cable_log[:, 1])
        phase = cable_phase[:, 1] * units.deg
        amp_interpolation = scipy.interpolate.interp1d(
            freqs,
            scipy.signal.savgol_filter(amp, n_convolve, filter_degree),
            bounds_error=False,
            fill_value=(amp[0], amp[-1])
        )
        phase_interpolation = scipy.interpolate.interp1d(
            freqs,
            scipy.ndimage.gaussian_filter1d(phase, 1),
            bounds_error=False,
            fill_value=(phase[1], phase[-2])
        )

        ax1_1.plot(
            freqs / units.MHz,
            amp,
            color='C{}'.format(i_cable)
        )
        ax1_1.plot(
            plot_freqs / units.MHz,
            amp_interpolation(plot_freqs),
            color='k',
            alpha=.5
        )
        ax2_1.plot(
            freqs / units.MHz,
            scipy.signal.detrend(scipy.ndimage.gaussian_filter1d(phase, 5)),
            color='C{}'.format(i_cable)
        )
        ax2_1.plot(
            plot_freqs / units.MHz,
            scipy.signal.detrend(phase_interpolation(plot_freqs)),
            color='C{}'.format(i_cable),
            linestyle='--',
            alpha=.5
        )
        ax3_1.plot(
            plot_times,
            fft.freq2time(
                amp_interpolation(plot_freqs) * np.exp(1.j * phase_interpolation(plot_freqs)),
                sampling_rate
            ),
            color='C{}'.format(i_cable)
        )
        phase_slope = np.polyfit(
            freqs,
            phase,
            1
        )
        cable_delay = - phase_slope[0] / 2. / np.pi
        cable_delay_dict[str(station_id)][str(cable_id)] = cable_delay
        ax3_1.axvline(
            cable_delay,
            color='k',
            linewidth=2
        )
        ax3_1.set_xlim([cable_delay - 10, cable_delay + 10])
        ax1_1.grid()
        ax2_1.grid()
        ax2_1.set_ylim([-.05, .1])
        ax3_1.grid()
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig1.savefig('plots/surface_cables/amplitude_station{}.png'.format(station_id))
    fig2.savefig('plots/surface_cables/phase_station{}.png'.format(station_id))
    fig3.savefig('plots/surface_cables/impulse_station{}.png'.format(station_id))
json.dump(cable_delay_dict, open('surface_cable_delays.json', 'w'), indent=2)