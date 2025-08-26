import json
import gzip
import os
import libconf
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d



def read_flower_data(path, read_data_in_volts=False):
    """Read flower data from a file/run.

    Parameters
    ----------
    path : str
        Path to the file or run directory.
    
    read_data_in_volts : bool
        If True, will convert adc to volts by applying a linear conversion and
        factoring out the digital flower gain

    Returns
    -------
    data : dict
        Flower data.
    """

    # hardcoded parameters intrinsic to flower
    adc_input_range = 2.
    nr_bits = 8
    trigger_board_amplifications = np.array([1, 1.25, 2, 2.5, 4, 5, 8, 10, 12.5, 16, 20, 25, 32, 50])


    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if os.path.isdir(path):
        if not os.path.exists(f"{path}/aux/flower_end.json.gz"):
            raise FileNotFoundError(f"File not found: {path}/aux/flower_end.json.gz")
        path = f"{path}/aux/flower_end.json.gz"

    directory = os.path.dirname(path)

    cfg_file = f"{directory}/../cfg/acq.cfg"
    with open(cfg_file, "r") as f:
        conf = libconf.load(f)

    turn_off_at_exit = conf["calib"].get("turn_off_at_exit", 1)
    if conf["calib"]["enable_cal"] and not turn_off_at_exit:
        atten = conf["calib"]["atten"]
        channel = conf["calib"]["channel"]
    else:
        atten = None
        channel = None

    runinfo_file = f"{directory}/runinfo.txt"
    with open(runinfo_file, "r") as f:
        lines = f.readlines()
        first_row = lines[0].split(" ")
        assert first_row[0] == "STATION", "Parsing of runinfo.txt went wrong"
        station = int(first_row[2])

        third_row = lines[2].split("=")
        assert third_row[0] == "RUN-START-TIME ", "Parsing of runinfo.txt went wrong, expected RUN-START-TIME on third line"
        run_start_time = float(third_row[1])

        last_row = lines[-1].split("=")
        assert last_row[0] == "RUN-END-TIME ", "Parsing of runinfo.txt went wrong, expected RUN-END-TIME on last line"
        run_end_time = float(last_row[1])


    flower_gain_file = f"{directory}/flower_gain_codes.0.txt"
    with open(flower_gain_file, "r") as f:
        flower_gains = [int(g) for g in f.readlines()[1].split(" ")]

    if path.endswith(".json"):
        with open(path, "r") as f:
            json_data = json.load(f)
    elif path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            json_data = json.load(f)
    else:
        raise ValueError("Unknown file format")

    json_data["station"] = station
    json_data["cal_channel"] = channel
    json_data["atten"] = atten
    json_data["flower_gains"] = flower_gains
    json_data["run_start_time"] = run_start_time
    json_data["run_end_time"] = run_end_time

    if read_data_in_volts:
        volts_per_adc = adc_input_range / (2**nr_bits - 1)
        gain_amplification = trigger_board_amplifications[flower_gains]

        events = json_data["events"]
        nr_channels = len(flower_gains)
        for i,event in enumerate(events):
            for channel_id in range(nr_channels):
                events[i]["ch" + str(channel_id)] = [wf*volts_per_adc/gain_amplification[channel_id]
                                                     for wf in events[i]["ch"+str(channel_id)]]


    return json_data


def get_waveforms_and_metadata(json_data, only_coinc=False, only_calpulser=False):

    if only_coinc:
        mask = np.array([ele["metadata"]["trigger_type"] for ele in json_data["events"]]) == "COINC"
    elif only_calpulser:
        mask = np.array([ele["metadata"]["pps_flag"] for ele in json_data["events"]])
    else:
        mask = np.ones(len(json_data["events"]), dtype=bool)

    events = np.array(json_data["events"])[mask]

    waveforms = np.array([
        [events[idx][f"ch{ch}"] for idx in range(len(events))]
        for ch in range(4)]
    )

    metadata = {}
    for key in events[0]["metadata"]:
        metadata[key] = np.array([ele["metadata"][key] for ele in events])

    return waveforms, metadata


def maximum_peak_to_peak_amplitude(trace, coincidence_window_size):
    return maximum_filter1d(trace, coincidence_window_size) - minimum_filter1d(trace, coincidence_window_size)


def calculate_snr(trace, coincidence_window_size, signal_window=None, noise_window=None):
    """
    Calculate the signal to noise ratio of a trace.

    SNR = Vpp_max / (2 * std(trace))
    Vpp_max is the maximum peak-to-peak amplitude within a sliding window.

    Parameters
    ----------
    trace : array
        Waveform to calculate the signal-to-noise ratio for.

    coincidence_window_size : int
        Number of bins of the sliding window within to calcuate the maximim peak-to-peak amplitude.

    signal_window : array of bools (Default: None)
        Select a "search" window for the maximum peak-to-peak amplitude

    noise_window : array of bools (Default: None)
        Defines the part of the trace from which to calculate the standard deviation

    Returns
    -------
    pos : int
        Position (sample) of the Vpp_max
    snr : float
        Maximum signal to noise ratio of the trace
    """

    if noise_window is None:
        noise_window = np.zeros_like(trace, dtype=bool)
        noise_window[250:] = True


    if signal_window is None:
        max_ampl_pp_window = maximum_peak_to_peak_amplitude(trace, coincidence_window_size)
    else:
        max_ampl_pp_window = maximum_peak_to_peak_amplitude(trace[signal_window], coincidence_window_size)

    if noise_window is None:
        rms = np.std(trace)
    else:
        rms = np.std(trace[noise_window])

    return np.argmax(max_ampl_pp_window), np.amax(max_ampl_pp_window) / (2 * rms)



class flowerDataset():
    """
    Helper class for handling flower data, does require NuRadio to be installed
    """
    def __init__(self, filepath, read_data_in_volts):
        data_dict = read_flower_data(filepath, read_data_in_volts=read_data_in_volts)
        self.in_volts = read_data_in_volts

        self.station_id = data_dict["station"]
        self.run = data_dict["run"]
        self.cal_channel = data_dict["cal_channel"]
        self.nr_channels = len(data_dict["flower_gains"])
        self.run_start_time = data_dict["run_start_time"]
        self.run_end_time = data_dict["run_end_time"]

        self.wfs = []
        for event in data_dict["events"]:
            wf_ch = []
            for channel_id in range(self.nr_channels):
                wf_ch.append(event["ch"+str(channel_id)])
            self.wfs.append(wf_ch)
        self.wfs = np.array(self.wfs)

        self.nr_events = len(data_dict["events"])
        self.nr_samples = len(self.wfs[0][0])

        #HARDCODED
        self.sampling_rate = 0.472

    def save_average_spectrum(self, filename, filt=None, debug=False):
        """
        saves an frequency spectrum averaged over the run in this dataset
        """
        import pickle
        from NuRadioReco.utilities.fft import freqs, time2freq, freq2time
        frequencies = freqs(self.nr_samples, self.sampling_rate)
        
        spectra = time2freq(self.wfs, self.sampling_rate)
        if filt is not None:
            spectra = spectra * np.abs(filt)

        # remove DC component
        spectra[:,:,0] = 0

        if debug:
            channel_id = 1
            plt.plot(frequencies, np.abs(spectra)[0][channel_id], label = "event 0")
            plt.plot(frequencies, np.mean(np.abs(spectra), axis=0)[channel_id], label="run mean")
            plt.legend()
            plt.xlabel("freq / GHz")
            plt.ylabel("spectral amplitude / V/GHz")
            plt.title(f"run {self.run}, channel {channel_id}")
            plt.savefig("figures/tests/test_flower_spectrum.png")

        average_ft = np.mean(np.abs(spectra), axis=0)
        var_average_ft = np.var(np.abs(spectra), axis=0)

        header_dic = {"nr_events" : self.nr_events,
                      "begin_time" : self.run_start_time,
                      "end_time" : self.run_end_time}
        save_dictionary = {"header" : header_dic,
                           "freq" : frequencies,
                           "frequency_spectrum" : average_ft,
                           "var_frequency_spectrum" : var_average_ft}

        with open(filename, "wb") as file:
            pickle.dump(save_dictionary, file)
    

    def plot_wf(self, wf_idx, channel_id, ax):
        wf = self.wfs[wf_idx][channel_id]
        ax.plot(wf)
        ax.set_xlabel("sample")
        ax.set_ylabel("V")
        ax.set_title(f"run {self.run}")
        plt.show()
