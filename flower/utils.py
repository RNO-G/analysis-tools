import json
import gzip
import os
import libconf
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d


def read_flower_data(path):
    """Read flower data from a file/run.

    Parameters
    ----------
    path : str
        Path to the file or run directory.

    Returns
    -------
    data : dict
        Flower data.
    """

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
        first_row = f.readlines()[0].split(" ")
        assert first_row[0] == "STATION", "Parsing of runinfo.txt went wrong"
        station = int(first_row[2])

    flower_gain_file = f"{directory}/flower_gain_codes.0.txt"
    with open(flower_gain_file, "r") as f:
        flower_gains = f.readlines()[1].split(" ")

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
