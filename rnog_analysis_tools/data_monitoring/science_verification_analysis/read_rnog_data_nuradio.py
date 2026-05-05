import os
import numpy as np
from tqdm import tqdm
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import NuRadioReco.framework.trigger
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
from NuRadioReco.framework.parameters import channelParametersRNOG as chp_rnog
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def convert_events_information(event_info, convert_to_arrays=True):

    data = defaultdict(list)

    for ele in event_info.values():
        for k, v in ele.items():
            data[k].append(v)

    if convert_to_arrays:
        for k in data:
            data[k] = np.array(data[k])

    return data

def read_rnog_data(station_id: int, run_numbers: list, backend: str = "pyroot"):
    '''Read RNO-G data for a given station and list of run numbers using the specified backend.'''
    file_list = []
    valid_run_numbers = []
    missing_runs = []

    for run_id in run_numbers:
        path = f"/pnfs/ifh.de/acs/radio/diskonly/data/inbox/station{station_id}/run{run_id}/combined.root"
        if os.path.isfile(path):
            file_list.append(path)
            valid_run_numbers.append(run_id)
        else:
            missing_runs.append(run_id)

    if missing_runs:
        logger.warning(f"!!!! Skipping {len(missing_runs)} missing runs: {missing_runs} !!!!")
    if not file_list:
        raise FileNotFoundError("No combined.root files found for selected runs.")
    
    n_files = len(file_list)
    n_batches = n_files // 100 + 1
    logger.info(f"Reading {n_files} files in {n_batches} batches using {backend} backend.")

    event_info = defaultdict(list)

    n_events_total = 0
    spec_batches = []
    trace_batches = []
    times_trace_batches = []
    snr_batches = []
    glitch_batches = []
    run_no_all = []
    times_all = []
    freqs = None

    from NuRadioReco.modules.channelSignalReconstructor import channelSignalReconstructor as csr
    csr = csr()
    csr.begin(debug=False)

    from NuRadioReco.modules.RNO_G.channelGlitchDetector import channelGlitchDetector as cgd_rnog
    cgd_rnog = cgd_rnog()
    cgd_rnog.begin()

    for batch in tqdm(np.array_split(np.array(file_list), n_batches), desc="Reading batches", unit="batch"):
        tableReader = dataProviderRNOG()
        tableReader.begin(files=batch.tolist(), 
                          det=None,
                          reader_kwargs={"overwrite_sampling_rate":2.4, 
                                         "convert_to_voltage":True,
                                         "apply_baseline_correction":"auto",
                                         "mattak_kwargs":{"backend":backend}})
        event_info_tmp = tableReader.reader.get_events_information(
            keys=["triggerType", "triggerTime", "readoutTime", "radiantThrs", "lowTrigThrs", "run", "eventNumber"])
        
        event_info_tmp = convert_events_information(event_info_tmp, False)
        for key, value in event_info_tmp.items():
            event_info[key] += value

        n_events = tableReader.reader.get_n_events()
        n_events_total += n_events
        logger.info(f"Reading {n_events} events in this batch.")

        channel_list = [i for i in range(24)]  
        spec_arr = np.zeros((len(channel_list), n_events, 1025))
        trace_arr = np.zeros((len(channel_list), n_events, 2048))
        times_trace_arr = np.zeros((len(channel_list), n_events, 2048))
        snr_arr = np.zeros((len(channel_list), n_events))
        glitch_arr = np.zeros((len(channel_list), n_events))

        run_no = []
        times = []
        event_ids = []
        
        for idx, event in enumerate(tqdm(tableReader.reader.run(), total=n_events, desc="Events", unit="evt", leave=False)):
            station = event.get_station()
            time = station.get_station_time().datetime64
            times.append(time)
            run_no.append(event.get_run_number())

            csr.run(evt=event, station=station, det=None, stored_noise=False)
            cgd_rnog.run(event=event, station=station, det=None)
            for i_ch, ch in enumerate(channel_list):
                channel = station.get_channel(ch)

                times_ch = channel.get_times()
                times_trace_arr[i_ch, idx, :] = times_ch

                snr_dict = channel.get_parameter(chp.SNR)
                snr_peak = snr_dict["peak_amplitude"]
                snr_arr[i_ch, idx] = snr_peak

                glitching_values = channel.get_parameter(chp_rnog.glitch_test_statistic)
                glitch_arr[i_ch, idx] = glitching_values
                
                spec = channel.get_frequency_spectrum()
                spec_arr[i_ch, idx, :] = np.abs(spec)

                trace = channel.get_trace()
                trace_arr[i_ch, idx, :] = trace

                if freqs is None and idx == 0 and i_ch == 0:
                    freqs = channel.get_frequencies()
        
        spec_batches.append(spec_arr)
        trace_batches.append(trace_arr)
        times_trace_batches.append(times_trace_arr)
        snr_batches.append(snr_arr)
        glitch_batches.append(glitch_arr)
        run_no_all.extend(run_no)
        times_all.extend(times)

    spec_arr = np.concatenate(spec_batches, axis=1)
    trace_arr = np.concatenate(trace_batches, axis=1)
    times_trace_arr = np.concatenate(times_trace_batches, axis=1)
    snr_arr = np.concatenate(snr_batches, axis=1)
    glitch_arr = np.concatenate(glitch_batches, axis=1)

    run_no = np.array(run_no_all)
    times = np.array(times_all)     

    for key, value in event_info.items():
        event_info[key] = np.array(value)

    inf_mask = np.isinf(event_info["triggerTime"])
    event_info["triggerTime"][inf_mask] = event_info["readoutTime"][inf_mask]
    if np.any(inf_mask):
        logger.warning(f"Found {np.sum(inf_mask)} events with inf trigger time (of {len(inf_mask)} events)")

    logger.info(f"n_events read: {spec_arr.shape[1]}, n_events_total: {n_events_total}")
    logger.debug(f"freqs shape: {freqs.shape}, spec_arr shape: {spec_arr.shape}, trace_arr shape: {trace_arr.shape}, times_trace_arr shape: {times_trace_arr.shape}, snr_arr shape: {snr_arr.shape}, times shape: {times.shape}, run_no shape: {run_no.shape}  ")
    logger.debug(f"trigger types: {np.unique(event_info['triggerType'])}")

    return spec_arr, trace_arr, times_trace_arr, snr_arr, run_no, times, freqs, event_info, glitch_arr
