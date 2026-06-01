import uproot 
import numpy as np
import os
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def stack_if_object(branch_data):
    arr = np.array(branch_data)
    if arr.dtype == object:
        arr = np.stack(arr)
    return arr

def open_file(path):
    if not os.path.isfile(path):
        logger.warning(f"File {path} does not exist. Skipping...") # For multiple runs, we can have some missing monitoring files, so we just log a warning and skip those runs instead of raising an error.
        return None
    return uproot.open(path)

def get_event_info_from_monitoring_file(file):
    try:
        event_tree = file["events"]
        EventSummary = event_tree["EventSummary"]
        event_number_arr = stack_if_object(EventSummary["event_number"])
        rms_arr = stack_if_object(EventSummary["rms"])
        max_abs_amplitude_arr = stack_if_object(EventSummary["max_abs_amplitude"])
        glitching_ts_arr = stack_if_object(EventSummary["glitching_test_statitic"]) # There is typo in the monitoring.root for glitch ts
        block_offsets_arr = stack_if_object(EventSummary["block_offset"])

        return {
            "event_number_arr": event_number_arr, # (n_events,)
            "rms_arr": rms_arr.T, # (n_ch, n_events)
            "max_abs_amplitude_arr": max_abs_amplitude_arr.T, # (n_ch, n_events)
            "glitching_test_statistic_arr": glitching_ts_arr.T, # (n_ch, n_events)
            "block_offsets_arr": block_offsets_arr.T # (n_ch, n_events)
        }
    
    except KeyError as e:
        logger.error(f"Key {e} not found in events tree of monitoring file. Please check the structure of the monitoring file.")
        return None

def get_run_summary_from_monitoring_file(file):
    try:
        run_summary = file["RunSummary"]
        run_summary_members = run_summary.members
        return run_summary_members # is already a dict with member names as keys and arrays as values

    except KeyError as e:
        logger.error(f"Key {e} not found in run summary of monitoring file. Please check the structure of the monitoring file.")
        return None

def get_info_from_header_file(header_file):
    try:
        header = header_file["header"]
        trigger_time = stack_if_object(header["trigger_time"])
        trigger_time_utc = trigger_time.astype("datetime64[s]")
        readout_time = stack_if_object(header["readout_time"])
        duration = np.max(readout_time) - np.min(readout_time) # in seconds
        run_no = stack_if_object(header["run_number"])
        event_number = stack_if_object(header["event_number"])
        station_id = stack_if_object(header["station_number"])
        
        trigger_info = header["trigger_info"]
        force_trigger = stack_if_object(trigger_info["trigger_info.force_trigger"])
        radiant_trigger = stack_if_object(trigger_info["trigger_info.radiant_trigger"])
        lt_trigger = stack_if_object(trigger_info["trigger_info.lt_trigger"])
        which_radiant = stack_if_object(trigger_info["trigger_info.which_radiant_trigger"])

        # If there are overlaps return None
        overlap_mask = (force_trigger.astype(bool) & radiant_trigger.astype(bool)) | (force_trigger.astype(bool) & lt_trigger.astype(bool)) | (radiant_trigger.astype(bool) & lt_trigger.astype(bool))
        if np.any(overlap_mask):
            n_overlap = np.sum(overlap_mask)
            overlap_event_indices = np.where(overlap_mask)[0]
            force_radiant_overlap = np.where(force_trigger.astype(bool) & radiant_trigger.astype(bool))[0]
            force_lt_overlap = np.where(force_trigger.astype(bool) & lt_trigger.astype(bool))[0]
            radiant_lt_overlap = np.where(radiant_trigger.astype(bool) & lt_trigger.astype(bool))[0]
            logger.error(f"Found {n_overlap} events with overlapping trigger types at indices: {overlap_event_indices} for run {run_no[0]}. FORCE-RADIANT: {force_radiant_overlap}, FORCE-LT: {force_lt_overlap}, RADIANT-LT: {radiant_lt_overlap}. Please check the trigger info in the header file.")
            return None
        
        return {
            "trigger_time_utc": trigger_time_utc,
            "run_no": run_no,
            "event_number": event_number,
            "station_id": station_id,
            "force_trigger": force_trigger,
            "radiant_trigger": radiant_trigger,
            "lt_trigger": lt_trigger,
            "which_radiant": which_radiant,
            "readout_time": readout_time,
            "duration": duration
        }
    
    except KeyError as e:
        logger.error(f"Key {e} not found in header file. Please check the structure of the header file.")
        return None

def assign_trigger_types(force_trigger, radiant_trigger, lt_trigger, which_trigger, default="UNKNOWN"):
    if len(force_trigger) != len(radiant_trigger) or len(force_trigger) != len(lt_trigger) or len(force_trigger) != len(which_trigger):
        logger.error("Trigger arrays must have the same length.")
        return None

    n_events = len(force_trigger)
    trigger_type_arr = np.full(n_events, default, dtype='<U10') 

    trigger_type_arr[force_trigger] = "FORCE"
    trigger_type_arr[lt_trigger] = "LT"

    # Choose RADIANT0 or RADIANT1 based on which_radiant
    trigger_type_arr[radiant_trigger & (which_trigger == 0)] = "RADIANT0"
    trigger_type_arr[radiant_trigger & (which_trigger == 1)] = "RADIANT1"
    trigger_type_arr[radiant_trigger & ~np.isin(which_trigger, [0, 1])] = "RADIANTX" # In case there are some events with which_radiant not 0 or 1, we assign them as RADIANTX

    ## Sanity checks
    overlap_count = (force_trigger.astype(int) + radiant_trigger.astype(int) + lt_trigger.astype(int))
    overlap_mask = overlap_count > 1
    if np.any(overlap_mask):
        overlap_idx = np.where(overlap_mask)[0]
        logger.error(f"Found {len(overlap_idx)} events with overlapping trigger types at indices {overlap_idx}. Please check the trigger info.")
        return None
    
    wrong_force = np.where(force_trigger & (trigger_type_arr != "FORCE"))[0]
    if len(wrong_force) > 0:
        logger.error(f"Found {len(wrong_force)} events where force_trigger is True but trigger type is not assigned as FORCE at indices {wrong_force}. Please check the trigger info.")
        return None
    
    wrong_lt = np.where(lt_trigger & (trigger_type_arr != "LT"))[0]
    if len(wrong_lt) > 0:
        logger.error(f"Found {len(wrong_lt)} events where lt_trigger is True but trigger type is not assigned as LT at indices {wrong_lt}. Please check the trigger info.")
        return None
    
    wrong_radiant = np.where(radiant_trigger & ~np.isin(trigger_type_arr, ["RADIANT0", "RADIANT1", "RADIANTX"]))[0]
    if len(wrong_radiant) > 0:
        logger.error(f"Found {len(wrong_radiant)} events where radiant_trigger is True but trigger type is not assigned as RADIANT at indices {wrong_radiant}. Please check the trigger info.")
        return None
    
    unknown_idx = np.where(trigger_type_arr == default)[0]
    if len(unknown_idx) > 0:
        logger.warning(f"Found {len(unknown_idx)} events with trigger type not in FORCE, LT, RADIANT0, RADIANT1 or RADIANTX at indices {unknown_idx}. They are assigned as {default}.")
    
    return trigger_type_arr

def check_event_numbers_according_to_trigger_types(trigger_type_arr, n_forced_triggers,n_lt_triggers,n_rf0_triggers,n_rf1_triggers,):
    
    unique, counts = np.unique(trigger_type_arr, return_counts=True)
    trigger_type_counts = dict(zip(unique, counts))

    expected_counts = {
        "FORCE": n_forced_triggers,
        "LT": n_lt_triggers,
        "RADIANT0": n_rf0_triggers,
        "RADIANT1": n_rf1_triggers,
    }

    for trigger_type, expected in expected_counts.items():
        found = trigger_type_counts.get(trigger_type, 0)

        if found != expected:
            logger.error(f"Mismatch in trigger type counts for {trigger_type}: expected {expected}, found {found}. Please check the trigger info and event numbers.")
            return False

    return True

def calculate_snr(max_abs_amplitude_arr, rms_arr):
    snr_arr = np.full_like(max_abs_amplitude_arr, np.inf, dtype=float)
    nonzero_mask = rms_arr != 0
    snr_arr[nonzero_mask] = max_abs_amplitude_arr[nonzero_mask] / rms_arr[nonzero_mask]

    if np.any(~nonzero_mask):
        logger.warning(f"Found {np.sum(~nonzero_mask)} zero RMS values. Assigned SNR as np.inf.")

    return snr_arr

def choose_trigger_type_header(trigger_type_arr, trigger_type:str):
    '''Choose events based on trigger type.'''
    if trigger_type not in ["FORCE", "LT", "RADIANT0", "RADIANT1"]:
        logger.error(f"Invalid trigger type {trigger_type}. Must be one of FORCE, LT, RADIANT0 or RADIANT1.")
        return None
    
    mask = trigger_type_arr == trigger_type
    return mask

def read_multiple_runs(base_path, station_id, run_numbers):
    all_event_info = []

    total_n_events = 0
    total_n_force_triggers = 0
    total_n_lt_triggers = 0
    total_n_rf0_triggers = 0
    total_n_rf1_triggers = 0

    run_event_counts = {}
    run_trigger_rates = {}

    spectrum_keys = ["avg_spectrum", "avg_spectrum_force", "avg_spectrum_lt", "avg_spectrum_rf0", "avg_spectrum_rf1"] # (n_ch, n_freqs)
    channel_event_keys = ["rms_arr", "max_abs_amplitude_arr", "glitching_test_statistic_arr", "block_offsets_arr", "snr_arr"] # (n_ch, n_events)
    event_keys = ["event_number_arr", "triggerType", "trigger_time_utc", "run_no", "station_id"] # 1D arrays with shape (n_events,)

    freqs = None

    failed_runs = []
    failed_run_info = {}

    for run_no in tqdm(run_numbers, desc=f"Reading monitoring and header files for runs between {run_numbers[0]} and {run_numbers[-1]} for station {station_id}"):
        monitoring_file_path = os.path.join(base_path, f"station{station_id}/run{run_no}", "monitoring.root")
        header_file_path = os.path.join(base_path, f"station{station_id}/run{run_no}", "headers.root")

        monitoring_file = open_file(monitoring_file_path)
        header_file = open_file(header_file_path)

        # Skip run if either monitoring or the header file is missing and report
        missing_files = []
        if monitoring_file is None:
            missing_files.append(monitoring_file_path)
        if header_file is None:
            missing_files.append(header_file_path)
        if missing_files:
            msg = f"Missing files for run {run_no} for station {station_id}: {missing_files}. Skipping this run."
            logger.error(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue

        event_info_dict = get_event_info_from_monitoring_file(monitoring_file)
        run_summary_dict = get_run_summary_from_monitoring_file(monitoring_file)
        header_info_dict = get_info_from_header_file(header_file)

        if event_info_dict is None:
            msg = f"Failed to read event info for run {run_no} for station {station_id}. Skipping this run."
            logger.error(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue    

        if run_summary_dict is None:
            msg = f"Failed to read run summary info for run {run_no} for station {station_id}. Skipping this run."
            logger.warning(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue
        
        if header_info_dict is None:
            msg = f"Failed to read header info for run {run_no} for station {station_id}. Skipping this run."
            logger.warning(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue

        # Some sanity checks to make sure event number and run number are consistent between monitoring and header files
        if not np.array_equal(event_info_dict["event_number_arr"], header_info_dict["event_number"]):
            msg = f"Event numbers in monitoring file and header file do not match for run {run_no} for station {station_id}. Skipping the run. Please check the files."
            logger.error(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue
        
        header_station_id = np.unique(header_info_dict["station_id"])
        header_run_no = np.unique(header_info_dict["run_no"])

        run_summary_station_id = run_summary_dict["station_number"]
        run_summary_run_no = run_summary_dict["run_number"]
        
        if header_station_id.size != 1:
            msg = f"Multiple station IDs found in header file for run {run_no} for station {station_id}. Found station IDs: {header_station_id}. Skipping the run. Please check the file."
            logger.error(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue

        if header_run_no.size != 1:
            msg = f"Multiple run numbers found in header file for run {run_no} for station {station_id}. Found run numbers: {header_run_no}. Skipping the run. Please check the file."
            logger.error(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue

        if header_station_id[0] != station_id:
            msg = f"Station ID in header file ({header_station_id[0]}) does not match the station ID in the path ({station_id}) for run {run_no}. Skipping the run. Please check the file."
            logger.error(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue
        
        if header_station_id[0] != run_summary_station_id:
            msg = f"Station ID in header file ({header_station_id[0]}) does not match the station ID in monitoring file run summary ({run_summary_station_id}) for run {run_no}. Skipping the run. Please check the file."
            logger.error(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue
        
        if header_run_no[0] != run_no:
            msg = f"Run number in header file ({header_run_no[0]}) does not match the run number in the path ({run_no}) for run {run_no}. Skipping the run. Please check the file."
            logger.error(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue
        
        if header_run_no[0] != run_summary_run_no:
            msg = f"Run number in header file ({header_run_no[0]}) does not match the run number in monitoring file run summary ({run_summary_run_no}) for run {run_no}. Skipping the run. Please check the file."
            logger.error(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue
        
        # Start processing the run if all checks are passed
        trigger_type_arr = assign_trigger_types(
            header_info_dict["force_trigger"],
            header_info_dict["radiant_trigger"],
            header_info_dict["lt_trigger"],
            header_info_dict["which_radiant"]
        )

        if trigger_type_arr is None:
            msg = f"Failed to assign trigger types for run {run_no} for station {station_id}. Skipping this run."
            logger.error(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue

        check_event_numbers = check_event_numbers_according_to_trigger_types(
            trigger_type_arr,
            run_summary_dict["n_forced_triggers"],
            run_summary_dict["n_lt_triggers"],
            run_summary_dict["n_rf0_triggers"],
            run_summary_dict["n_rf1_triggers"],
        )

        if not check_event_numbers:
            msg = f"Event number check according to trigger types failed for run {run_no} for station {station_id}. Skipping this run."
            logger.error(msg)
            failed_runs.append(run_no)
            failed_run_info[run_no] = msg
            continue

        # Add trigger type and some header info to event info dict for easier access later for each event
        event_info_dict["triggerType"] = trigger_type_arr # (n_events,)
        event_info_dict["trigger_time_utc"] = header_info_dict["trigger_time_utc"] # (n_events,)
        event_info_dict["run_no"] = header_info_dict["run_no"] # (n_events,)
        event_info_dict["station_id"] = header_info_dict["station_id"] # (n_events,)
        event_info_dict["duration"] = header_info_dict["duration"] # a number representing the duration of the run in seconds, calculated as the difference between the max and min readout time in the header file
        event_info_dict["readout_time"] = header_info_dict["readout_time"] # (n_events,) 

        # Add runsummary 
        event_info_dict["avg_spectrum"] = stack_if_object(run_summary_dict["avg_spectrum"]) # (n_ch, n_freqs)
        event_info_dict["avg_spectrum_force"] = stack_if_object(run_summary_dict["avg_spectrum_force"]) # (n_ch, n_freqs)
        event_info_dict["avg_spectrum_lt"] = stack_if_object(run_summary_dict["avg_spectrum_lt"]) # (n_ch, n_freqs)
        event_info_dict["avg_spectrum_rf0"] = stack_if_object(run_summary_dict["avg_spectrum_rf0"]) #RADIANT0
        event_info_dict["avg_spectrum_rf1"] = stack_if_object(run_summary_dict["avg_spectrum_rf1"]) #RADIANT1

        # Calculate SNR and add to event info dict
        snr_arr = calculate_snr(event_info_dict["max_abs_amplitude_arr"], event_info_dict["rms_arr"])
        event_info_dict["snr_arr"] = snr_arr # (n_ch, n_events)

        # Count total events:
        total_n_events += run_summary_dict["n_events"]
        total_n_force_triggers += run_summary_dict["n_forced_triggers"]
        total_n_lt_triggers += run_summary_dict["n_lt_triggers"]
        total_n_rf0_triggers += run_summary_dict["n_rf0_triggers"]  
        total_n_rf1_triggers += run_summary_dict["n_rf1_triggers"]


        # Calculate trigger rates and add to run_trigger_rates dict for this run
        trigger_rate_force = run_summary_dict["n_forced_triggers"] / event_info_dict["duration"]
        trigger_rate_lt = run_summary_dict["n_lt_triggers"] / event_info_dict["duration"]
        trigger_rate_rf0 = run_summary_dict["n_rf0_triggers"] / event_info_dict["duration"]
        trigger_rate_rf1 = run_summary_dict["n_rf1_triggers"] / event_info_dict["duration"]

        run_trigger_rates[run_no] = {
            "force_trigger_rate": trigger_rate_force,
            "lt_trigger_rate": trigger_rate_lt,
            "rf0_trigger_rate": trigger_rate_rf0,
            "rf1_trigger_rate": trigger_rate_rf1,
            "run_start_time_utc": np.min(event_info_dict["readout_time"].astype("datetime64[s]")), 
        }

        # Add event counts for this run to the run_event_counts dict
        run_event_counts[run_no] = {
            "n_events": run_summary_dict["n_events"],
            "n_forced_triggers": run_summary_dict["n_forced_triggers"],
            "n_lt_triggers": run_summary_dict["n_lt_triggers"],
            "n_rf0_triggers": run_summary_dict["n_rf0_triggers"],
            "n_rf1_triggers": run_summary_dict["n_rf1_triggers"],
        }

        
        if freqs is None:
            freqs = stack_if_object(run_summary_dict["frequencies"]) # (n_freqs,)

        all_event_info.append(event_info_dict)
        
    if len(all_event_info) == 0:
        raise ValueError(f"No valid runs were processed for station {station_id}. Please check the files and the run numbers. Or try reading using the dataProviderRNOG method which reads from combined.root files in /inbox/ and can be used for older data before 2026 which do not have monitoring.root files.")
    
    combined_event_info = {}
    for key in channel_event_keys:
        combined_event_info[key] = np.concatenate([event_info[key] for event_info in all_event_info], axis=1) # concatenate along events axis, so final shape is (n_ch, n_events_total)
    for key in event_keys:
        combined_event_info[key] = np.concatenate([event_info[key] for event_info in all_event_info], axis=0) # concatenate along events axis, so final shape is (n_events_total,)
    for key in spectrum_keys:
        combined_event_info[key] = np.stack([event_info[key] for event_info in all_event_info], axis=1) # stack along new axis for runs, so final shape is (n_ch, n_runs, n_freqs)
    
    combined_event_info["freqs"] = freqs # (n_freqs,)
    combined_event_info["total_n_events"] = total_n_events
    combined_event_info["total_n_force_triggers"] = total_n_force_triggers
    combined_event_info["total_n_lt_triggers"] = total_n_lt_triggers
    combined_event_info["total_n_rf0_triggers"] = total_n_rf0_triggers
    combined_event_info["total_n_rf1_triggers"] = total_n_rf1_triggers
    combined_event_info["run_event_counts"] = run_event_counts # dict with run number as key and value as another dict with n_events, n_forced_triggers, n_lt_triggers, n_rf0_triggers, n_rf1_triggers for that run
    combined_event_info["run_trigger_rates"] = run_trigger_rates # dict with run number as key and value as another dict with force_trigger_rate, lt_trigger_rate, rf0_trigger_rate, rf1_trigger_rate for that run
    combined_event_info["failed_runs"] = failed_runs if len(failed_runs) > 0 else None
    combined_event_info["failed_run_info"] = failed_run_info if len(failed_run_info) > 0 else None

    logger.info(f"Successfully read and combined data from {len(all_event_info)} runs for station {station_id} using the monitoring data. Total events: {total_n_events}, total FORCE triggers: {total_n_force_triggers}, total LT triggers: {total_n_lt_triggers}, total RADIANT0 triggers: {total_n_rf0_triggers}, total RADIANT1 triggers: {total_n_rf1_triggers}.")

    return combined_event_info
