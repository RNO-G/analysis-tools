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
    event_tree = file["events"]
    EventSummary = event_tree["EventSummary"]
    event_number_arr = stack_if_object(EventSummary["event_number"])
    rms_arr = stack_if_object(EventSummary["rms"])
    max_abs_amplitude_arr = stack_if_object(EventSummary["max_abs_amplitude"])
    glitching_ts_arr = stack_if_object(EventSummary["glitching_test_statitic"]) # There is typo in the monitoring.root for glitch ts
    block_offsets_arr = stack_if_object(EventSummary["block_offsets"])

    return {
        "event_number_arr": event_number_arr,
        "rms_arr": rms_arr.T,
        "max_abs_amplitude_arr": max_abs_amplitude_arr.T, # transpose to get shape (n_ch, n_events)
        "glitching_test_statistic_arr": glitching_ts_arr.T,
        "block_offsets_arr": block_offsets_arr.T
    }

def get_run_summary_from_monitoring_file(file):
    run_summary = file["RunSummary"]
    run_summary_members = run_summary.members
    return run_summary_members # is already a dict with member names as keys and arrays as values

def get_info_from_header_file(header_file):
    header = header_file["header"]
    trigger_time = stack_if_object(header["trigger_time"])
    trigger_time_utc = trigger_time.astype("datetime64[s]")
    run_no = stack_if_object(header["run_number"])
    event_number = stack_if_object(header["event_number"])
    station_id = stack_if_object(header["station_number"])
    
    trigger_info = header["trigger_info"]
    force_trigger = stack_if_object(trigger_info["trigger_info.force_trigger"])
    radiant_trigger = stack_if_object(trigger_info["trigger_info.radiant_trigger"])
    lt_trigger = stack_if_object(trigger_info["trigger_info.lt_trigger"])
    which_radiant = stack_if_object(trigger_info["trigger_info.which_radiant_trigger"])

    # If there are overlaps raise error
    overlap_mask = (force_trigger.astype(bool) & radiant_trigger.astype(bool)) | (force_trigger.astype(bool) & lt_trigger.astype(bool)) | (radiant_trigger.astype(bool) & lt_trigger.astype(bool))
    if np.any(overlap_mask):
        n_overlap = np.sum(overlap_mask)
        logger.error(f"Found {n_overlap} events with overlapping trigger types. Please check the trigger info in the header file.")
        raise ValueError(f"Found {n_overlap} events with overlapping trigger types. Please check the trigger info in the header file.")

    return {
        "trigger_time_utc": trigger_time_utc,
        "run_no": run_no,
        "event_number": event_number,
        "station_id": station_id,
        "force_trigger": force_trigger,
        "radiant_trigger": radiant_trigger,
        "lt_trigger": lt_trigger,
        "which_radiant": which_radiant
    }

def assign_trigger_types(force_trigger, radiant_triger, lt_trigger, which_trigger, default="UNKNOWN"):
    if len(force_trigger) != len(radiant_triger) or len(force_trigger) != len(lt_trigger) or len(force_trigger) != len(which_trigger):
        logger.error("Trigger arrays must have the same length.")
        raise ValueError("Trigger arrays must have the same length.")

    n_events = len(force_trigger)
    trigger_type_arr = np.full(n_events, default, dtype=object)

    trigger_type_arr[force_trigger] = "FORCE"
    trigger_type_arr[lt_trigger] = "LT"

    # Choose RADIANT0 or RADIANT1 based on which_radiant
    trigger_type_arr[radiant_triger & (which_trigger == 0)] = "RADIANT0"
    trigger_type_arr[radiant_triger & (which_trigger == 1)] = "RADIANT1"
    trigger_type_arr[radiant_triger & ~np.isin(which_trigger, [0, 1])] = "RADIANTX" # In case there are some events with which_radiant not 0 or 1, we assign them as RADIANTX

    ## Sanity checks
    overlap_count = (force_trigger.astype(int) + radiant_triger.astype(int) + lt_trigger.astype(int)).sum()
    overlap_mask = overlap_count > 1
    if np.any(overlap_mask):
        overlap_idx = np.where(overlap_mask)[0]
        logger.error(f"Found {len(overlap_idx)} events with overlapping trigger types at indices {overlap_idx}. Please check the trigger info.")
        raise ValueError(f"Found {len(overlap_idx)} events with overlapping trigger types at indices {overlap_idx}. Please check the trigger info.")
    
    wrong_force = np.where(force_trigger & (trigger_type_arr != "FORCE"))[0]
    if len(wrong_force) > 0:
        logger.error(f"Found {len(wrong_force)} events where force_trigger is True but trigger type is not assigned as FORCE at indices {wrong_force}. Please check the trigger info.")
        raise ValueError(f"Found {len(wrong_force)} events where force_trigger is True but trigger type is not assigned as FORCE at indices {wrong_force}. Please check the trigger info.")
    
    wrong_lt = np.where(lt_trigger & (trigger_type_arr != "LT"))[0]
    if len(wrong_lt) > 0:
        logger.error(f"Found {len(wrong_lt)} events where lt_trigger is True but trigger type is not assigned as LT at indices {wrong_lt}. Please check the trigger info.")
        raise ValueError(f"Found {len(wrong_lt)} events where lt_trigger is True but trigger type is not assigned as LT at indices {wrong_lt}. Please check the trigger info.")
    
    wrong_radiant = np.where(radiant_triger & ~np.isin(trigger_type_arr, ["RADIANT0", "RADIANT1", "RADIANTX"]))[0]
    if len(wrong_radiant) > 0:
        logger.error(f"Found {len(wrong_radiant)} events where radiant_trigger is True but trigger type is not assigned as RADIANT at indices {wrong_radiant}. Please check the trigger info.")
        raise ValueError(f"Found {len(wrong_radiant)} events where radiant_trigger is True but trigger type is not assigned as RADIANT at indices {wrong_radiant}. Please check the trigger info.")
    
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
            msg = (
                f"Number of {trigger_type} triggers in header ({expected}) does not match the count from trigger type array ({found}). Please check the trigger info.")
            logger.error(msg)
            raise ValueError(msg)

def calculate_snr(max_abs_amplitude_arr, rms_arr):
    snr_arr = []
    for ch in range(max_abs_amplitude_arr.shape[0]):
        if rms_arr[ch] == 0:
            logger.warning(f"RMS value is zero for channel {ch}. Assigning SNR as np.inf for this channel.")
            snr_ch = np.inf
        else:
            snr_ch = max_abs_amplitude_arr[ch] / rms_arr[ch]
            snr_arr.append(snr_ch)
    return np.array(snr_arr)

def choose_trigger_type_header(trigger_type_arr, trigger_type:str):
    '''Choose events based on trigger type.'''
    if trigger_type not in ["FORCE", "LT", "RADIANT0", "RADIANT1"]:
        logger.error(f"Invalid trigger type {trigger_type}. Must be one of FORCE, LT, RADIANT0 or RADIANT1.")
        raise ValueError(f"Invalid trigger type {trigger_type}. Must be one of FORCE, LT, RADIANT0 or RADIANT1.")
    
    mask = trigger_type_arr == trigger_type
    return mask
