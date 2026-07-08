import os
import csv
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def write_failed_runs_to_csv(station_id, failed_run_info, run_label, results_dir):
    failed_runs_file = os.path.join(results_dir, f"station{station_id}_failed_runs_in_runrange_{run_label}.csv")
    with open(failed_runs_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run Number", "Reason for Failure"])
        for run_no, reason in failed_run_info.items():
            writer.writerow([run_no, reason])
    logger.warning(f"Failed to process some runs for station {station_id}: {list(failed_run_info.keys())}. Information about these runs has been written to {failed_runs_file} in the {results_dir} directory. Please check the file for details as this might indicate potential issues!")

def write_spectral_results(ch, excess_info_results, station_id, run_label, results_dir, log_once = False, reset_file = False):
    spectral_results_file = os.path.join(results_dir, f"spectral_analysis_results_{station_id}_{run_label}.txt")
    if reset_file:
        open(spectral_results_file, "w").close()  
    
    with open(spectral_results_file, "a") as f:
        f.write(f"Channel {ch:02d}:\n")
        for band, results in excess_info_results.items():
            f.write(f"\n=== {band} ===\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        f.write("\n")
    if log_once:
        logger.info(f"Spectral analysis results written to {spectral_results_file}")

def write_snr_outlier_details(outlier_details, station_id, run_label, n_events_force, results_dir):
    outlier_results_file = os.path.join(results_dir, f"force_snr_details_{station_id}_{run_label}.txt")
    with open(outlier_results_file, "w") as f:
        for ch in sorted(outlier_details.keys()):
            entries = outlier_details[ch]
            n_outliers = len(entries)

            if n_outliers == 0:
                f.write(f"\nChannel {ch:02d}:\n 0 outliers\n")
                continue

            k_ch = entries[0]["k"]
            outlier_fraction = n_outliers / n_events_force if n_events_force > 0 else 0.0
            max_delta = max(abs(e.get("z_minus_k", 0.0)) for e in entries)
            f.write(f"\nChannel {ch:02d}:\n {n_outliers} outliers (k = {k_ch:.2f}), outlier fraction: {outlier_fraction:.2f}, max delta: {max_delta:.2f}\n")
            
            for e in entries:
                f.write(f"  - run {e['run']}, event {e['eventNumber']}, |z| = {e['z_abs']:.2f} (delta = {e['z_minus_k']:.2f} above k)\n")
            
    logger.info(f"SNR outlier details written to {outlier_results_file}")

def write_vrms_outlier_details(outlier_details, station_id, run_label, trigger_label, n_events, results_dir, use_monitoring = False):
    if use_monitoring:
        file_label = "rms"
    else:
        file_label = "vrms"
    
    outlier_results_file = os.path.join(results_dir, f"{file_label}_details_{station_id}_{run_label}_{trigger_label}.txt")
    with open(outlier_results_file, "w") as f:
        for ch in sorted(outlier_details.keys()):
            entries = outlier_details[ch]
            n_outliers = len(entries)

            if n_outliers == 0:
                f.write(f"\nChannel {ch:02d}:\n 0 outliers\n")
                continue

            k_ch = entries[0]["k"]
            outlier_fraction = n_outliers / n_events if n_events > 0 else 0.0
            max_delta = max(abs(e.get("z_minus_k", 0.0)) for e in entries)
            f.write(f"\nChannel {ch:02d}:\n {n_outliers} outliers (k = {k_ch:.2f}), outlier fraction: {outlier_fraction:.2f}, max delta: {max_delta:.2f}\n")
            
            for e in entries:
                f.write(f"  - run {e['run']}, event {e['eventNumber']}, |z| = {e['z_abs']:.2f} (delta = {e['z_minus_k']:.2f} above k)\n")
            
    logger.info(f"{file_label.capitalize()} outlier details written to {outlier_results_file}")

def write_vrms_modality_results(modality, tail_label, trigger_label, station_id, run_label, results_dir, use_monitoring = False):
    if use_monitoring:
        file_label = "rms"
    else:
        file_label = "vrms"
    
    modality_results_file = os.path.join(results_dir, f"{file_label}_modality_{station_id}_{run_label}_{trigger_label}.txt")
    with open(modality_results_file, "w") as f:
        for ch in sorted(modality.keys()):
            modality_result = modality[ch]
            tail_label_result = tail_label[ch]
            f.write(f"Channel {ch} ({trigger_label} events): {modality_result} ({tail_label_result})\n")
    logger.info(f"{file_label.capitalize()} modality results for {trigger_label} events written to {modality_results_file}")

def write_glitching_results(glitch_info, station_id, run_label, all_channels, results_dir):
    glitch_results_file = os.path.join(results_dir, f"glitching_analysis_results_{station_id}_{run_label}.txt")
    lines = [
        (
            f"Channel {ch:2d} | "
            f"n_glitches: {glitch_info[ch]['n_glitches']:4d} | "
            f"n_events: {glitch_info[ch]['n_events']:<4d} | "
            f"(frac={glitch_info[ch]['glitch_fraction']:.3f}) | "
            f"p={glitch_info[ch]['pval']:.2e} | "
            f"CI99%={glitch_info[ch]['confidence_interval']} | "
            f"{glitch_info[ch]['validation']}"
        )
        for ch in all_channels]
    with open(glitch_results_file, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Glitching analysis results written to {glitch_results_file}")

def write_block_offset_results(block_offset_stats, station_id, run_label, ref_block_off_dict, results_dir, use_monitoring=False):
    block_offset_results_file = os.path.join(results_dir, f"block_offset_analysis_results_{station_id}_{run_label}.txt")
    results_dict = {}
    with open(block_offset_results_file, "w") as f:
        for ch in sorted(block_offset_stats.keys()):
            stats = block_offset_stats[ch]
            results = ref_block_off_dict.get(str(ch), {})
            print(f"stats for channel {ch}: {stats}")
            if results == {}:
                f.write(f"Channel {ch:02d}:\n")
                f.write("  No reference block offset data available for this channel.\n")
                results_dict[ch] = "?"
                continue
            
            if use_monitoring:
                median_ref = results.get("median_adc_offset_counts", None)
                p99_ref = results.get("p99_adc_offset_counts", None)
                f.write(f"Channel {ch:02d}:\n")
                f.write(f"  Mean block offset: {stats['mean']}, median: {stats['median']}, std: {stats['std']}, IQR: {stats['iqr']}, P99: {stats['p99']}\n")
                if median_ref is not None and stats["median"] > median_ref:
                    results_dict[ch] = "X"
                    logger.warning(f"Channel {ch:02d} has a high median block offset of {stats['median']}, which may indicate a potential issue with the channel.")
                elif p99_ref is not None and stats["p99"] > p99_ref:
                    results_dict[ch] = "X"
                    logger.warning(f"Channel {ch:02d} has a high P99 of block offsets ({stats['p99']}), indicating significant variability that may need further investigation.")
                else:
                    results_dict[ch] = "OK"
            
            else:
                median_ref = results.get("median_adc_offset_mv", None)
                p99_ref = results.get("p99_adc_offset_mv", None)
                f.write(f"Channel {ch:02d}:\n")
                f.write(f"  Before removal - mean: {stats['before_mean']} V, median: {stats['before_median']} V, std: {stats['before_std']} V, IQR: {stats['iqr_before']} V, P99: {stats['p99_before']} V\n")
                f.write(f"  After removal - mean: {stats['after_mean']} V, median: {stats['after_median']} V, std: {stats['after_std']} V, IQR: {stats['iqr_after']} V, P99: {stats['p99_after']} V\n")
                f.write(f"  Removal fraction (based on median): {stats['removal_fraction']*100:.1f}%\n")
                f.write(f"  P99 reduction fraction: {stats['p99_reduction_fraction']*100:.1f}%\n")

                if median_ref is not None and stats["before_median"] > median_ref:
                    results_dict[ch] = "X"
                    logger.warning(f"Channel {ch:02d} has a high median block offset of {stats['before_median']} V before removal, which may indicate a potential issue with the channel.")
                elif p99_ref is not None and stats["p99_before"] > p99_ref:
                    results_dict[ch] = "X"
                    logger.warning(f"Channel {ch:02d} has a high P99 of block offsets ({stats['p99_before']} V) before removal, indicating significant variability that may need further investigation.")
                elif median_ref is not None and stats["after_median"] > median_ref:
                    results_dict[ch] = "X"
                    logger.warning(f"Channel {ch:02d} has a relatively high median block offset of {stats['after_median']} V after removal, removal was not fully effective.")
                elif p99_ref is not None and stats["p99_after"] > p99_ref:
                    results_dict[ch] = "X"
                    logger.warning(f"Channel {ch:02d} has a relatively high P99 of block offsets ({stats['p99_after']} V) after removal, indicating that there may still be significant variability in block offsets.")
                else:
                    results_dict[ch] = "OK"
    
    logger.info(f"Block offset analysis results written to {block_offset_results_file}")
    return results_dict

def channel_health(row):
    severity = {"OK": 0, "!!": 1, "X": 2}
    vals = [severity[v] for v in row if v in severity]
    if not vals:
        return "-"
    inv = {0: "OK", 1: "!!", 2: "X"}
    return inv[max(vals)]

def create_result_csv_file(station_id, run_label, n_events_force, surface_channels, downward_channels, upward_channels, all_channels, validation_results, glitch_info, block_offsets_result_dict, rms_results, modality_dict_force, modality_dict_lt, 
                           modality_dict_radiant0, modality_dict_radiant1, outlier_details, csv_dir, rms_label):
    out_csv_file = os.path.join(csv_dir, f"validation_summary_station{station_id}_{run_label}.csv")
    ch_list = list(all_channels)

    spectral_col = []
    glitch_col = []
    block_offset_col = []
    rms_stability_col = []
    modality_force_col = []
    modality_lt_col = []
    modality_radiant0_col = []
    modality_radiant1_col = []
    snr_col = []

    for ch in ch_list:
        df_spec_val = ""
        if ch in surface_channels:
            spectral_validation = None
            vr = validation_results.get(ch, {})
            spectral_validation = vr.get("galactic_excess", {})

            if spectral_validation is None:
                df_spec_val = "?"
            else:
                if ch in downward_channels:
                    if spectral_validation == "NO EXCESS":
                        df_spec_val = "OK"
                    elif spectral_validation == "WEAK EXCESS":
                        df_spec_val = "!!"
                    elif spectral_validation in ["MODERATE EXCESS", "STRONG EXCESS"]:
                        df_spec_val = "X"
                    else:
                        df_spec_val = "?"
                elif ch in upward_channels:
                    if spectral_validation in ["STRONG EXCESS", "MODERATE EXCESS"]:
                        df_spec_val = "OK"
                    elif spectral_validation == "WEAK EXCESS":
                        df_spec_val = "!!"
                    elif spectral_validation == "NO EXCESS":
                        df_spec_val = "X"
                    else:
                        df_spec_val = "?"
                else:
                    df_spec_val = "?"
        else:
            df_spec_val = "-"

        spectral_col.append(df_spec_val)

        # Glitching column
        if glitch_info is None:
            glitch_val = "-"
            glitch_col.append(glitch_val)
        else:
            info = glitch_info.get(ch, None)
            glitch_val_raw = info.get("validation") if info is not None else "-"
            if glitch_val_raw == "NO EXCESSIVE GLITCHING":
                glitch_val = "OK"
            elif glitch_val_raw == "WEAK EXCESSIVE GLITCHING":
                glitch_val = "!!"
            elif glitch_val_raw in ["MODERATE EXCESSIVE GLITCHING", "STRONG EXCESSIVE GLITCHING"]:
                glitch_val = "X"
            else:
                glitch_val = "-"
            glitch_col.append(glitch_val)

        # Vrms analysis column
        if rms_results[ch] is None:
            rms_val = "-"
            rms_stability_col.append(rms_val)
        else:
            rms_value = rms_results[ch].get("decision", "-")
            rms_stability_col.append(rms_value)

        if modality_dict_force is None:
            modality_value = "-"
            modality_force_col.append(modality_value)
        else:
            n_peaks = modality_dict_force[ch]["n_peaks"]
            if n_peaks == 0:
                modality_value = "!!"
            elif n_peaks == 1:
                modality_value = "OK"
            elif n_peaks == 2:
                modality_value = "X"
            else:
                modality_value = f"X"
            modality_force_col.append(modality_value)

        if modality_dict_lt is None:
            modality_value = "-"
            modality_lt_col.append(modality_value)
        else:
            n_peaks = modality_dict_lt[ch]["n_peaks"]
            if n_peaks == 0:
                modality_value = "!!"
            elif n_peaks == 1:
                modality_value = "OK"
            elif n_peaks == 2:
                modality_value = "X"
            else:
                modality_value = f"X"
            modality_lt_col.append(modality_value)

        if modality_dict_radiant0 is None:
            modality_value = "-"
            modality_radiant0_col.append(modality_value)
        else:
            n_peaks = modality_dict_radiant0[ch]["n_peaks"]
            if n_peaks == 0:
                modality_value = "!!"
            elif n_peaks == 1:
                modality_value = "OK"
            elif n_peaks == 2:
                modality_value = "X"
            else:
                modality_value = "X"
            modality_radiant0_col.append(modality_value)
        if modality_dict_radiant1 is None:
            modality_value = "-"
            modality_radiant1_col.append(modality_value)
        else:
            n_peaks = modality_dict_radiant1[ch]["n_peaks"]
            if n_peaks == 0:
                modality_value = "!!"
            elif n_peaks == 1:
                modality_value = "OK"
            elif n_peaks == 2:
                modality_value = "X"
            else:
                modality_value = "X"
            modality_radiant1_col.append(modality_value)

        # SNR validation column
        outlier_ch_info = outlier_details.get(ch, [])
        n_out = len(outlier_ch_info)
        if n_out == 0:
            snr_value = "OK"
        else:
            max_delta = max(abs(o.get("z_minus_k", 0.0)) for o in outlier_ch_info)
            frac_out = n_out / n_events_force if n_events_force > 0 else 0.0
            if max_delta < 3.0:
                snr_value = "OK"

            elif max_delta < 5.0:
                snr_value = "OK" if frac_out < 0.002 else "!!"

            else:  # max_delta >= 5
                if n_out == 1 and frac_out < 0.002:
                    snr_value = "OK"
                elif frac_out < 0.004:
                    snr_value = "!!"
                else:
                    snr_value = "X"
        snr_col.append(snr_value)

        # Block offsets column
        block_offset_result = block_offsets_result_dict.get(ch, None)
        if block_offset_result is None:
            block_offset_val = "-"
            block_offset_col.append(block_offset_val)
        else:
            block_offset_val = block_offset_result
            block_offset_col.append(block_offset_val)
        
    df = pd.DataFrame({
        "Channel": ch_list,
        "SNR": snr_col,
        "Galaxy (FORCE)": spectral_col,
        f"{rms_label.capitalize()} Stability (FORCE)": rms_stability_col,
        f"{rms_label.capitalize()} (FORCE)": modality_force_col,
        f"{rms_label.capitalize()} (LT)": modality_lt_col,
        f"{rms_label.capitalize()} (RADIANT0)": modality_radiant0_col,
        f"{rms_label.capitalize()} (RADIANT1)": modality_radiant1_col,
        "Glitching": glitch_col,
        "Block Offsets": block_offset_col
    })

    health_cols =["SNR", "Galaxy (FORCE)", f"{rms_label.capitalize()} Stability (FORCE)", f"{rms_label.capitalize()} (FORCE)", "Glitching"]
    df["Channel Health (FORCE)"] = df[health_cols].apply(channel_health, axis=1)
    df.to_csv(out_csv_file, index=False)
    logger.info(f"Validation summary saved to {out_csv_file}")



