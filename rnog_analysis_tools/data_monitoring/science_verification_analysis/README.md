# Science Verification Analysis (SVA)

## Purpose

The Science Verification Analysis checks whether an RNO-G station is behaving as expected, using a set of automated diagnostic tests run over a chosen set of runs. Each test looks at a different failure mode (galactic-noise sensitivity, SNR/RMS stability, channel glitching, ADC block offsets, trigger rates) and reports a per-channel health verdict (`OK`, `!!`, or `X`). The output is a concise CSV summary for every channel plus a set of detailed plots and text/JSON reports for anyone who needs to dig into a flagged channel.

The analysis is meant to be run regularly (e.g. after a station comes back online, or periodically to catch slow degradation) and to be usable by anyone on the collaboration, not just the analysis' original author — channel maps, thresholds, and plotting styles all live in config files rather than in code, and reference ("expected") values are calculated once per station from a known-good time period and then reused.

There are two ways to read the underlying data:

- **`monitoring.root`** (current, recommended): reads pre-computed per-event summary quantities (RMS, max amplitude, glitching test statistic, block offsets, spectra) from the station's monitoring stream. Fast, and works for all data taken with the monitoring pipeline.
- **`dataProviderRNOG`** (outdated/legacy): reads full waveforms via NuRadioReco for older runs that predate `monitoring.root`. Kept only for archival purposes — see the [[Legacy path](https://claude.ai/cowork/local_dbaff4ef-131f-445a-bc91-dabf4583340d#sva_dataproviderrnog-legacy)](#sva_dataproviderrnog-legacy) section below for its caveats.

## Quick start / usage example

Run the main analysis for station 14 over an explicit list of runs:

```bash
python science_verification_analysis_main.py -st 14 --runs 260080 260090 260100
```

Or over a run range, excluding a couple of bad runs:

```bash
python science_verification_analysis_main.py -st 14 --run_range 260080 260150 --exclude-runs 260101 260102
```

Or over a date range (run numbers are looked up from the RNO-G run table):

```bash
python science_verification_analysis_main.py -st 14 --time_range 2026-04-26 2026-05-02
```

`--runs`, `--run_range`, and `--time_range` are mutually exclusive — pick exactly one way to select runs. `-st`/`--station_id` and exactly one of these are required; `-ex`/`--exclude-runs` and `--debug_plot` (produce extra diagnostic plots) are optional.

Use `--data_location` to choose where the raw data is read from and where results are written (default `desy`):

- `desy` — reads from the DESY inbox (`/pnfs/ifh.de/acs/radio/diskonly/data/inbox/`), writes results under `/pnfs/ifh.de/acs/radio/diskonly/NuRadioMC/science_verification_analysis/`.
- `uchicago` — reads from the UChicago mirror (`/data/satellite`), writes results under `/data/sva`.
- any other value is treated as a custom base data path, with results written to a `results/` subdirectory under it.

Each run of the script creates its own uniquely-named output directory (so concurrent/repeated runs never collide or overwrite each other): `<base_results_path>/<YY-MM-DD>_station-<ID>_run<first>-run<last>_<random6chars>/`, containing:

- `channel_health_summary/validation_summary_station<ID>_<run_label>.csv` — the top-level per-channel verdict table.
- `detailed_results/` — per-test text/JSON files with the numbers behind each verdict (spectral results, SNR/RMS outlier details, glitching stats, block offset stats, failed-run report).
- `plots/standard_plots/`, `plots/failed_test_plots/`, `plots/other_debug_plots/` — plots are split into these three subdirectories: always-produced standard plots, and debug plots routed to `failed_test_plots` or `other_debug_plots` depending on whether the corresponding test passed for all channels.
- `logs/logging_science_verification_analysis_station<ID>_<run_label>.log` — a full log of the run, including warnings about missing files, invalid timestamps, and borderline channels.

> **Note:** the results base paths (`/pnfs/...`) are dCache-mounted storage. dCache enforces write-once semantics per file — a file can only be opened for writing once; a second `open(..., "w")` or `open(..., "a")` on the same path fails with `PermissionError: [Errno 1] Operation not permitted` even though the file exists and is nominally writable. Because of this, every output-writing function in `helper_functions/output_writer.py` builds its full content in memory first and performs exactly one `open()`/write/close per file — don't reintroduce per-item append/reopen loops when editing these.

## Analysis overview

Each station is configured (in `config_station.json`) with a `daq_type` of either `radiant` or `didaq`, which determines the non-FORCE trigger names used throughout (`LT`/`RADIANT0`/`RADIANT1` for RADIANT stations vs. `DIDAQ_DEEP_PHASED`/`DIDAQ_SURF_UP`/`DIDAQ_SURF_DOWN` for DIDAQ stations) and which summary-CSV builder is used (`create_result_csv_file()` vs. `create_result_csv_file_didaq()`). Glitching and block-offset analysis (steps 6–7 below) only run for RADIANT stations — DIDAQ stations skip them entirely.

For a given station and set of runs, `science_verification_analysis_main.py` runs the following steps in order:

1. **Read data** — `read_multiple_runs()` reads `monitoring.root` and `headers.root` for every requested run, validates them against each other (matching event numbers, station/run IDs, non-overlapping trigger types), and concatenates everything into one combined dataset. Runs with missing or inconsistent files are skipped and reported rather than crashing the whole analysis.
2. **Spectral (galactic noise) test** — normalizes surface-channel spectra to a reference band and checks, in several frequency bands, whether upward-facing channels show more galactic excess than downward-facing ones (as expected).
3. **SNR stability (z-score) test** — compares each channel's log-SNR distribution for the current runs against a previously-computed reference (mean/std/k-value per channel) and flags statistically significant outlier events.
4. **RMS/Vrms modality & tail test** — uses a KDE of the RMS distribution per channel to check it's unimodal (not bimodal/flat, which would suggest a mis-biased or noisy channel), and separately checks for excess skew/tails.
5. **RMS stability test** — same z-score approach as SNR, plus a relative-median-shift metric across runs, combined into an overall stability decision per channel.
6. **Glitching test** (RADIANT only) — a one-sided binomial test on how often each channel's glitching test statistic is triggered, against an expected background rate.
7. **Block offset test** (RADIANT only) — summarizes each channel's ADC block-offset statistics (mean/median/std/IQR/P99) and compares them to reference values; informational only (not counted in the overall channel-health verdict).
8. **Trigger rate plots** — plots trigger rates over time per trigger type, no pass/fail verdict.
9. **Summary CSV** — `create_result_csv_file()` (RADIANT) or `create_result_csv_file_didaq()` (DIDAQ) combines the SNR, spectral, RMS, RMS-stability, and (for RADIANT) glitching verdicts into one `OK`/`!!`/`X` per channel and writes the summary table.

Steps 2–7 each have their own config file (thresholds, band definitions, KDE parameters, etc. — see below) so they can be tuned without touching analysis code, and their own reference/"expected value" script under `expected_values/` to (re-)generate reference numbers for a new station or after a known-good re-calibration period.

## Directory structure

### `science_verification_analysis_main.py`
The entry point described above. Also home to `setup_logging()`. `REFERENCE_DIR` and `CONFIG_DIR` are script-relative constants so the script can be run from anywhere, but the per-run output directories (`plots/`, `detailed_results/`, `channel_health_summary/`, `logs/`) are computed inside `if __name__ == "__main__":` under a uniquely-named directory whose base path depends on `--data_location` (see Quick start above).

### `config_files_sva/`
All tunable parameters, one JSON file per topic (plus one small Python helper). Editing these does not require touching analysis code:

- `config_station.json` — channel maps (`all_channels`, `surface_channels`, `deep_channels`, `upward_channels`, `downward_channels`, `vpol_channels`, `hpol_channels`, `phased_array_channels`, `reference_channels`, `reference_channels_galaxy`) under a `default_config`, with a `station_specific_adjustments` block keyed by station ID (currently station 14 has a different channel layout).
- `config_helper.py` — `get_station_config(station_id, default_config, station_specific_adjustments)` merges the default config with any station-specific overrides.
- `config_spectral_analysis.json` — frequency bands used for the galactic-noise test (`galactic_excess`, plus a few RFI bands), the normalization band, and the significance thresholds (`alpha_spec`, `ci_threshold_spec`, `log_ratio_thresholds_spec`).
- `config_rms.json` — parameters for the KDE modality test (`bandwidth`, `grid_points`, `peak_prominence`, `height_threshold`), the tail/skewness test (percentiles, `extreme_k`, minimum event count), and the human-readable reporting thresholds (`strong_skew`, `extreme_skew`, etc.).
- `config_glitching.json` — the binomial test parameters for the glitching test (`alpha`, expected background `pvalue`, confidence level, and the CI thresholds that separate "weak"/"moderate"/"strong" excessive glitching).
- `config_block_offsets.json` — acceptable median/IQR block-offset limits.
- `config_plotting.py` — `set_plot_style()` centralizes all matplotlib `rcParams` (fonts, tick sizes, line widths, dpi) plus a shared 24-color palette (`COLORS`), so every plot in the analysis looks consistent.

### `analysis_functions_sva/`
The statistics/physics behind each test, split by topic and independent of I/O:

- `spectral_analysis_sva.py` — `normalize_channels()`, `find_amplitude_ratio_in_band[_specific_bkg]()`, `excess_info_from_ratio[_specific_bkg]()`, `validate_excess_in_bands()`. Computes upward/downward amplitude ratios per frequency band and classifies them as NO/WEAK/MODERATE/STRONG excess (method differs slightly for monitoring vs. dataProviderRNOG data).
- `z_score_analysis_sva.py` — shared statistics machinery used by both the SNR and RMS tests: log-parameter statistics, z-scores against a reference mean/std, k-value derivation (`find_k_value`), outlier flagging/detail extraction, saving/loading reference values to/from JSON (including the metadata block), and a rolling-window z-score variant.
- `vrms_analysis_sva.py` — `calculate_vrms()` (from raw waveforms, dataProviderRNOG path) and `get_rms_per_trigger_monitoring()` (from precomputed RMS, monitoring path) both split values by trigger type; `kde_modality()` and `tail_fraction_and_trimmed_skew_two_sided()` implement the modality/tail tests; `report_vrms_characteristics()` turns those into human-readable labels.
- `vrms_stability_analysis_sva.py` — `get_rms_per_run()`, `relative_median_shift()` (pairwise relative shift in median RMS between runs), and `decision_metric()`, which combines outlier fraction, largest z-score excess, and median-shift into the final `OK`/`!!`/`X` RMS-stability verdict. Also contains a block of commented-out, currently-unused helpers (Wasserstein-distance and linear-regression-based stability metrics) kept for possible future use.
- `glitching_analysis_sva.py` — `binomtest_glitch_fraction()`, a one-sided binomial test per channel classifying glitching as NO/WEAK/MODERATE/STRONG excessive.
- `block_offsets_analysis_sva_monitoring.py` — `get_force_block_offsets_monitoring()`, `block_offset_statistics_monitoring()` (mean/median/std/IQR/P99/P95), and a violin-plot helper, for the monitoring.root path.
- `block_offsets_analysis_sva_dataproviderrnog.py` — equivalent block-offset functions for the legacy dataProviderRNOG path (computes offsets before *and* after removal, since that path removes block offsets from the waveform).

### `monitoring_data_functions_sva/`
- `get_monitoring_data_uproot.py` — everything related to reading `monitoring.root`/`headers.root` with `uproot`: low-level readers (`get_event_info_from_monitoring_file`, `get_run_summary_from_monitoring_file`, `get_info_from_header_file`), trigger-type assignment and consistency checks (`assign_trigger_types`, `check_event_numbers_according_to_trigger_types`), SNR calculation, and the main `read_multiple_runs()` entry point that loops over runs, validates and concatenates everything, and reports failed runs instead of raising.

### `helper_functions/`
Small, reusable utilities that don't belong to a specific test:

- `config_helper.py` is imported from `config_files_sva/` (see above) but conceptually lives here too.
- `output_writer.py` — every "write results to disk" function used by the main script: failed-run CSV, spectral results text file, SNR/RMS outlier-detail text files, RMS modality text files, glitching results text file, block-offset results text file, and `create_result_csv_file()`/`create_result_csv_file_didaq()`, which assemble the final per-channel summary CSV and compute the combined `channel_health()` verdict. Because the results filesystem (dCache/pnfs) only allows a file to be opened for writing once, `write_spectral_results()` takes the results for *all* surface channels at once and writes them in a single `open()` call, rather than being called once per channel.
- `read_rnog_runtable.py` — `read_rnog_runtable()`, a thin wrapper around `rnog_data.runtable` used to turn a `--time_range` into a list of run numbers.

### `plotting_functions_sva/`
One module per plot family, all consuming already-computed arrays/dicts (no analysis logic lives here):

- `plotting_sva_spectrum.py` — time-integrated surface/deep spectrum plots (normalized and unnormalized).
- `plotting_sva_snr.py` — SNR-vs-time plots per channel, including the outlier flags and z-score/k-value bands; also `choose_day_interval()`, a small helper that picks a sensible tick spacing based on the time range.
- `plotting_sva_vrms.py` — RMS/Vrms-vs-time plots (all triggers together and per trigger), the z-score single-trigger plot, rolling-mean plots, and the relative-median-shift heatmap (`create_heatmap_plot`).
- `plotting_sva_glitch.py` — glitching violin plots and the 99th-percentile-glitching-over-time plot.
- `plotting_sva_debug.py` — extra diagnostic plots (amplitude-ratio distributions, raw/z-scored SNR distributions, RMS distributions with the KDE overlay) enabled with `--debug_plot`.
- `plotting_sva_trigger_rate.py` — trigger-rate-over-time and trigger-rate-heatmap plots.

### `expected_values/`
Scripts to (re-)generate the reference ("expected") values that the SNR and RMS stability tests compare against, plus the values themselves:

- `expected_snr_values.py` / `expected_rms_values.py` — standalone CLIs (same run-selection arguments as the main script, plus `--save-values`) that read a known-stable period for a station and compute per-channel k-value/mean/std, writing them to `expected_snr/expected_snr_values_station<ID>.json` / `expected_rms/expected_rms_station<ID>.json`. Each JSON file carries a `metadata` block (station ID, run range, excluded runs, event count, trigger type, start/end time, and any comment — including an automatic note when a k-value hit the min/max cap) alongside the `values` block, so a reference file is self-documenting about how and when it was produced.
- `expected_snr/`, `expected_rms/` — one combined JSON file per station (not split by parameter as in earlier versions).
- `expected_block_offsets/` — one JSON per station with the reference block-offset statistics (median/IQR/P95/P99, in both ADC counts and mV) and the simulation settings used to derive them.
- `outdated/` — reference-value scripts and results for the legacy dataProviderRNOG method (`expected_snr_values_dataproviderrnog.py`, `expected_rms_values_dataproviderrnog.py`). Per its own README: "These scripts won't be updated anymore."

### `sva_dataproviderrnog/` (legacy)
- `read_rnog_data_nuradio.py` — reads full waveforms via NuRadioReco's `readRNOGDataMattak` from `combined.root` files.
- `science_verification_analysis_dataprovider.py` — the original, pre-monitoring.root version of the full analysis, kept only for stations/periods without `monitoring.root` files. Its own module docstring spells out the caveats: no reference-value files exist for it by default (you'd need to run the `expected_values/outdated` scripts first and update the file paths), and its RMS analysis is done in ADC units, which is explicitly flagged as incorrect. Prefer `science_verification_analysis_main.py` whenever `monitoring.root` is available.

### `outdated/` (top-level, legacy)
Pre-JSON-config versions of the station/trigger-rate configuration (`config_station.py`, `trigger_rate.py`), superseded by `config_files_sva/config_station.json`. Kept for reference only; not imported by any current analysis code.

### Output directories (created at runtime, not checked in)
For `science_verification_analysis_main.py`, output directories are created dynamically per run under a `--data_location`-dependent base path — see [Quick start](#quick-start--usage-example) above, not next to the script. For the `expected_values/` scripts, `plots_reference/`, `logs_reference/`, `results_reference/` are created next to the script itself. The `channel_health_summary/`, `detailed_results/`, `logs/`, and `plots/` directories present directly under this folder are historical outputs from before the dynamic per-run output scheme was introduced; they aren't written to by the current version of `science_verification_analysis_main.py`. None of these need to exist beforehand — every script creates its output directories with `os.makedirs(..., exist_ok=True)`.