Obtain the trigger efficiency vs SNR curve given flower traces

1. Run python3 extract_snr.py to get the second highest SNR across 4 channels for each calibration pulse at each attenuation
- define path to flower traces (main_path)
- flower_st11_runAtt.csv maps each run taken for station 11 with corresponding attenuation
2. Run python3 extract_trigger_efficiency.py to obtain the trigger efficiency vs SNR and trigger efficiency vs attenuation curves
- define path to flower traces (main_path)
- define path to SNRs for each attenuation (indir)
- adjust parameters for exponential snr vs attenuation fit in scipy.optimize.curve_fit
