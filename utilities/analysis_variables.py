from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy import signal
from scipy.signal import hilbert
import numpy as np

## Helper funcs
def coherent_sum(traces, ref_channel=0):
    """Obtain the coherently summed waveform from the phased array channels. This does vary slightly depending on the reference channel, and a more robust version will likely be created soon.
    
    This is also currently using the hilbert evenlope, but may not be necessary if traces are upsampled.
    
    Args:
        traces (array): array containing channel waveforms (traces[0] should return ch 0's trace, traces[1] ch 1's trace, and so on for at least the phased array channels)
        
    Returns:
        array representing phased array coherently summed waveform
    
    """

    sum_chan = traces[ref_channel]
    channels = [0, 1, 2, 3]
    channels.pop(ref_channel)
    
    for ch in channels: ## exclude reference channel since already accounted for above
        cor = signal.correlate(np.abs(hilbert(sum_chan)), np.abs(hilbert(traces[ch])), mode="full")
        lag = int(np.argmax((cor)) - (np.size(cor)/2.))
        
        aligned_wf = np.roll(traces[ch], lag)
        sum_chan = sum_chan + aligned_wf
        
    return sum_chan

def maximum_peak_to_peak_amplitude(trace, coincidence_window_size=6):
    """Obtains peak to peak amplitude values for each element of a given voltage waveform for a given window size
    
    Args:
        trace (array): array containing voltage waveform
        coincidence_window_size (int): 
        
    Returns:
        array equal size of input trace representing peak to peak amplitude at each index for the coincidence window size specified
    
    """
    return maximum_filter1d(
        trace, coincidence_window_size
    ) - minimum_filter1d(trace, coincidence_window_size)
    
def noise_rms(trace):
    """Obtain the noise RMS for a given voltage waveform. To work for waveforms with impulsive signals in them, the array is split into four chunks, the RMS calculated for each chunk, and the average of the two lowest RMS chunks is returned.
    
    Args:
        trace (array): array containing voltage waveform
        
    Returns:
        float representing noise RMS of waveform
    
    """
    
    split_array = np.array_split(trace, 4)
    RMS_of_splits = np.std(split_array, axis=1)
    ordered_RMSs = np.sort(RMS_of_splits)
    lowest_two = ordered_RMSs[:2]
    RMS_noise = np.mean(lowest_two)
    
    return RMS_noise

## Analysis variables    
def avg_ch_SNR(traces):
    """Obtain the average of the SNRs for the phased array channels (0, 1, 2, 3)
    
    Args:
        traces (array): array containing channel waveforms (traces[0] should return ch 0's trace, traces[1] ch 1's trace, and so on for at least the phased array channels)
        
    Returns:
        float representing PA average SNR
    
    """
    
    channels = [0, 1, 2, 3]

    SNRs = []
    for ch in channels:      
        RMS = noise_rms(traces[ch])

        SNR = np.amax(maximum_peak_to_peak_amplitude(traces[ch])) / (
            2 * RMS
        )

        SNRs.append(SNR)

    avg_SNR = np.mean(SNRs)

    return avg_SNR

def coherent_SNR(traces):
    """Obtain SNR for the phased array channels (0, 1, 2, 3) coherently summed waveform
    
    Args:
        traces (array): array containing channel waveforms (traces[0] should return ch 0's trace, traces[1] ch 1's trace, and so on for at least the phased array channels)
        
    Returns:
        float representing phased array coherently summed waveform SNR
    
    """  
      
    coherent_sum = coherent_sum(traces[:4])
    RMS = noise_rms(coherent_sum)
    
    SNR = np.amax(maximum_peak_to_peak_amplitude(coherent_sum)) / (
        2 * RMS
    )

    return SNR

def max_amp(traces):
    """Obtain the maximum peak to peak amplitude across the phased array channels (0, 1, 2, 3). They are not coherently summed or averaged for this.
    
    Args:
        traces (array): array containing channel waveforms (traces[0] should return ch 0's trace, traces[1] ch 1's trace, and so on for at least the phased array channels)
        
    Returns:
        float representing maximum peak to peak amplitude across all phased array channels 
    
    """
    
    max_amp = 0
    channels = [0, 1, 2, 3]
    for ch in channels:
        normalized_wf = traces[ch] / np.std(traces[ch])
        this_max = np.amax(
            maximum_peak_to_peak_amplitude(normalized_wf)
        )

        if this_max > max_amp:
            max_amp = this_max
            
    return max_amp

def impulsivity_and_maxspot(traces):
    """Obtain impulsivity of the phased array channels (0, 1, 2, 3) coherently summed waveform, and the maxspot which is the location (index) of peak power in coherently summed waveform hilbert envelope
    
    Args:
        traces (array): array containing channel waveforms (traces[0] should return ch 0's trace, traces[1] ch 1's trace, and so on for at least the phased array channels)
        
    Returns:
        float, int: float is the impulsivity of the phased array coherently summed waveform, and int is the maxspot
    
    """
    csw = coherent_sum(traces[:4])
    
    analytical_signal = signal.hilbert(
        csw
    )  
    
    envelope = np.abs(analytical_signal)
    
    maxv = np.argmax(envelope)
    
    maxspot = (
        maxv  ## index where the max voltage of the coherent sum is located
    )
    
    power_indexes = np.linspace(
        0, len(envelope) - 1, len(envelope)
    )  ## just a list of indices the same length as the array
    
    closeness = list(
        np.abs(power_indexes - maxv)
    )  ## create an array containing index distance to max voltage (lower the value, the closer it is)

    sorted_power = [x for _, x in sorted(zip(closeness, envelope))]
    
    CDF = np.cumsum(sorted_power)
    CDF = CDF / CDF[-1]

    CDF_avg = (np.mean(np.asarray([CDF])) * 2.0) - 1.0

    if CDF_avg < 0:
        CDF_avg = 0.0

    return CDF_avg, maxspot