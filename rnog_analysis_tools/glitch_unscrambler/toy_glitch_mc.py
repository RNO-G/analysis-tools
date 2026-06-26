"""
Toy Monte Carlo: RNO-G deep-channel noise + "glitching".

What it does
------------
1. Builds the complex response of an RNO-G *deep* channel signal chain
   (IGLU + DRAB + fiber + RADIANT) from the locally shipped placeholder
   measurement (`analog_components.load_amp_response('deep_impulse')`).
2. Generates thermal-noise traces with `channelGenericNoiseAdder` (Rayleigh-
   distributed per-bin amplitudes + uniform random phases = Gaussian thermal
   noise) and gives them the spectral imprint of that response: white noise at
   the antenna -> shaped by the signal chain.
3. Simulates "glitching" the way the RNO-G LAB4D digitizer actually does it:
   the 64-sample readout blocks get scrambled by a *specific, deterministic*
   permutation (even blocks shift +2 positions, odd blocks shift -2, cyclically).
   This is exactly the pattern that `NuRadioReco.modules.RNO_G.channelGlitchDetector`
   is built to detect/undo, so we model the glitch as the inverse of that module's
   `unscramble` operation. (A naive *random* block shuffle is invisible to the
   detector and is therefore NOT a faithful glitch model -- see the sanity check
   at the end of this script.)
4. Compares the average amplitude spectrum of clean vs. glitched traces, and
   verifies that the simulated glitches are flagged by the real detector logic.

All FFTs use NuRadio's normalization convention (`NuRadioReco.utilities.fft`,
spectrum in V/GHz). The glitch injects sharp discontinuities at the 64-sample
block seams, so the glitched spectrum picks up extra high-frequency power and a
comb at multiples of f_sample / 64.

Run:  python toy_glitch_mc.py
"""
import numpy as np
import matplotlib.pyplot as plt

from NuRadioReco.detector.RNO_G import analog_components as ac
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.modules.RNO_G.channelGlitchDetector import unscramble, diff_sq
from NuRadioReco.utilities import units, fft


# ---------------------------------------------------------------- parameters
SAMPLING_RATE = 3.2 * units.GHz       # RNO-G LAB4D deep-channel sampling rate
N_SAMPLES     = 2048                  # samples per readout window (LAB4D buffer)
BLOCK_SIZE    = 64                   # LAB4D sampling-block size (the real value)
N_EVENTS      = 20000                 # toy-MC traces to average over
NOISE_VRMS    = 10 * units.micro * units.volt  # thermal noise V_rms at antenna
SEED          = 42
p             = 0.1

# The real glitch is the exact inverse of the detector's `unscramble`: each
# 64-sample block is shifted cyclically by +/-2 positions (even blocks +2, odd
# blocks -2), which is a true permutation that `unscramble` reverses exactly.
_NBLOCKS = N_SAMPLES // BLOCK_SIZE
_blk = np.arange(_NBLOCKS)
_src_blk = np.where(_blk % 2 == 0, (_blk + 2) % _NBLOCKS, (_blk - 2) % _NBLOCKS)
_SCRAMBLE_PERM = (_src_blk[:, None] * BLOCK_SIZE + np.arange(BLOCK_SIZE)).ravel()

print(np.unique(_SCRAMBLE_PERM, return_counts=True))

def get_deep_response(freqs):
    """Complex response H(f) of an RNO-G deep channel at the given freqs."""
    amp = ac.load_amp_response(amp_type='deep_impulse')
    # gain(f) is the magnitude, phase(f) already returns exp(i*phi)
    return amp['gain'](freqs) * amp['phase'](freqs)


def generate_noise_trace(noise_adder, response):
    """White thermal noise (channelGenericNoiseAdder) shaped by the response."""
    # generate the white thermal-noise spectrum directly in the frequency domain
    spec = noise_adder.bandlimited_noise(
        min_freq=0, max_freq=None, n_samples=N_SAMPLES,
        sampling_rate=SAMPLING_RATE, amplitude=NOISE_VRMS,
        type='rayleigh', time_domain=False)
    spec = spec * response                       # imprint the channel response
    return fft.freq2time(spec, SAMPLING_RATE, n=N_SAMPLES)


def glitch(trace):
    """Apply the real LAB4D 64-block scramble (exact inverse of unscramble)."""
    return trace[_SCRAMBLE_PERM]


def glitch_test_statistic(trace):
    """Reproduce channelGlitchDetector's TS; >0 means flagged as a glitch."""
    ts = diff_sq(trace, BLOCK_SIZE) - diff_sq(unscramble(trace, BLOCK_SIZE), BLOCK_SIZE)
    return ts / np.var(trace)


def main():
    rng = np.random.default_rng(SEED)
    noise_adder = channelGenericNoiseAdder()
    noise_adder.begin(seed=SEED)

    freqs = fft.freqs(N_SAMPLES, SAMPLING_RATE)
    response = get_deep_response(freqs)

    clean_psum = np.zeros(len(freqs))
    glitch_psum = np.zeros(len(freqs))
    clean_flagged = 0           # flags on the always-clean trace -> false positives
    n_glitched = 0              # how many events were actually glitched
    n_glitched_flagged = 0      # of those, how many the detector caught -> TPR
    for _ in range(N_EVENTS):
        clean = generate_noise_trace(noise_adder, response)

        is_glitched = rng.random() < p
        glitched = glitch(clean) if is_glitched else clean

        clean_psum += np.abs(fft.time2freq(clean, SAMPLING_RATE))**2
        glitch_psum += np.abs(fft.time2freq(glitched, SAMPLING_RATE))**2
        clean_flagged += glitch_test_statistic(clean) > 0
        if is_glitched:
            n_glitched += 1
            n_glitched_flagged += glitch_test_statistic(glitched) > 0

    clean_spec = np.sqrt(clean_psum / N_EVENTS)
    glitch_spec = np.sqrt(glitch_psum / N_EVENTS)

    f_block = SAMPLING_RATE / BLOCK_SIZE   # comb spacing introduced by glitching
    f_MHz = freqs / units.MHz

    # ----- report ----------------------------------------------------------
    # Out of band (above the deep-channel passband) the clean spectrum is
    # essentially zero, so any power there is purely glitch-induced.
    oob = freqs > 1.05 * units.GHz
    print(f"Deep-channel passband peak at "
          f"{f_MHz[np.argmax(clean_spec)]:.0f} MHz")
    print(f"Glitch comb spacing f_sample/{BLOCK_SIZE} = {f_block / units.MHz:.1f} MHz")
    print(f"Out-of-band (>1.05 GHz) mean amplitude ratio glitched/clean = "
          f"{glitch_spec[oob].mean() / clean_spec[oob].mean():.2e}")
    # sanity check against the real detector logic
    fpr = clean_flagged / N_EVENTS
    tpr = n_glitched_flagged / n_glitched if n_glitched else float('nan')
    print(f"glitch fraction p = {p}: {n_glitched}/{N_EVENTS} events glitched")
    print(f"channelGlitchDetector: false positive rate "
          f"{clean_flagged}/{N_EVENTS} = {100 * fpr:.1f}%, "
          f"true positive rate {n_glitched_flagged}/{n_glitched} = {100 * tpr:.1f}%")

    # ----- plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(f_MHz, clean_spec, label="clean (response imprint)", lw=2)
    ax.plot(f_MHz, glitch_spec, label="glitched (LAB4D 64-sample block scramble)",
            lw=1.2, alpha=0.9)
    for k in range(1, int((SAMPLING_RATE / 2) / f_block) + 1):
        ax.axvline(k * f_block / units.MHz, color='grey', ls=':', lw=0.6,
                   zorder=0)

    ax.set_ylim(1e-2, None)
    ax.set_xlim(0, 1200)
    ax.set_xlabel("frequency / MHz")
    ax.set_ylabel("average |spectrum|  [V/GHz]")
    ax.set_title("RNO-G deep channel: clean vs. glitched noise spectrum")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = f"toy_glitch_mc_{p}.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

if __name__ == "__main__":
    main()
