from .glitch_unscrambler import unscramble
import numpy as np


def diff_sq(eventdata):
    """
    Returns sum of squared differences of samples across seams of 128-sample chunks.

    `eventdata`: channel waveform
    """
    runsum = 0.0
    deltaN = 64
    for chunk in range(len(eventdata) // 128 - 1):
        runsum += (eventdata[chunk * 128 + deltaN - 1] - eventdata[chunk * 128 + deltaN]) ** 2
    return np.sum(runsum)

def is_channel_scrambled(eventdata):
    """
    Returns test statistic T sensitive to presence glitches in an event.

    p(T|glitches present) has support on predominantly positive T.
    p(T|no glitches present) has support on predominantly negative T.

    To test whether a single event contains glitches, pick a value T0 and evaluate T.
    If T > T0, the event should be classified as containing glitches.
    T0 can be set to control the true-positive and false-positive rates.
    """
    eventdata_us = unscramble(eventdata)
    return diff_sq(eventdata) - diff_sq(eventdata_us)
