import numpy as np
from scipy.stats import binomtest


def binomtest_glitch_fraction(glitch_arr, channel_list, config_glitching=None):
    '''Perform binomial test on glitch fractions for each channel (with p0=0.1 - can be adjusted from the config_glitching.py).'''
    if config_glitching is None:
        alpha = 0.01
        pvalue = 0.1
        ci_level = 0.99
        strong_ci_threshold = 0.3
        moderate_ci_threshold = 0.2
    else:
        alpha = config_glitching.get("alpha", 0.01)
        pvalue = config_glitching.get("pvalue", 0.1)
        ci_level = config_glitching.get("ci_level", 0.99)
        strong_ci_threshold = config_glitching.get("strong_ci_threshold", 0.3)
        moderate_ci_threshold = config_glitching.get("moderate_ci_threshold", 0.2)
    
    glitch_info = {}
    n_events = glitch_arr.shape[1]

    for ch in channel_list:
        glitch_ch = glitch_arr[ch]
        n_glitches = np.sum(glitch_ch > 0)

        result = binomtest(n_glitches, n_events, p=pvalue, alternative="greater")
        pval = result.pvalue
        statistic = result.statistic
        confidence_interval = result.proportion_ci(confidence_level=ci_level)

        if pval > alpha:
            validation = "NO EXCESSIVE GLITCHING"
        else:
            if confidence_interval.low > strong_ci_threshold:
                validation = "STRONG EXCESSIVE GLITCHING"
            elif confidence_interval.low > moderate_ci_threshold:
                validation = "MODERATE EXCESSIVE GLITCHING"
            else:
                validation = "WEAK EXCESSIVE GLITCHING"

        glitch_info[ch] = {
            "n_glitches": int(n_glitches),
            "n_events": int(n_events),
            "pval": float(pval),
            "confidence_interval": (float(confidence_interval.low), float(confidence_interval.high)),
            "glitch_fraction": float(n_glitches / n_events),
            "validation": validation,
        }

    return glitch_info