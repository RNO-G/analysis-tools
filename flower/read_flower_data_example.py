import json
import numpy as np
import sys
import matplotlib.pyplot as plt

from utils import read_flower_data, get_waveforms_and_metadata

fname = sys.argv[1]

data = read_flower_data(fname)

waveforms, metadata = get_waveforms_and_metadata(data)

fig, ax = plt.subplots()


run = data["run"]
station = data["station"]
label = f"st{station}_run{run}"

pps_flag = metadata["pps_flag"]


for trig, trig_mask in zip(["pps", "no_pps"], [pps_flag, ~pps_flag]):

    print(f"{np.sum(trig_mask)} of {len(trig_mask)} events are of type {trig}")
    for idx in range(5):
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        for jdx, (wfs, ax) in enumerate(zip(waveforms, axs.flatten())):
            ax.plot(wfs[idx], label=jdx, lw=1, c=f"C{jdx}")
            ax.legend()

        fig.supxlabel("samples")
        fig.supylabel("amplitude / ADC")

        fig.tight_layout()
        fig.savefig(f"waveforms_{idx}_{trig}_{label}.png")
