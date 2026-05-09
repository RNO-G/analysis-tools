import matplotlib as mpl
from cycler import cycler

COLORS = [
    "#4477AA",  # strong blue
    "#EE6677",  # coral / red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # light blue
    "#AA3377",  # purple

    "#882255",  # wine
    "#44AA99",  # teal
    "#084C1F",  # deep green
    "#332288",  # indigo
    "#AA4499",  # magenta
    "#771122",  # burgundy

    "#7089A1",  # steel blue
    "#DDCC77",  # sand
    "#B0D8EC",  # pale blue
    "#CC6677",  # dusty red
    "#999933",  # olive
    "#DCB43C",  # warm yellow

    "#006699",  # dark cyan
    "#0099CC",  # vivid cyan
    "#9955AA",  # lavender purple
    "#55AA55",  # bright green
    "#CC7711",  # warm orange
    "#555555",  # neutral gray
]

def set_plot_style():
    mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 17,

    'axes.labelsize': 17,
    'axes.titlesize': 18,
    'axes.linewidth': 1.2,
    'axes.grid': False,

    'axes.prop_cycle': cycler('color', COLORS),

    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,

    'lines.linewidth': 1.6,
    'lines.antialiased': True,
    'lines.markersize': 6,

    'legend.fontsize': 14,
    'legend.frameon': False,
    'legend.handlelength': 1,
    'legend.borderpad': 0.3,

    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})