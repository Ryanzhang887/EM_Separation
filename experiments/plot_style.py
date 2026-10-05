"""Shared plotting style for the paper's simulation figures.

Trajectories use tab10 colors, gray crosses for initial means, colored dots
for final means, and red stars for true means. Figure sizes are in inches.
"""

from cycler import cycler

TAB10 = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
LABELS = {"fixed_delta": r"$\Delta=6$", "power_045": r"$\Delta=d^{0.45}$",
          "sqrt_10": r"$\Delta=10\sqrt{d}$"}
WIDTH = 6.6
RC = {
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "mathtext.fontset": "dejavusans", "axes.labelsize": 8.5,
    "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 8, "legend.frameon": False,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": "0.45", "axes.linewidth": .65,
    "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
    "axes.prop_cycle": cycler(color=TAB10),
    "xtick.color": "0.25", "ytick.color": "0.25",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": .55, "ytick.major.width": .55,
    "lines.linewidth": .85, "lines.markersize": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.pad_inches": .04,
}


def zero_line(ax):
    ax.axhline(0, color="0.6", linestyle=":", linewidth=.65, zorder=0)


def component_style(index):
    # Cycle through ten colors, then distinguish additional components by dashes.
    return {"color": TAB10[index % 10], "linestyle": "-" if index < 10 else "--",
            "linewidth": .75, "alpha": .6}
