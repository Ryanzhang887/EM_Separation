"""Shared figure style, visually matched to Figure 3 of the supplied PDF.

The source figure/style files were not supplied. These settings reconstruct
its visible conventions: tab10 colors, light thin paths, gray start crosses,
colored end dots, red truth stars, boxed axes, and no grid. Figure sizes are
in publication inches so labels remain readable at the appendix text width.
"""

from cycler import cycler
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, SymmetricalLogLocator
import numpy as np

TAB10 = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
COLORS = {"fixed_delta": TAB10[0], "power_045": TAB10[0],
          "power_0495": TAB10[1], "sqrt_10": TAB10[2]}
LINESTYLES = {"fixed_delta": "-", "power_045": "-",
              "power_0495": "--", "sqrt_10": ":"}
LABELS = {"fixed_delta": r"$\Delta=6$", "power_045": r"$\Delta=d^{0.45}$",
          "power_0495": r"$\Delta=d^{0.495}$", "sqrt_10": r"$\Delta=10\sqrt{d}$"}
SEED_MARKERS = ("o", "^", "s")
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


def regime_key(name):
    return name.removeprefix("batch_16_")


def dimension_label(d):
    exponent = len(str(d)) - 1
    return rf"$10^{{{exponent}}}$" if d == 10 ** exponent else str(d)


def regime_handles(names):
    return [Line2D([], [], color=COLORS[regime_key(name)],
                   linestyle=LINESTYLES[regime_key(name)], linewidth=1.2,
                   label=LABELS[regime_key(name)]) for name in names]


def zero_line(ax):
    ax.axhline(0, color="0.6", linestyle=":", linewidth=.65, zorder=0)


def signed_scale(ax, linthresh):
    """Readable signed axes with padding in display space, including SD bars.

    Using dataLim also includes error-bar collections; relim() would omit them.
    Limit the tick count so small panels do not print every power of ten.
    """
    low, high = ax.dataLim.intervaly
    low, high = min(0., low), max(0., high)
    ax.set_yscale("symlog", linthresh=linthresh)
    transform = ax.yaxis.get_transform()
    limits = transform.transform(np.array([low, high]))
    pad = max(limits[1] - limits[0], linthresh) * .055
    lower, upper = transform.inverted().transform(limits + np.array([-pad, pad]))
    if low == 0:
        lower = -.08 * linthresh
    ax.set_ylim(lower, upper)
    locator = SymmetricalLogLocator(base=10, linthresh=linthresh)
    locator.set_params(numticks=5)
    candidates = [tick for tick in locator.tick_values(lower, upper) if lower <= tick <= upper]
    # Across many decades, +/- linthresh can otherwise overlap the zero label.
    height_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    tlo, thi = transform.transform(np.array([lower, upper]))
    accepted = []
    for tick in sorted(candidates, key=abs):
        position = (transform.transform(np.array([tick]))[0] - tlo) / (thi - tlo)
        if all(abs(position - previous) * height_pt >= 16 for _, previous in accepted):
            accepted.append((tick, position))
    ax.yaxis.set_major_locator(FixedLocator(sorted(tick for tick, _ in accepted)))
    zero_line(ax)


def component_style(index):
    # Preserve the original ten colors; distinguish the next ten by dashes.
    return {"color": TAB10[index % 10], "linestyle": "-" if index < 10 else "--",
            "linewidth": .75, "alpha": .6}
