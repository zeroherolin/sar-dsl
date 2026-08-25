#!/usr/bin/env python3
"""Figures redrawn from the recorded measurements, for the top-level README.

    hls_resource_utilization.png  production designs against device budgets,
                              with the hand-written omega-K baseline
    hls_budget_sweep.png      compiler-selected designs across the configured
                              BRAM, URAM, and LUTRAM caps
    cpu_throughput.png        SAR-DSL CPU throughput across the measured
                              scene sizes
    cpu_speedup.png           SAR-DSL CPU speedup over the NumPy reference
                              across the measured scene sizes

Both read checked-in JSON measurements rather than re-running anything. That
keeps the figures and the documented values on one data source and makes them
regenerable on a machine with neither Vitis nor the reference host.

Unlike `plot_cpu_impulse_response.py`, nothing here focuses a scene, so the
figures redraw identically on any machine.

Usage:
    python benchmarks/plot_cpu_hls_results.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
RESULTS = HERE / "results"

# This runner reads recorded numbers rather than running a chain, so it must
# import without the package or a build on the path -- that independence is
# what lets the figures be redrawn anywhere. Only the shared plot style comes
# from outside, and it is reached directly.
sys.path[:0] = [
    str(HERE),
    str(HERE.parent / "examples"),
    str(HERE.parent / "python")
]

from common.plot import PALETTE, apply_style  # noqa: E402
from run_hls_resources import budget_figure  # noqa: E402

#: The production synthesis record the resource figure is drawn from.
_PRODUCTION = RESULTS / "hls_algorithms_c64_production_vitis_2022_2.json"

_CPU_PERFORMANCE = RESULTS / "cpu_performance_c128_llvm_22.json"
_BUDGET_SWEEP = RESULTS / "hls_budget_sweep_n1024.json"

#: The independent hand-written omega-K implementation, kept beside the
#: sources it was measured from.
_HANDWRITTEN = (HERE.parent / "examples" / "wka" / "handwritten_hls" /
                "reports" / "production_csynth.json")

#: Resource axes, as (record key, budget key, axis label). Ordered the way
#: the constraint tables read: memory tiers first, then arithmetic, then
#: the fabric totals.
_RESOURCES = (
    ("bram18k", "bram18k_budget", "BRAM18K"),
    ("uram", "uram_budget", "URAM"),
    ("dsp", "dsp_budget", "DSP"),
    ("ff", "ff_budget", "FF"),
    ("lut", "lut_budget", "LUT"),
)

_SHORT = {
    "omega-K": "omega-K",
    "Range-Doppler": "RDA",
    "Chirp Scaling": "CSA",
    "Polar Format": "PFA",
}


def _production_designs() -> tuple:
    """(constraints, [(label, record), ...]) from the recorded synthesis."""
    data = json.loads(_PRODUCTION.read_text())
    designs = [(_SHORT.get(d["algorithm"], d["algorithm"]), d)
               for d in data["designs"]]
    return data["constraints"], designs


def resource_utilization() -> None:
    """Recorded utilization as a share of the budget, per resource.

    A share rather than a count: the five resources differ by four orders
    of magnitude, so plotting them against a common axis is only meaningful
    once each is normalized by the budget it is actually constrained by.
    The 100% line is then the constraint itself.
    """
    import matplotlib.pyplot as plt

    constraints, designs = _production_designs()
    hand = json.loads(_HANDWRITTEN.read_text())["report"]

    labels = [label for label, _ in designs] + ["omega-K\n(hand-written)"]
    records = [record for _, record in designs] + [hand]
    colors = list(PALETTE[:len(designs)]) + ["0.55"]

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    width = 0.15
    positions = np.arange(len(_RESOURCES))

    for index, (label, record, color) in enumerate(zip(labels, records,
                                                       colors)):
        shares = [
            100.0 * record[key] / constraints[budget]
            for key, budget, _ in _RESOURCES
        ]
        offset = (index - (len(labels) - 1) / 2.0) * width
        # The hand-written baseline is an independent omega-K implementation,
        # shown in gray so it is distinct from generated designs.
        bars = ax.bar(positions + offset,
                      shares,
                      width,
                      color=color,
                      label=label,
                      edgecolor="white",
                      linewidth=0.4)
        ax.bar_label(bars, fmt="%.0f", fontsize=6.5, padding=1.5)

    ax.axhline(100.0, color="#C00000", linewidth=0.9, linestyle="--")
    ax.text(len(_RESOURCES) - 0.52,
            103.0,
            "budget (80% of device)",
            fontsize=8,
            color="#C00000",
            ha="right")
    ax.set_xticks(positions)
    ax.set_xticklabels([name for _, _, name in _RESOURCES])
    ax.set_ylabel("Utilization (% of budget)")
    ax.set_ylim(0.0, 122.0)
    ax.set_title("HLS production design utilization on xcvu13p "
                 f"({constraints['clock_ns']:g} ns target)")
    ax.legend(fontsize=7.5, ncol=5, loc="upper left")

    out = ASSETS / "hls_resource_utilization.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def _cpu_sweep() -> dict:
    """Returns recorded ``{algorithm: [(n, warm_s, numpy_s), ...]}``."""
    data = json.loads(_CPU_PERFORMANCE.read_text())
    rows = {}
    for measurement in data["measurements"]:
        rows.setdefault(measurement["algorithm"], []).append(
            (int(measurement["n"]), float(measurement["warm_s"]),
             float(measurement["numpy_s"])))
    return {name: sorted(points) for name, points in rows.items()}


def cpu_throughput() -> None:
    """Recorded warm throughput against scene size."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext
    from matplotlib.transforms import Bbox

    sweep = _cpu_sweep()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")

    all_sizes = []
    for index, (name, points) in enumerate(sweep.items()):
        sizes = np.array([n for n, _, _ in points], dtype=float)
        throughput = np.array([n * n / warm for n, warm, _ in points])
        all_sizes.extend(sizes)
        color = PALETTE[index % len(PALETTE)]
        ax.plot(sizes,
                throughput,
                "o-",
                color=color,
                linewidth=1.6,
                markersize=4,
                label=name)
        if len(sizes) >= 2:
            slope = float(
                np.polyfit(np.log10(sizes), np.log10(throughput), 1)[0])
            ax.annotate(f"  {slope:+.2f}",
                        xy=(sizes[-1], throughput[-1]),
                        fontsize=7.5,
                        color=color,
                        va="center")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    lo = int(np.floor(np.log2(min(all_sizes))))
    hi = int(np.ceil(np.log2(max(all_sizes))))
    ax.set_xticks([2**power for power in range(lo, hi + 1)])
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2))
    ax.set_xlabel("Input edge N  (NxN samples)")
    ax.set_ylabel("Throughput  (samples / s)")
    ax.set_title("SAR-DSL CPU throughput vs scene size (c128)")
    ax.legend(fontsize=8.5)

    out = ASSETS / "cpu_throughput.png"
    fig.savefig(out, bbox_inches=Bbox.from_bounds(0, 0, 7.2, 4.2))
    plt.close(fig)
    print(f"wrote {out}")


def cpu_speedup() -> None:
    """Speedup over the NumPy reference against scene size.

    The complement of the throughput figure: throughput says how fast the
    compiled chain runs, this says what the compilation bought against the
    same algorithm expressed in NumPy. Both axes are logarithmic because
    the scene sizes are octaves and the speedups span more than one.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    sweep = _cpu_sweep()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")

    for index, (name, points) in enumerate(sweep.items()):
        # A reference time rounded to 0.00 s carries no ratio.
        usable = [(n, warm, ref) for n, warm, ref in points if ref > 0.0]
        if not usable:
            continue
        sizes = np.array([n for n, _, _ in usable], dtype=float)
        speedup = np.array([ref / warm for _, warm, ref in usable])
        ax.plot(sizes,
                speedup,
                "o-",
                color=PALETTE[index % len(PALETTE)],
                linewidth=1.6,
                markersize=4,
                label=name)

    ax.axhline(1.0, color="0.45", linewidth=0.8, linestyle="--")
    ax.text(140.0, 1.12, "NumPy reference", fontsize=8, color="0.35")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    all_sizes = [n for points in sweep.values() for n, _, _ in points]
    lo = int(np.floor(np.log2(min(all_sizes))))
    hi = int(np.ceil(np.log2(max(all_sizes))))
    ax.set_xticks([2**power for power in range(lo, hi + 1)])
    from matplotlib.ticker import LogFormatterMathtext
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2))
    ax.set_xlabel("Input edge N  (NxN samples)")
    ax.set_ylabel("Speedup over NumPy (x)")
    ax.set_title("SAR-DSL CPU speedup vs NumPy (c128)")
    ax.legend(fontsize=8.5, loc="upper left")

    out = ASSETS / "cpu_speedup.png"
    fig.savefig(out, bbox_inches=Bbox.from_bounds(0, 0, 7.2, 4.2))
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ASSETS.mkdir(exist_ok=True)
    apply_style()
    resource_utilization()
    budget = json.loads(_BUDGET_SWEEP.read_text())
    budget_figure(budget["results"], int(budget["size"]))
    cpu_throughput()
    cpu_speedup()


if __name__ == "__main__":
    main()
