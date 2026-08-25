#!/usr/bin/env python3
"""Impulse-response figures, written to `benchmarks/assets/`.

    cpu_point_target_response.png  range/azimuth impulse-response cuts of the
                                three stripmap chains, with the ideal Hann
                                reference level
    cpu_pfa_sva_response.png    PFA range cut, uniform weighting vs SVA

These focus a scene to draw them, so they need the CPU backend.
`plot_cpu_hls_results.py` redraws the checked-in CPU/HLS measurements and
needs no toolchain.

Usage:
    python benchmarks/plot_cpu_impulse_response.py [--n 512] [--sva-n 256]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms import LABELS, focus_point_target, load  # noqa: E402
from common.plot import PALETTE, apply_style  # noqa: E402
from metrics import UPSAMPLE, upsample  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"


def _db_cut(cut: np.ndarray, span: float):
    """Upsampled impulse-response cut in dB, centered on the peak."""
    fine = upsample(cut.astype(complex))
    peak = int(np.argmax(fine))
    half = int(span * UPSAMPLE)
    lo, hi = max(0, peak - half), min(len(fine), peak + half + 1)
    x = (np.arange(lo, hi) - peak) / UPSAMPLE
    y = 20.0 * np.log10(fine[lo:hi] / fine[peak] + 1e-12)
    return x, y


def _decorate(ax, title: str, span: float) -> None:
    ax.set_title(title)
    ax.set_xlabel("Offset from peak (samples)")
    ax.set_xlim(-span, span)
    ax.set_ylim(-65.0, 2.0)


def point_target_response(n: int, span: float = 14.0) -> None:
    """Impulse-response cuts of the three stripmap chains."""
    import matplotlib.pyplot as plt
    fig, (ax_r, ax_a) = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)

    for (name, image), color in zip(focus_point_target(n), PALETTE):
        i, j = np.unravel_index(np.argmax(image), image.shape)
        for ax, cut in ((ax_r, image[i, :]), (ax_a, image[:, j])):
            x, y = _db_cut(cut, span)
            ax.plot(x, y, color=color, linewidth=1.4, label=LABELS[name])

    for ax, axis in ((ax_r, "Range"), (ax_a, "Azimuth")):
        ax.axhline(-31.5, color="0.45", linewidth=0.8, linestyle="--")
        _decorate(ax, f"{axis} impulse response", span)
    ax_r.set_ylabel("Normalized magnitude (dB)")
    ax_r.text(-13.4,
              -30.3,
              "ideal Hann reference (-31.5 dB)",
              fontsize=8,
              color="0.35")
    ax_a.legend(loc="upper right", fontsize=8.5)

    out = ASSETS / "cpu_point_target_response.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def sva_response(n: int = 256, span: float = 10.0) -> None:
    """PFA range cut: uniform aperture weighting vs SVA."""
    chain = load("pfa", n)
    uniform, filtered = chain.run(chain.compile_kernel())
    i = int(np.unravel_index(np.argmax(uniform), uniform.shape)[0])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for cut, label, color in ((uniform[i, :], "uniform weighting", PALETTE[0]),
                              (filtered[i, :], "SVA", PALETTE[1])):
        # The PFA image grid is 2x oversampled: one resolution cell
        # spans two pixels.
        x, y = _db_cut(np.asarray(cut, dtype=np.float64), 2 * span)
        ax.plot(x / 2.0, y, color=color, linewidth=1.4, label=label)
    ax.axhline(-13.26, color="0.45", linewidth=0.8, linestyle="--")
    ax.text(-9.6,
            -12.1,
            "uniform first sidelobe (-13.3 dB)",
            fontsize=8,
            color="0.35")
    _decorate(ax, "PFA range impulse response", span)
    ax.set_xlabel("Offset from peak (resolution cells)")
    ax.set_ylabel("Normalized magnitude (dB)")
    ax.legend(loc="upper right", fontsize=8.5)

    out = ASSETS / "cpu_pfa_sva_response.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",
                        type=int,
                        default=512,
                        help="stripmap point-target scene size")
    parser.add_argument("--sva-n",
                        type=int,
                        default=256,
                        help="PFA scene size for the SVA figure")
    args = parser.parse_args()

    ASSETS.mkdir(exist_ok=True)
    apply_style()
    point_target_response(args.n)
    sva_response(args.sva_n)


if __name__ == "__main__":
    main()
