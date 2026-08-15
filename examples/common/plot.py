"""Plotting helpers shared by the example runners and benchmarks."""

from __future__ import annotations

import numpy as np

__all__ = ["apply_style", "save_db_image", "print_targets"]

#: Okabe-Ito colorblind-safe palette used across all figures.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def apply_style() -> None:
    """Applies the shared publication style (call before plotting)."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def save_db_image(image: np.ndarray,
                  path: str,
                  title: str,
                  floor_db: float = -60.0) -> None:
    """Saves a magnitude image on a dB scale relative to its peak."""
    apply_style()
    import matplotlib.pyplot as plt

    display = 20.0 * np.log10(image / image.max() + 1e-12)
    # Keep at least ~2.5 output pixels per data sample so single-pixel
    # mainlobes stay visible on larger rasters.
    size = 6.4 * max(1.0, max(image.shape) / 512.0)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.grid(False)
    im = ax.imshow(display,
                   cmap="gray",
                   vmin=floor_db,
                   vmax=0.0,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="dB", fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Range (samples)")
    ax.set_ylabel("Azimuth (samples)")
    fig.savefig(path)
    plt.close(fig)


def print_targets(targets) -> None:
    for t in targets:
        print(f"      target: range {t.range_offset:+9.1f} m, "
              f"azimuth {t.azimuth_offset:+9.1f} m, rcs {t.rcs}")
