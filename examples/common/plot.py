"""Plotting helpers for the example runners."""

from __future__ import annotations

import numpy as np

__all__ = ["save_db_image", "print_targets"]


def save_db_image(image: np.ndarray, path: str, title: str,
                  floor_db: float = -60.0) -> None:
    """Saves a magnitude image on a dB scale relative to its peak."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display = 20.0 * np.log10(image / image.max() + 1e-12)
    plt.figure(figsize=(8, 8))
    plt.imshow(display, cmap="gray", vmin=floor_db, vmax=0.0)
    plt.colorbar(label="dB")
    plt.title(title)
    plt.xlabel("Range")
    plt.ylabel("Azimuth")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def print_targets(targets) -> None:
    for t in targets:
        print(f"      target: range {t.range_offset:+9.1f} m, "
              f"azimuth {t.azimuth_offset:+9.1f} m, rcs {t.rcs}")
