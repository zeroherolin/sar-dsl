"""Point-target and scene image-quality reporting for the runners.

Thin printing layer over `benchmarks/metrics.py`, which owns the
measurement code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from metrics import measure_image, urban_contrast  # noqa: E402

__all__ = ["print_focus_quality", "print_scene_contrast"]


def print_focus_quality(image: np.ndarray, expected_peak=None) -> dict:
    """Prints and returns peak position plus range/azimuth IRW, PSLR, ISLR
    of the brightest point target in `image`."""
    m = measure_image(image, expected_peak=expected_peak)
    peak = m["peak"]
    line = f"      peak at ({peak[0]}, {peak[1]})"
    if expected_peak is not None:
        line += (f", expected ({expected_peak[0]}, {expected_peak[1]}), "
                 f"error ({m['peak_error'][0]:+d}, {m['peak_error'][1]:+d})")
    print(line)
    for axis in ("range", "azimuth"):
        cut = m[axis]
        print(f"      {axis:>7}: IRW {cut['irw']:5.2f} samples, "
              f"PSLR {cut['pslr']:6.1f} dB, ISLR {cut['islr']:6.1f} dB")
    return m


def print_scene_contrast(image: np.ndarray) -> float:
    """Prints and returns the urban-area contrast of a focused scene
    (std/mean over the brightest multilooked tile; higher is sharper)."""
    contrast = urban_contrast(image)
    print(f"      urban contrast {contrast:.3f}")
    return contrast
