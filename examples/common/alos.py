"""Shared helpers for the ALOS-1 San Francisco scene runners."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .params import RadarParams

__all__ = ["SIZE", "load_raw", "save_scene"]

#: Raster size of the extracted dataset.
SIZE = 16384


def load_raw(path: str) -> np.ndarray:
    """Loads the extracted raw echoes (see wka/data/extract_alos.py)."""
    bin_path = Path(path)
    if not bin_path.exists():
        raise SystemExit(
            f"{bin_path} not found -- extract the CEOS product first with "
            "`python examples/wka/data/extract_alos.py`")
    return np.fromfile(bin_path, dtype=np.complex64).reshape((SIZE, SIZE))


def save_scene(image: np.ndarray, p: RadarParams, path: str,
               title: str) -> None:
    """Post-processes and saves a focused ALOS scene: flip, crop to the
    valid swath, percentile stretch and ground-square azimuth resampling."""
    image = np.flipud(image)
    crop = image[:, :9600]
    vmin = np.percentile(crop, 2.0)
    vmax = np.percentile(crop, 99.0)
    norm = np.clip((crop - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)

    dx_ground = (p.c / (2 * p.fs)) / np.sin(np.radians(38.0))
    aspect = (p.vr / p.prf) / dx_ground
    h, w = norm.shape
    out_h = int(h * aspect + 0.5)
    src_y = np.arange(out_h, dtype=np.float64) / aspect
    y0 = np.clip(np.floor(src_y).astype(np.int64), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wy = np.clip(src_y - y0, 0.0, 1.0)[:, np.newaxis]
    resized = norm[y0, :] * (1.0 - wy) + norm[y1, :] * wy

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10 * out_h / w))
    plt.imshow(resized, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.xlabel("Range (ground projected)")
    plt.ylabel("Azimuth")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
