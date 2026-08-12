#!/usr/bin/env python3
"""Focuses the ALOS-1 San Francisco dataset with the SAR-DSL WKA kernel.

Prerequisites:
    1. Extract the raw echoes from the CEOS L1.0 product (once):
         python data/extract_alos.py
       which produces `alos_raw_16384x16384.bin` (complex64, 2 GiB).
    2. Run this script (tens of GiB of RAM at full 16384^2 size):
         python run_alos.py [--bin alos_raw_16384x16384.bin]

The post-processing (flip, crop, percentile stretch, aspect resampling)
matches the original reference script.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wka_dsl import build_wka_kernel, make_kernel_inputs  # noqa: E402
from wka_numpy import ALOS_PARAMS                          # noqa: E402

N = 16384


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", default="alos_raw_16384x16384.bin",
                        help="extracted raw data (complex64, 16384x16384)")
    parser.add_argument("--output", default="san_francisco_wka.png")
    args = parser.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.exists():
        raise SystemExit(
            f"{bin_path} not found -- run `python data/extract_alos.py` "
            "first (see data/README notes in extract_alos.py)")

    print(f"[1/4] Loading {bin_path} ...")
    raw = np.fromfile(bin_path, dtype=np.complex64).reshape((N, N))

    print("[2/4] Compiling WKA kernel ...")
    kernel = build_wka_kernel(N, ALOS_PARAMS).compile("cpu")
    fa, fr, win_r, win_a = make_kernel_inputs(N, ALOS_PARAMS)

    print("[3/4] Running WKA imaging (this is the heavy part) ...")
    t0 = time.time()
    image = kernel(raw, fa, fr, win_r, win_a)
    print(f"      focused in {time.time() - t0:.1f} s")
    del raw

    print("[4/4] Post-processing and saving ...")
    image = np.flipud(image)
    crop = image[:, :9600]                      # valid swath
    vmin = np.percentile(crop, 2.0)
    vmax = np.percentile(crop, 99.0)
    norm = np.clip((crop - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)

    # Resample azimuth to approximate ground-square pixels.
    dx_ground = (ALOS_PARAMS.c / (2 * ALOS_PARAMS.fs)) / np.sin(
        np.radians(38.0))
    aspect = (ALOS_PARAMS.vr / ALOS_PARAMS.prf) / dx_ground
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
    plt.title("San Francisco Bay (ALOS-1, SAR-DSL omega-K)")
    plt.xlabel("Range (ground projected)")
    plt.ylabel("Azimuth")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
