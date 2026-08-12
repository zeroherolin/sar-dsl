#!/usr/bin/env python3
"""Focuses a synthetic point-target scene with the SAR-DSL WKA kernel.

Usage:
    python run_synthetic.py [--n 512] [--output wka_synthetic.png]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import demo_scene, synthetic_params           # noqa: E402
from wka_dsl import build_wka_kernel, make_kernel_inputs     # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512,
                        help="raster size (power of two)")
    parser.add_argument("--output", default="wka_synthetic.png")
    args = parser.parse_args()

    n = args.n
    params = synthetic_params(n)
    print(f"[1/4] Simulating {n}x{n} synthetic scene ...")
    raw, targets = demo_scene(n, params)
    for t in targets:
        print(f"      target: range {t.range_offset:+9.1f} m, "
              f"azimuth {t.azimuth_offset:+9.1f} m, rcs {t.rcs}")

    print("[2/4] Compiling WKA kernel (CPU backend) ...")
    t0 = time.time()
    kernel = build_wka_kernel(n, params)
    compiled = kernel.compile("cpu")
    print(f"      compiled in {time.time() - t0:.2f} s")

    print("[3/4] Running WKA imaging ...")
    fa, fr, win_r, win_a = make_kernel_inputs(n, params)
    t0 = time.time()
    image = compiled(raw, fa, fr, win_r, win_a)
    print(f"      focused in {time.time() - t0:.2f} s")

    print(f"[4/4] Saving {args.output} ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display = 20.0 * np.log10(image / image.max() + 1e-12)
    plt.figure(figsize=(8, 8))
    plt.imshow(display, cmap="gray", vmin=-60.0, vmax=0.0)
    plt.colorbar(label="dB")
    plt.title(f"SAR-DSL omega-K, {n}x{n} synthetic point targets")
    plt.xlabel("Range")
    plt.ylabel("Azimuth")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print("done.")


if __name__ == "__main__":
    main()
