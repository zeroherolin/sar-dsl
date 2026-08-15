#!/usr/bin/env python3
"""Chirp Scaling on the CPU backend, end to end: simulate a synthetic
point-target scene, compile the kernel, focus it and save a PNG.

Usage:
    python run_point_target_cpu.py [--n 512] [--output csa_cpu.png]
"""

import argparse
import sys
import time
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_EXAMPLES), str(_EXAMPLES.parent / "python")]

from common.params import synthetic_params  # noqa: E402
from common.plot import print_targets, save_db_image  # noqa: E402
from common.simulate import demo_scene  # noqa: E402
from csa.algorithm import build_kernel, make_inputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",
                        type=int,
                        default=512,
                        help="raster size (power of two)")
    parser.add_argument("--output", default="csa_cpu.png")
    args = parser.parse_args()

    n = args.n
    params = synthetic_params(n)
    print(f"[1/3] Simulating {n}x{n} synthetic scene ...")
    raw, targets = demo_scene(n, params)
    print_targets(targets)

    print("[2/3] Compiling and running the CSA kernel (cpu backend) ...")
    kernel = build_kernel(n, params).compile("cpu")
    t0 = time.time()
    image = kernel(raw, *make_inputs(n, params))
    print(f"      focused in {time.time() - t0:.2f} s")

    output = Path(args.output).resolve()
    print(f"[3/3] Saving {output} ...")
    save_db_image(
        image, str(output),
        f"SAR-DSL Chirp Scaling (cpu), {n}x{n} synthetic point targets")
    print("done.")


if __name__ == "__main__":
    main()
