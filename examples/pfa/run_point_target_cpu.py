#!/usr/bin/env python3
"""PFA on the CPU backend, end to end: simulate a spotlight point-target
collection, compile the kernel and save a PNG.

Usage:
    python run_point_target_cpu.py [--n 512] [--output pfa_cpu.png]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(_EXAMPLES),
    str(_EXAMPLES.parent / "python"),
    str(_EXAMPLES.parent / "benchmarks")
]

from common.plot import save_db_image  # noqa: E402
from pfa.algorithm import build_kernel, make_inputs  # noqa: E402
from pfa.geometry import Geometry  # noqa: E402
from metrics import measure_cut  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",
                        type=int,
                        default=512,
                        help="raster size (power of two)")
    parser.add_argument("--output", default="pfa_cpu.png")
    args = parser.parse_args()

    n = args.n
    geometry = Geometry(n)
    targets = geometry.demo_targets()
    print(f"[1/3] Simulating {len(targets)} point targets "
          f"({n}x{n} polar phase history) ...")
    for x, y in targets:
        print(f"      target: x {x:+7.1f} px, y {y:+7.1f} px")
    raw = geometry.simulate(targets)

    print("[2/3] Compiling and running the PFA kernel (cpu backend) ...")
    kernel = build_kernel(n, geometry).compile("cpu")
    t0 = time.time()
    uniform, filtered = kernel(raw, *make_inputs(n, geometry))
    print(f"      focused in {time.time() - t0:.2f} s")

    # Every scatterer must land on its predicted pixel (2x oversampled
    # output); SVA must beat the -13 dB sidelobes of uniform weighting.
    for x, y in targets:
        row, col = n + int(round(2 * y)), n + int(round(2 * x))
        window = filtered[row - 2:row + 3, col - 2:col + 3]
        assert window.max() == filtered[row, col], (x, y)
    i, j = np.unravel_index(np.argmax(uniform), uniform.shape)
    for name, image in (("uniform", uniform), ("SVA", filtered)):
        cut = measure_cut(image[i, :])
        print(f"      {name:>7}: range PSLR {cut['pslr']:6.1f} dB, "
              f"IRW {cut['irw'] / 2:.2f} cells")

    output = Path(args.output).resolve()
    print(f"[3/3] Saving {output} ...")
    save_db_image(
        filtered, str(output),
        f"SAR-DSL PFA + SVA from Python-defined operators, "
        f"{n}x{n}")
    print("done.")


if __name__ == "__main__":
    main()
