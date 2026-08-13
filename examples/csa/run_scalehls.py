#!/usr/bin/env python3
"""Chirp Scaling on the ScaleHLS backend, end to end: trace the complete
kernel and emit a Vitis HLS C++ design (requires `make scalehls`).

CSA is interpolation-free, so this design contains only phase multiplies
and Stockham FFT loop nests.

Usage:
    python run_scalehls.py [--n 256] [--output csa_hls.cpp]
"""

import argparse
import shutil
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_EXAMPLES), str(_EXAMPLES.parent / "python")]

from common.params import synthetic_params        # noqa: E402
from csa.algorithm import build_kernel            # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256,
                        help="raster size (power of two)")
    parser.add_argument("--output", default="csa_hls.cpp")
    args = parser.parse_args()

    print(f"[1/2] Emitting the {args.n}x{args.n} CSA kernel through "
          "ScaleHLS-HIDA ...")
    kernel = build_kernel(args.n, synthetic_params(args.n))
    design = kernel.compile(backend="scalehls")

    output = Path(args.output).resolve()
    shutil.copyfile(design.cpp_path, output)
    print(f"[2/2] Saved {output}")
    print(f"done: {design.source().count(chr(10))} lines of HLS C++ "
          f"(flow={design.flow}, top function 'csa')")


if __name__ == "__main__":
    main()
