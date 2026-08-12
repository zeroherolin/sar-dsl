#!/usr/bin/env python3
"""Emits the complete omega-K imaging kernel as a Vitis HLS C++ design.

Usage:
    python emit_hls.py [--n 256] [--output wka_hls.cpp]

Requires the ScaleHLS toolchain (`make scalehls`). The generated top
function takes the raw data as two float planes (re, im), followed by the
frequency axes and window vectors, then the output magnitude plane.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import synthetic_params      # noqa: E402
from wka_dsl import build_wka_kernel        # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256,
                        help="raster size (power of two)")
    parser.add_argument("--output", default="wka_hls.cpp")
    args = parser.parse_args()

    kernel = build_wka_kernel(args.n, synthetic_params(args.n))
    print(f"Emitting {args.n}x{args.n} omega-K kernel through "
          "ScaleHLS-HIDA ...")
    design = kernel.compile(backend="scalehls")
    shutil.copyfile(design.cpp_path, args.output)
    lines = design.source().count("\n")
    print(f"done: {args.output} ({lines} lines, flow={design.flow})")


if __name__ == "__main__":
    main()
