#!/usr/bin/env python3
"""omega-K on the HLS backend at ALOS-1 scene geometry.

Emits the full artifact set: a synthesizable design at the 16384 x 16384
raster the San Francisco example processes on the CPU, and a
C-simulation package at a smaller raster with the same radar parameters.
`examples/common/hls_artifacts.py` explains why the two cannot be one
design.

The Stolt interpolation keeps its band weights on chip alongside the
frequency tables, so the streamed set is the planes themselves.

No raw data is read: the design is a function of the geometry, not of
the samples.

Usage:
    python run_alos_hls.py [--n 16384] [--csim-n 1024]
                           [--output PATH] [--no-testbench]
"""

import argparse
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_EXAMPLES), str(_EXAMPLES.parent / "python")]

from common.hls_artifacts import emit_alos_artifacts  # noqa: E402
from wka.algorithm import build_kernel, make_inputs  # noqa: E402
from wka.reference import WKAProcessor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",
                        type=int,
                        default=16384,
                        help="raster of the synthesizable design")
    parser.add_argument("--csim-n",
                        type=int,
                        default=1024,
                        help="raster of the C-simulation package")
    parser.add_argument("--output",
                        default=str(_EXAMPLES / "hls_project" / "wka_alos"),
                        help="artifact directory")
    parser.add_argument("--no-testbench",
                        action="store_true",
                        help="emit the designs only")
    args = parser.parse_args()

    emit_alos_artifacts("wka",
                        build_kernel,
                        make_inputs,
                        WKAProcessor,
                        n=args.n,
                        csim_n=args.csim_n,
                        out=Path(args.output).resolve(),
                        testbench=not args.no_testbench)
    print("done: csim with `vitis_hls -f wka_alos_csim.tcl`")


if __name__ == "__main__":
    main()
