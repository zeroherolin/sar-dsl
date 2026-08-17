#!/usr/bin/env python3
"""Chirp Scaling on the HLS backend at ALOS-1 scene geometry.

Emits the full artifact set: a synthesizable design at the 16384 x 16384
raster the San Francisco example processes on the CPU, and a
C-simulation package at a smaller raster with the same radar parameters.
`examples/common/hls_artifacts.py` explains why the two cannot be one
design.

Being interpolation-free costs CSA nothing on chip but buys no traffic
back: the Doppler-dependent factors are recomputed into each consumer
rather than stored, so it is the phase-multiply chain that streams.

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
from csa.algorithm import build_kernel, make_inputs  # noqa: E402
from csa.reference import CSAProcessor  # noqa: E402


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
                        default=str(_EXAMPLES / "hls_project" / "csa_alos"),
                        help="artifact directory")
    parser.add_argument("--no-testbench",
                        action="store_true",
                        help="emit the designs only")
    args = parser.parse_args()

    emit_alos_artifacts("csa",
                        build_kernel,
                        make_inputs,
                        CSAProcessor,
                        n=args.n,
                        csim_n=args.csim_n,
                        out=Path(args.output).resolve(),
                        testbench=not args.no_testbench)
    print("done: csim with `vitis_hls -f csa_alos_csim.tcl`")


if __name__ == "__main__":
    main()
