#!/usr/bin/env python3
"""Range-Doppler on the HLS backend at ALOS-1 scene geometry.

Emits one design and its matching HLS validation package at the requested
raster. The interface and other hardware settings come from the normal HLS
configuration chain; this runner does not override them.

The RCMC gather keeps its interpolation weights on chip alongside the
range reference, so the streamed set is the planes themselves.

No raw data is read: the design is a function of the geometry, not of
the samples.

Usage:
    python run_alos_hls.py [--n 16384] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_EXAMPLES), str(_EXAMPLES.parent / "python")]

from common.hls_artifacts import emit_alos_artifacts  # noqa: E402
from rda.algorithm import build_kernel, make_inputs  # noqa: E402
from rda.reference import RDAProcessor  # noqa: E402


def emit(n: int = 16384, output=None, options=None) -> dict:
    """Python API: emits the package, optionally with HLS compile options."""
    destination = (Path(output) if output is not None else _EXAMPLES /
                   "hls_project" / "rda_alos")
    return emit_alos_artifacts("rda",
                               build_kernel,
                               make_inputs,
                               RDAProcessor,
                               n=n,
                               out=destination,
                               options=options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",
                        type=int,
                        default=16384,
                        help="raster of the synthesizable design")
    parser.add_argument("--output",
                        default=str(_EXAMPLES / "hls_project" / "rda_alos"),
                        help="artifact directory")
    args = parser.parse_args()

    result = emit(n=args.n, output=Path(args.output).resolve())
    print("done: C-sim through Vitis HLS with "
          "`vitis_hls -f rda_alos_hls_csim.tcl`")
    print("      without Vitis: `sh rda_alos_portable_cpp_sim.sh`")
    print(f"      top={result['top']}, interface={result['interface']}, "
          f"n={result['n']}")


if __name__ == "__main__":
    main()
