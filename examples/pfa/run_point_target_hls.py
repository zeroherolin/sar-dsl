#!/usr/bin/env python3
"""PFA on the HLS backend: emit a Vitis HLS C++ design and a
validation package with golden data from the NumPy reference.

The Python-defined operators inline at trace time, so the emitted design
is the same single dataflow module the built-in algorithms produce (the
in-kernel geometry math lowers through decomplexify/affine like any
other op chain).

Polar format resamples onto a raster twice the input on each axis, so a
given `--n` holds four times the plane bytes the stripmap chains do. The
interface follows the HLS configuration and the testbench drives its physical
port schema.

Usage:
    python run_point_target_hls.py [--n 64] [--output PATH]
                           [--no-testbench]
"""

import argparse
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_EXAMPLES), str(_EXAMPLES.parent / "python")]

from pfa.algorithm import build_kernel, make_inputs  # noqa: E402
from pfa.geometry import Geometry  # noqa: E402
from pfa.reference import PFAProcessor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",
                        type=int,
                        default=64,
                        help="input raster size (power of two); the "
                        "resampled raster is twice this on each axis")
    parser.add_argument("--output",
                        default=str(_EXAMPLES / "hls_project" / "pfa"),
                        help="HLS validation package directory")
    parser.add_argument("--no-testbench",
                        action="store_true",
                        help="emit the design only")
    args = parser.parse_args()

    n = args.n
    geometry = Geometry(n)
    print(f"[1/2] Tracing and lowering the {n}x{n} PFA kernel ...")
    design = build_kernel(n, geometry).compile(backend="hls")

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.no_testbench:
        (out / "pfa.cpp").write_text(design.source())
    else:
        raw = geometry.simulate(geometry.demo_targets())
        golden = PFAProcessor(n, geometry).process(raw)
        design.write_testbench([raw, *make_inputs(n, geometry)], list(golden),
                               out)
    print(f"[2/2] Saved {out}")
    print(f"done: {design.source().count(chr(10))} lines of HLS C++, "
          f"top function 'pfa'")
    print(f"      C-sim through Vitis HLS: cd {out} && "
          "vitis_hls -f pfa_hls_csim.tcl")
    print("      without Vitis: sh pfa_portable_cpp_sim.sh")


if __name__ == "__main__":
    main()
