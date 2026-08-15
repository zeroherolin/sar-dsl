#!/usr/bin/env python3
"""Chirp Scaling on the ScaleHLS backend: emit a Vitis HLS C++ design
and a C-simulation package with golden data from the NumPy reference


Usage:
    python run_point_target_hls.py [--n 256] [--output hls_project/csa]
                           [--no-testbench]
"""

import argparse
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_EXAMPLES), str(_EXAMPLES.parent / "python")]

from common.params import synthetic_params  # noqa: E402
from common.simulate import demo_scene  # noqa: E402
from csa.algorithm import build_kernel, make_inputs  # noqa: E402
from csa.reference import CSAProcessor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",
                        type=int,
                        default=256,
                        help="raster size (power of two)")
    parser.add_argument("--output",
                        default="hls_project/csa",
                        help="C-simulation package directory")
    parser.add_argument("--no-testbench",
                        action="store_true",
                        help="emit the design only")
    args = parser.parse_args()

    n = args.n
    params = synthetic_params(n)
    print(f"[1/2] Emitting the {n}x{n} CSA kernel through HLS ...")
    design = build_kernel(n, params).compile(backend="hls")

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.no_testbench:
        (out / "csa.cpp").write_text(design.source())
    else:
        raw, _ = demo_scene(n, params)
        golden = CSAProcessor(n, params).process(raw)
        design.write_testbench([raw, *make_inputs(n, params)], [golden], out)
    print(f"[2/2] Saved {out}")
    print(f"done: {design.source().count(chr(10))} lines of HLS C++ "
          f"(flow=affine, top function 'csa'); "
          "csim: cd {out} && vitis_hls -f csa_csim.tcl".format(out=out))


if __name__ == "__main__":
    main()
