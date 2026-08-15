#!/usr/bin/env python3
"""omega-K on the ScaleHLS backend, end to end: trace the complete kernel,
emit a Vitis HLS C++ design and a C-simulation package with golden data
from the NumPy reference (requires `make scalehls`).

The generated top function takes the raw data as two float planes
(re, im), followed by the window vectors, then the output magnitude
plane.

Usage:
    python run_point_target_scalehls.py [--n 256] [--output hls_project/wka]
                           [--no-testbench]
"""

import argparse
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_EXAMPLES), str(_EXAMPLES.parent / "python")]

from common.params import synthetic_params  # noqa: E402
from common.simulate import demo_scene  # noqa: E402
from wka.algorithm import build_kernel, make_inputs  # noqa: E402
from wka.reference import WKAProcessor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",
                        type=int,
                        default=256,
                        help="raster size (power of two)")
    parser.add_argument("--output",
                        default="hls_project/wka",
                        help="C-simulation package directory")
    parser.add_argument("--no-testbench",
                        action="store_true",
                        help="emit the design only")
    args = parser.parse_args()

    n = args.n
    params = synthetic_params(n)
    print(f"[1/2] Emitting the {n}x{n} WKA kernel through ScaleHLS-HIDA ...")
    design = build_kernel(n, params).compile(backend="scalehls")

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.no_testbench:
        (out / "wka.cpp").write_text(design.source())
    else:
        raw, _ = demo_scene(n, params)
        golden = WKAProcessor(n, params).process(raw)
        design.write_testbench([raw, *make_inputs(n, params)], [golden], out)
    print(f"[2/2] Saved {out}")
    print(f"done: {design.source().count(chr(10))} lines of HLS C++ "
          f"(flow={design.flow}, top function 'wka'); "
          "csim: cd {out} && vitis_hls -f wka_csim.tcl".format(out=out))


if __name__ == "__main__":
    main()
