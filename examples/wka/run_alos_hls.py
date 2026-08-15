#!/usr/bin/env python3
"""omega-K on the ScaleHLS backend at the full ALOS-1 scene size.

Emits a synthesizable Vitis HLS C++ design for the 16384 x 16384 raster
the San Francisco example processes on the CPU. Unlike the point-target
runner this one compiles with `axi_interface=True`, which is what makes
the size reachable: the raw echoes, the output and the full-size
intermediates become AXI master ports backed by DRAM, leaving only the
constant tables and the one-line transform scratch on chip. Intermediates
whose lifetimes do not overlap share a plane, and ports of the same
element type share one bundle, so the design presents a dozen ports on a
couple of interfaces rather than one per intermediate.

This runner emits the design only. A C-simulation package needs golden
data for every top-level port, and the promoted intermediate ports have
none; use `run_point_target_hls.py` for a csim-able package, and
validate the arithmetic there.

No raw data is read: the design is a function of the geometry, not of the
samples.

Usage:
    python run_alos_hls.py [--n 16384] [--output hls_project/wka_alos]
"""

import argparse
import sys
import time
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_EXAMPLES), str(_EXAMPLES.parent / "python")]

from common.params import ALOS_PARAMS  # noqa: E402
from wka.algorithm import build_kernel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",
                        type=int,
                        default=16384,
                        help="raster size (power of two)")
    parser.add_argument("--output",
                        default="hls_project/wka_alos",
                        help="directory for the emitted design")
    args = parser.parse_args()

    n = args.n
    print(f"[1/2] Emitting the {n}x{n} WKA kernel through HLS ...")
    started = time.time()
    design = build_kernel(n,
                          ALOS_PARAMS).compile(backend="hls",
                                               options={"axi_interface": True})
    source = design.source()
    print(f"      emitted in {time.time() - started:.1f} s")

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "wka_alos.cpp"
    target.write_text(source)

    ports = source.count("m_axi")
    bundles = len({
        line.split("bundle=")[1].split()[0]
        for line in source.splitlines() if "m_axi" in line
    })
    print(f"[2/2] Saved {target}")
    print(f"done: {source.count(chr(10))} lines of HLS C++ "
          f"(flow=affine, top function 'wka'); "
          f"{ports} AXI ports on {bundles} bundle(s)")


if __name__ == "__main__":
    main()
