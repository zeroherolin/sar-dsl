#!/usr/bin/env python3
"""Chirp Scaling on the CPU backend with the real ALOS-1 dataset.

Prerequisites: extract the raw echoes once with
`python ../data/extract_alos.py`. CSA operates directly on the
uncompressed chirp (range compression happens inside the algorithm), so
this is a full raw-to-image run without interpolation.

Usage:
    python run_alos_cpu.py [--bin ../data/alos_raw_16384x16384.bin]
"""

import argparse
import sys
import time
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_EXAMPLES), str(_EXAMPLES.parent / "python")]

from common.alos import SIZE, load_raw, save_scene  # noqa: E402
from common.params import ALOS_PARAMS  # noqa: E402
from csa.algorithm import build_kernel, make_inputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin",
                        default=str(_EXAMPLES / "data" /
                                    "alos_raw_16384x16384.bin"))
    parser.add_argument("--output", default="san_francisco_csa.png")
    args = parser.parse_args()

    print(f"[1/4] Loading {args.bin} ...")
    raw = load_raw(args.bin)

    print("[2/4] Compiling the CSA kernel ...")
    kernel = build_kernel(SIZE, ALOS_PARAMS).compile("cpu")

    print("[3/4] Running CSA imaging ...")
    t0 = time.time()
    image = kernel(raw, *make_inputs(SIZE, ALOS_PARAMS))
    print(f"      focused in {time.time() - t0:.1f} s")
    del raw

    output = Path(args.output).resolve()
    print(f"[4/4] Post-processing and saving {output} ...")
    save_scene(image, ALOS_PARAMS, str(output),
               "San Francisco Bay (ALOS-1, SAR-DSL Chirp Scaling)")
    print("done.")


if __name__ == "__main__":
    main()
