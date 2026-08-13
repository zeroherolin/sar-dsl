#!/usr/bin/env python3
"""omega-K on the CPU backend with the real ALOS-1 San Francisco dataset.

Prerequisites: extract the raw echoes once with
`python ../data/extract_alos.py` (produces `alos_raw_16384x16384.bin`,
complex64, 2 GiB). Running needs tens of GiB of RAM at full size.

Usage:
    python run_alos.py [--bin ../data/alos_raw_16384x16384.bin]
"""

import argparse
import sys
import time
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_EXAMPLES), str(_EXAMPLES.parent / "python")]

from common.alos import SIZE, load_raw, save_scene   # noqa: E402
from common.params import ALOS_PARAMS                # noqa: E402
from wka.algorithm import build_kernel, make_inputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin",
                        default=str(_EXAMPLES / "data"
                                    / "alos_raw_16384x16384.bin"))
    parser.add_argument("--output", default="san_francisco_wka.png")
    args = parser.parse_args()

    print(f"[1/4] Loading {args.bin} ...")
    raw = load_raw(args.bin)

    print("[2/4] Compiling the WKA kernel ...")
    kernel = build_kernel(SIZE, ALOS_PARAMS).compile("cpu")

    print("[3/4] Running WKA imaging (this is the heavy part) ...")
    t0 = time.time()
    image = kernel(raw, *make_inputs(SIZE, ALOS_PARAMS))
    print(f"      focused in {time.time() - t0:.1f} s")
    del raw

    output = Path(args.output).resolve()
    print(f"[4/4] Post-processing and saving {output} ...")
    save_scene(image, ALOS_PARAMS, str(output),
               "San Francisco Bay (ALOS-1, SAR-DSL omega-K)")
    print("done.")


if __name__ == "__main__":
    main()
