#!/usr/bin/env python3
"""Range-Doppler on the CPU backend with the real ALOS-1 dataset.

Prerequisites: extract the raw echoes once with
`python ../data/extract_alos.py`. Note the basic RDA here omits
secondary range compression, so range focus degrades slightly at the
Doppler band edges compared to omega-K / chirp scaling.

Usage:
    python run_alos_cpu.py [--bin ../data/alos_raw_16384x16384.bin]
"""

import argparse
import sys
import time
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_EXAMPLES), str(_EXAMPLES.parent / "python")]

from common.alos import SIZE, load_raw, save_scene  # noqa: E402
from common.params import alos_params  # noqa: E402
from common.quality import print_scene_contrast  # noqa: E402
from rda.algorithm import build_kernel, make_inputs  # noqa: E402

PARAMS = alos_params(SIZE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin",
                        default=str(_EXAMPLES / "data" /
                                    "alos_raw_16384x16384.bin"))
    parser.add_argument("--output",
                        default=str(_HERE / "assets" /
                                    "san_francisco_rda.png"))
    args = parser.parse_args()

    print(f"[1/4] Loading {args.bin} ...")
    raw = load_raw(args.bin)

    print("[2/4] Compiling the RDA kernel ...")
    kernel = build_kernel(SIZE, PARAMS).compile("cpu")

    print("[3/4] Running RDA imaging ...")
    t0 = time.time()
    image = kernel(raw, *make_inputs(SIZE, PARAMS))
    print(f"      focused in {time.time() - t0:.1f} s")
    print_scene_contrast(image)
    del raw

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[4/4] Post-processing and saving {output} ...")
    save_scene(image, PARAMS, str(output),
               "San Francisco Bay (ALOS-1, SAR-DSL Range-Doppler)")
    print("done.")


if __name__ == "__main__":
    main()
