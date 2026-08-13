#!/usr/bin/env python3
"""WKA imaging benchmark: SAR-DSL CPU backend vs the NumPy reference.

Usage:
    python benchmarks/bench_wka.py [--sizes 1024 4096] [--repeats 3] [--numpy]

The NumPy comparison is optional because the reference Stolt loop is slow at
large sizes.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "examples"))

from common.params import synthetic_params            # noqa: E402
from common.simulate import demo_scene                 # noqa: E402
from wka.algorithm import build_kernel, make_inputs    # noqa: E402
from wka.reference import WKAProcessor                 # noqa: E402


def bench(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.mean(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[1024, 4096])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--numpy", action="store_true",
                        help="also time the NumPy reference implementation")
    args = parser.parse_args()

    print(f"{'size':>8} {'compile':>9} {'run best':>10} {'run mean':>10} "
          f"{'numpy':>10} {'speedup':>8}")
    for n in args.sizes:
        params = synthetic_params(n)
        raw, _ = demo_scene(n, params)
        fa, fr, wr, wa = make_inputs(n, params)

        t0 = time.perf_counter()
        kernel = build_kernel(n, params).compile("cpu")
        compile_s = time.perf_counter() - t0

        best, mean = bench(lambda: kernel(raw, fa, fr, wr, wa),
                           args.repeats)

        if args.numpy:
            processor = WKAProcessor(n, params)
            np_best, _ = bench(lambda: processor.process(raw),
                               max(1, args.repeats // 3))
            np_txt, speedup = f"{np_best:9.2f}s", f"{np_best / best:7.1f}x"
        else:
            np_txt, speedup = "-", "-"

        print(f"{n:>8} {compile_s:8.2f}s {best:9.3f}s {mean:9.3f}s "
              f"{np_txt:>10} {speedup:>8}")


if __name__ == "__main__":
    main()
