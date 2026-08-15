#!/usr/bin/env python3
"""Timing benchmark: compiled SAR-DSL kernels vs the NumPy references.

Times compilation and execution of the complete imaging chains. `--numpy`
also times the reference implementations (slow at large sizes: their
interpolation loops are per-line Python).

For PFA, `size` is the polar-grid edge; the focused image is 2x
oversampled (`2n x 2n`).

Usage:
    python benchmarks/run_performance.py [--algs wka rda csa pfa]
                                         [--sizes 1024 4096]
                                         [--repeats 3] [--numpy]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms import ALL, load  # noqa: E402


def bench(fn, repeats: int):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.mean(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algs", nargs="+", choices=ALL, default=list(ALL))
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--numpy",
                        action="store_true",
                        help="also time the NumPy reference implementation")
    args = parser.parse_args()

    print(f"{'algorithm':>9} {'size':>6} {'compile':>9} {'run best':>10} "
          f"{'run mean':>10} {'numpy':>10} {'speedup':>8}")
    for name in args.algs:
        for n in args.sizes:
            chain = load(name, n)

            t0 = time.perf_counter()
            kernel = chain.compile_kernel()
            compile_s = time.perf_counter() - t0

            best, mean = bench(lambda: chain.run(kernel), args.repeats)

            if args.numpy:
                np_best, _ = bench(chain.run_reference,
                                   max(1, args.repeats // 3))
                np_txt = f"{np_best:9.2f}s"
                speedup = f"{np_best / best:7.1f}x"
            else:
                np_txt, speedup = "-", "-"

            print(f"{name:>9} {n:>6} {compile_s:8.2f}s {best:9.3f}s "
                  f"{mean:9.3f}s {np_txt:>10} {speedup:>8}")


if __name__ == "__main__":
    main()
