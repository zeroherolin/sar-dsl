#!/usr/bin/env python3
"""CPU timing benchmark: SAR-DSL kernels vs the NumPy references.

Times compilation and execution of the complete imaging chains.  `--numpy`
also times the reference implementations (slow at large sizes: their
interpolation loops are per-line Python).

Throughput is N*N / run_best (input samples per second), measured as the
minimum of 5 timed runs after 3 warmup passes with `time.perf_counter`.
Points whose cold run exceeds 120 s are flagged and skipped before repetition.

Cold and warm are reported separately.  `cold` is the first call after
compilation, which faults in and zeroes the kernel's planes; `run best`
is the warm steady state, where the runtime's plane pool hands the same
mapped pages back and that cost is gone.  The pool lives for the life of
the process, so a cold number is only cold in a fresh interpreter that
has not yet run that chain at that size -- drive one point per process
(`--algs X --sizes N --json out.jsonl`) when the cold column matters.

Saves a log-log CPU throughput-vs-N figure to
`benchmarks/assets/cpu_throughput.png`
(all four chains on one plot; the fitted log-log slope is annotated per line).

For PFA, `size` is the polar-grid edge; the focused image is 2x
oversampled (`2n x 2n`).

Usage:
    python benchmarks/run_cpu_performance.py [--algs wka rda csa pfa]
                                         [--sizes 128 256 512 1024 2048 4096
                                                  8192 16384]
                                         [--repeats 5] [--numpy] [--no-figure]
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms import ALL, LABELS, load  # noqa: E402
from provenance import check_result_preconditions  # noqa: E402
from provenance import result_environment  # noqa: E402

_DEFAULT_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
_WARMUP = 3
_SLOW_POINT_S = 120.0

ASSETS = Path(__file__).resolve().parent / "assets"
_MAX_INPUT_SIZE = {"pfa": 8192}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _timing_statistics(times: list) -> dict:
    """Robust and legacy statistics for a list of wall-clock timings."""
    ordered = sorted(times)
    rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "best": min(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "p95": ordered[rank],
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "samples": times,
    }


def _timed(fn, repeats: int):
    """Returns robust and legacy statistics for back-to-back calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return _timing_statistics(times)


def throughput_figure(results: list, out_dir=None) -> None:
    """Saves a log-log throughput-vs-N figure.

    `results` is the list of dicts produced by `main` (keys: name, n,
    throughput, skipped).  Pass a `pathlib.Path` as `out_dir` to redirect
    output (e.g. a pytest `tmp_path`) and suppress writing to assets/.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import LogFormatterMathtext

    from common.plot import PALETTE, apply_style

    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for idx, name in enumerate(ALL):
        pts = sorted(
            [r for r in results if r["name"] == name and not r["skipped"]],
            key=lambda r: r["n"],
        )
        if not pts:
            continue
        ns = np.array([p["n"] for p in pts], dtype=float)
        ts = np.array([p["throughput"] for p in pts])
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(ns,
                ts,
                "o-",
                color=color,
                linewidth=1.6,
                markersize=4,
                label=LABELS[name])

        # Annotate the log-log slope when at least two points exist.
        if len(ns) >= 2:
            slope = float(np.polyfit(np.log10(ns), np.log10(ts), 1)[0])
            ax.annotate(
                f"  {slope:+.2f}",
                xy=(ns[-1], ts[-1]),
                fontsize=7.5,
                color=color,
                va="center",
            )

    skipped = [r["n"] for r in results if r["skipped"]]
    if skipped:
        ax.axvline(min(skipped),
                   color="0.6",
                   linewidth=0.8,
                   linestyle=":",
                   label=f"slow point (>{_SLOW_POINT_S:.0f} s)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    all_ns = [r["n"] for r in results if not r["skipped"]]
    if all_ns:
        lo = int(np.floor(np.log2(min(all_ns))))
        hi = int(np.ceil(np.log2(max(all_ns))))
        ax.set_xticks([2**power for power in range(lo, hi + 1)])
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2))
    ax.set_xlabel("Input edge N  (NxN samples)")
    ax.set_ylabel("Throughput  (samples / s)")
    ax.set_title(
        "CPU throughput vs scene size  [slope: log-log fit per algorithm]")
    ax.legend(fontsize=8.5)

    dest = Path(out_dir) if out_dir is not None else ASSETS
    dest.mkdir(exist_ok=True)
    out = dest / "cpu_throughput.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algs", nargs="+", choices=ALL, default=list(ALL))
    parser.add_argument("--sizes",
                        type=_positive_int,
                        nargs="+",
                        default=_DEFAULT_SIZES)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--numpy",
                        action="store_true",
                        help="also time the NumPy reference implementation")
    parser.add_argument("--no-figure",
                        action="store_true",
                        help="skip writing cpu_throughput.png")
    parser.add_argument("--json",
                        help="also append the results as JSON lines to this "
                        "file (one object per measured point)")
    parser.add_argument("--allow-dirty",
                        action="store_true",
                        help="permit JSON output from a dirty worktree and "
                        "record its diff hash")
    args = parser.parse_args()
    if args.json:
        check_result_preconditions(args.allow_dirty)

    print(f"{'algorithm':>9} {'size':>6} {'compile':>9} {'cold':>9} "
          f"{'run best':>10} {'run mean':>10} {'Msamp/s':>9} {'numpy':>10} "
          f"{'speedup':>8}")

    results = []
    for name in args.algs:
        for n in args.sizes:
            if n > _MAX_INPUT_SIZE.get(name, n):
                print(f"{name:>9} {n:>6} {'-':>9} {'-':>9} "
                      f"{'UNSUPPORTED':>10} {'-':>10} {'-':>9} "
                      f"{'-':>10} {'-':>8}")
                continue
            chain = load(name, n)

            t0 = time.perf_counter()
            kernel = chain.compile_kernel()
            compile_s = time.perf_counter() - t0

            # First call: pays the page-fault and zeroing cost for planes
            # the plane pool has not handed out yet.  Only genuinely cold
            # when this process has not run this chain at this size before
            # -- the pool is per process, so a meaningful cold number needs
            # a fresh interpreter per point.
            t0 = time.perf_counter()
            chain.run(kernel)
            cold_s = time.perf_counter() - t0

            if cold_s > _SLOW_POINT_S:
                note = f"(>{_SLOW_POINT_S:.0f}s)"
                print(f"{name:>9} {n:>6} {compile_s:8.2f}s {cold_s:8.2f}s "
                      f"{'SKIP':>10} {note:>10} {'':>9} {'':>10} {'':>8}")
                results.append(
                    dict(name=name,
                         n=n,
                         throughput=0.0,
                         skipped=True,
                         compile_s=compile_s,
                         cold_s=cold_s))
                continue

            for _ in range(_WARMUP - 1):
                chain.run(kernel)

            timing = _timed(lambda: chain.run(kernel), args.repeats)
            best, mean = timing["best"], timing["mean"]
            throughput = n * n / best

            np_best = None
            if args.numpy:
                numpy_timing = _timed(chain.run_reference,
                                      max(1, args.repeats // 3))
                np_best = numpy_timing["best"]
                np_txt = f"{np_best:9.2f}s"
                speedup = f"{np_best / best:7.1f}x"
            else:
                np_txt, speedup = "-", "-"

            print(f"{name:>9} {n:>6} {compile_s:8.2f}s {cold_s:8.3f}s "
                  f"{best:9.3f}s {mean:9.3f}s {throughput / 1e6:9.2f} "
                  f"{np_txt:>10} {speedup:>8}")
            results.append(
                dict(name=name,
                     n=n,
                     throughput=throughput,
                     skipped=False,
                     compile_s=compile_s,
                     cold_s=cold_s,
                     best_s=best,
                     mean_s=mean,
                     median_s=timing["median"],
                     p95_s=timing["p95"],
                     stdev_s=timing["stdev"],
                     samples_s=timing["samples"],
                     numpy_s=np_best))
            if args.numpy:
                results[-1]["numpy_samples_s"] = numpy_timing["samples"]

    if args.json:
        with open(args.json, "a") as fh:
            fh.write(
                json.dumps({
                    "type": "environment",
                    **result_environment(args.allow_dirty)
                }) + "\n")
            for r in results:
                fh.write(json.dumps({"type": "measurement", **r}) + "\n")

    if not args.no_figure:
        throughput_figure(results)


if __name__ == "__main__":
    main()
