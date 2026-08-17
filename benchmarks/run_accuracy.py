#!/usr/bin/env python3
"""Cross-backend accuracy: CPU and HLS csim against the same NumPy reference.

Focuses one scene per chain three ways -- the NumPy reference, the
compiled CPU kernel, and the emitted HLS design run through C simulation
-- and reports how far each backend lands from the reference.

The csim leg needs no Vitis: `write_testbench` ships header stubs that
stand in for the Vitis ones, so the package builds with any C++ compiler.
This runner compiles it with the in-tree clang++ and runs it, parsing the
`max |err|` the generated testbench prints per output port.

csim is a *functional* simulation of the emitted C++.  Its wall time says
nothing about FPGA throughput and is not reported as performance.

Usage:
    python benchmarks/run_accuracy.py [--n 128] [--algs wka rda csa pfa]
                                      [--dtype c128|c64] [--keep-dir DIR]

`--dtype c64` builds the chains single-precision; the reference stays
the f64 NumPy implementation, so the columns then show what f32 costs
on each backend rather than backend disagreement alone.
"""

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms import ALL, LABELS, load  # noqa: E402

import sar  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _as_tuple(x):
    """Normalizes one-or-many kernel results to a tuple of arrays."""
    if isinstance(x, (tuple, list)):
        return tuple(np.asarray(a, dtype=np.float64) for a in x)
    return (np.asarray(x, dtype=np.float64), )


def _max_abs(a, b) -> float:
    return float(max(np.abs(x - y).max() for x, y in zip(a, b)))


def _peak(a) -> float:
    return float(max(np.abs(x).max() for x in a))


def cpu_error(chain):
    """(max abs error, reference peak, warm wall time) for the CPU backend."""
    reference = _as_tuple(chain.run_reference())
    kernel = chain.compile_kernel()
    got = _as_tuple(chain.run(kernel))  # cold, also warms the pool
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        chain.run(kernel)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return _max_abs(got, reference), _peak(reference), best, reference


def csim_error(chain, name: str, reference, work: Path, single: bool):
    """(max abs error, csim wall time) from the emitted design under csim.

    Builds the package `write_testbench` emits with the in-tree clang++
    against the shipped stubs, runs it, and parses the per-port
    `max |err|` lines out of the testbench's own output.
    """
    from sar.compiler.toolchain import find_tool

    design = chain.compile_kernel(backend="hls")
    # The golden files must carry the kernel's declared result dtypes; the
    # reference itself is f64 numpy whatever the build precision.
    golden = [
        np.asarray(r, dtype=t.dtype.to_numpy())
        for r, t in zip(reference, chain.kernel.declared_result_types)
    ]
    out = work / name
    # An f32 build measured against the f64 reference carries single-
    # precision rounding scaled by the signal, so its tolerance follows
    # the peak; the default tolerances expect the near-bit-exact f64 case.
    tolerances = {"rtol": 1e-4, "atol": _peak(golden) * 1e-5} if single \
        else {}
    design.write_testbench(chain.args, golden, out, **tolerances)

    top = design.name
    binary = out / "csim"
    compile_cmd = [
        find_tool("clang++"),
        "-O2",
        "-I",
        str(out / "stubs"),
        str(out / f"{top}.cpp"),
        str(out / f"{top}_tb.cpp"),
        "-o",
        str(binary),
        "-pthread",
    ]
    built = subprocess.run(compile_cmd, capture_output=True, text=True)
    if built.returncode != 0:
        raise RuntimeError(f"csim build failed for {name}:\n{built.stderr}")

    t0 = time.perf_counter()
    run = subprocess.run(
        [str(binary), str(out / f"{top}_tb_data")],
        capture_output=True,
        text=True)
    elapsed = time.perf_counter() - t0
    errors = [
        float(m)
        for m in re.findall(r"max \|err\| = ([0-9.eE+-]+)", run.stdout)
    ]
    if not errors:
        raise RuntimeError(f"csim produced no error lines for {name}:\n"
                           f"{run.stdout}\n{run.stderr}")
    return max(errors), elapsed, "PASS" in run.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--algs", nargs="+", default=list(ALL), choices=ALL)
    parser.add_argument("--dtype",
                        choices=("c128", "c64"),
                        default="c128",
                        help="working precision the chains are built with")
    parser.add_argument("--keep-dir",
                        help="write csim packages here instead of a "
                        "temporary directory")
    args = parser.parse_args()
    dtype = sar.c128 if args.dtype == "c128" else sar.c64

    if args.keep_dir:
        work = Path(args.keep_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
        ctx = None
    else:
        ctx = tempfile.TemporaryDirectory()
        work = Path(ctx.name)

    print(f"scene size N={args.n}, dtype={args.dtype}  "
          "(PFA: polar-grid edge, image 2N x 2N)")
    print(f"{'chain':<14} {'CPU vs ref':>12} {'CPU rel':>10} "
          f"{'csim vs ref':>12} {'csim rel':>10} {'CPU warm/s':>11} "
          f"{'csim s*':>9} {'csim':>6}")
    print("-" * 92)

    for name in args.algs:
        chain = load(name, args.n, dtype=dtype)

        cpu_err, peak, cpu_s, reference = cpu_error(chain)
        try:
            sim_err, sim_s, passed = csim_error(chain, name, reference, work,
                                                dtype is sar.c64)
            sim_txt = f"{sim_err:12.3e}"
            sim_rel = f"{sim_err / peak:10.2e}"
            sim_s_txt = f"{sim_s:9.2f}"
            verdict = "PASS" if passed else "FAIL"
        except Exception as exc:  # noqa: BLE001
            sim_txt = f"{'n/a':>12}"
            sim_rel = f"{'n/a':>10}"
            sim_s_txt = f"{'n/a':>9}"
            verdict = "ERR"
            print(f"  ! {name}: {exc}", file=sys.stderr)

        print(f"{LABELS[name]:<14} {cpu_err:12.3e} {cpu_err / peak:10.2e} "
              f"{sim_txt} {sim_rel} {cpu_s:11.3f} {sim_s_txt} {verdict:>6}")

    print("\n* csim seconds are functional simulation of the emitted C++ on "
          "the host,\n  single-threaded and untimed by any hardware model -- "
          "NOT an FPGA figure.")
    if ctx is not None:
        ctx.cleanup()


if __name__ == "__main__":
    main()
