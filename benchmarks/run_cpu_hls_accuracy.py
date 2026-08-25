#!/usr/bin/env python3
"""Cross-backend accuracy through Vitis or its portable C++ fallback.

Focuses one scene per chain through the NumPy reference, the compiled CPU
kernel, and the emitted HLS design. When `vitis_hls` is installed, the
generated design runs in C-sim through Vitis HLS. Otherwise the benchmark
invokes the explicitly named portable C++ fallback supplied in the package.

Usage:
    python benchmarks/run_cpu_hls_accuracy.py [--n 128]
        [--algs wka rda csa pfa] [--dtype c128|c64] [--keep-dir DIR]
        [--vitis-hls PATH]

`--dtype c64` builds the chains single-precision; the reference stays the
f64 NumPy implementation, so the columns show what f32 costs on each backend.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms import ALL, LABELS, load  # noqa: E402
from provenance import environment  # noqa: E402

import sar  # noqa: E402


def _as_tuple(value):
    """Normalizes one-or-many kernel results to a tuple of arrays."""
    if isinstance(value, (tuple, list)):
        return tuple(np.asarray(array, dtype=np.float64) for array in value)
    return (np.asarray(value, dtype=np.float64), )


def _max_abs(actual, expected) -> float:
    if not actual or len(actual) != len(expected):
        raise ValueError(f"result count mismatch: got {len(actual)}, expected "
                         f"{len(expected)}")
    return float(
        max(np.abs(lhs - rhs).max() for lhs, rhs in zip(actual, expected)))


def _peak(arrays) -> float:
    return float(max(np.abs(array).max() for array in arrays))


def cpu_error(chain):
    """Returns error, reference peak, warm time, and reference outputs."""
    reference = _as_tuple(chain.run_reference())
    kernel = chain.compile_kernel()
    got = _as_tuple(chain.run(kernel))
    best = None
    for _ in range(3):
        started = time.perf_counter()
        chain.run(kernel)
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)
    return _max_abs(got, reference), _peak(reference), best, reference


def hls_csim_error(chain,
                   name: str,
                   reference,
                   work: Path,
                   single: bool,
                   vitis_hls: str = "vitis_hls",
                   timeout: float = 1800.0):
    """Returns error, elapsed time, status, and selected simulation mode."""
    design = chain.compile_kernel(backend="hls", interface="axi")
    if len(reference) != len(chain.kernel.declared_result_types):
        raise ValueError(
            f"{name}: expected {len(chain.kernel.declared_result_types)} "
            f"results, got {len(reference)}")

    out = work / name
    tolerances = ({
        "rtol": 1e-4,
        "atol": _peak(reference) * 1e-5
    } if single else {})
    design.write_testbench(chain.args, reference, out, **tolerances)

    top = design.name
    vitis = shutil.which(vitis_hls)
    if vitis is not None:
        command = [vitis, f"{top}_hls_csim.tcl"]
        mode = "hls_csim"
    else:
        command = ["sh", f"{top}_portable_cpp_sim.sh"]
        mode = "portable_cpp_sim"

    started = time.perf_counter()
    run = subprocess.run(command,
                         cwd=out,
                         capture_output=True,
                         text=True,
                         timeout=timeout)
    elapsed = time.perf_counter() - started
    output = run.stdout + run.stderr
    log_name = ("hls_csim.log"
                if mode == "hls_csim" else "portable_cpp_sim.log")
    (out / log_name).write_text(output)
    if run.returncode != 0:
        raise RuntimeError(f"{mode} failed for {name}:\n{output[-4000:]}")

    errors = [
        float(match)
        for match in re.findall(r"max \|err\| = ([0-9.eE+-]+)", output)
    ]
    if not errors:
        raise RuntimeError(
            f"{mode} produced no error lines for {name}:\n{output[-4000:]}")
    passed = re.search(r"(?m)^PASS$", output) is not None
    if mode == "hls_csim":
        passed &= "CSim done with 0 errors" in output
    return max(errors), elapsed, passed, mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--algs", nargs="+", default=list(ALL), choices=ALL)
    parser.add_argument("--dtype",
                        choices=("c128", "c64"),
                        default="c128",
                        help="working precision the chains are built with")
    parser.add_argument("--keep-dir",
                        help="write validation packages here instead of a "
                        "temporary directory")
    parser.add_argument("--vitis-hls", default="vitis_hls")
    parser.add_argument("--vitis-timeout", type=float, default=1800.0)
    parser.add_argument("--json", help="write machine-readable results here")
    args = parser.parse_args()
    dtype = sar.c128 if args.dtype == "c128" else sar.c64

    if args.keep_dir:
        work = Path(args.keep_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
        context = None
    else:
        context = tempfile.TemporaryDirectory()
        work = Path(context.name)

    print(f"scene size N={args.n}, dtype={args.dtype}  "
          "(PFA: polar-grid edge, image 2N x 2N)")
    print(f"{'chain':<14} {'CPU vs ref':>12} {'CPU rel':>10} "
          f"{'C-sim err':>14} {'C-sim rel':>10} {'CPU warm/s':>11} "
          f"{'sim s':>9} {'mode':>22} {'result':>7}")
    print("-" * 116)

    results = []
    failed = False
    for name in args.algs:
        chain = load(name, args.n, dtype=dtype)
        cpu_err, peak, cpu_s, reference = cpu_error(chain)
        try:
            sim_err, sim_s, passed, mode = hls_csim_error(
                chain, name, reference, work, dtype is sar.c64, args.vitis_hls,
                args.vitis_timeout)
            failed |= not passed
            result = {
                "name": name,
                "cpu_abs_error": cpu_err,
                "cpu_relative_error": cpu_err / peak,
                "hls_csim_abs_error": sim_err,
                "hls_csim_relative_error": sim_err / peak,
                "cpu_warm_s": cpu_s,
                "hls_csim_s": sim_s,
                "hls_csim_mode": mode,
                "hls_csim_passed": passed,
            }
            sim_text = f"{sim_err:12.3e} {sim_err / peak:10.2e}"
            time_text = f"{sim_s:9.2f}"
            verdict = "PASS" if passed else "FAIL"
        except Exception as error:  # noqa: BLE001
            failed = True
            result = {
                "name": name,
                "cpu_abs_error": cpu_err,
                "cpu_relative_error": cpu_err / peak,
                "hls_csim_passed": False,
                "error": str(error),
            }
            sim_text = f"{'n/a':>12} {'n/a':>10}"
            time_text = f"{'n/a':>9}"
            mode = "error"
            verdict = "ERR"
            print(f"  ! {name}: {error}", file=sys.stderr)
        results.append(result)
        print(f"{LABELS[name]:<14} {cpu_err:12.3e} {cpu_err / peak:10.2e} "
              f"{sim_text} {cpu_s:11.3f} {time_text} {mode:>22} "
              f"{verdict:>7}")

    print("\nC-sim through Vitis HLS and portable_cpp_sim report host "
          "validation time, not FPGA performance.")
    if context is not None:
        context.cleanup()
    if args.json:
        payload = {
            "environment": environment(),
            "benchmark": "cross_backend_accuracy",
            "command": [sys.executable, *sys.argv],
            "scene_size": args.n,
            "dtype": args.dtype,
            "results": results,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
