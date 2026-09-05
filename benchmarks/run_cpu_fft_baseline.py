#!/usr/bin/env python3
"""Compare batched c128 FFT leaves: SAR runtime, NumPy and optional MKL."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from provenance import result_environment  # noqa: E402
from sar.compiler.toolchain import find_runtime_library  # noqa: E402
from sar.runtime import make_descriptor  # noqa: E402


def _time(fn, repeats, warmups=3):
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {
        "best_s": min(samples),
        "median_s": statistics.median(samples),
        "samples_s": samples,
    }


def _mkl_library(build_dir: Path):
    source = Path(__file__).with_name("mkl_fft_baseline.cpp")
    output = build_dir / "libmkl_fft_baseline.so"
    build_dir.mkdir(parents=True, exist_ok=True)
    if not output.is_file() or output.stat().st_mtime_ns < source.stat(
    ).st_mtime_ns:
        scratch = output.with_suffix(f".{os.getpid()}.tmp.so")
        subprocess.run([
            os.environ.get("CXX", "c++"), "-O3", "-shared", "-fPIC",
            str(source), "-I/usr/local/include",
            "/usr/local/lib/libmkl_rt.so.1", "-Wl,-rpath,/usr/local/lib", "-o",
            str(scratch)
        ],
                       check=True)
        os.replace(scratch, output)
    library = ctypes.CDLL(str(output))
    create = library.sar_mkl_plan_c128
    create.restype = ctypes.c_void_p
    create.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
    execute = library.sar_mkl_execute_c128
    execute.restype = ctypes.c_int
    execute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    free = library.sar_mkl_plan_free
    free.restype = None
    free.argtypes = [ctypes.c_void_p]
    return create, execute, free


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-mkl", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    # libsar_runtime reads this once when its shared library is loaded below.
    # NumPy thread control depends on its linked FFT implementation, so the
    # report names that leg as uncontrolled instead of implying otherwise.
    os.environ["SAR_RT_NUM_THREADS"] = str(args.threads)
    rng = np.random.default_rng(20260827)
    values = (rng.standard_normal(
        (args.rows, args.cols)) + 1j * rng.standard_normal(
            (args.rows, args.cols)))
    values = np.ascontiguousarray(values)
    expected = np.fft.fft(values, axis=1)
    runtime_out = np.empty_like(values)
    runtime = ctypes.CDLL(find_runtime_library())
    runtime_fn = runtime._mlir_ciface_sar_rt_fft_2d_c128
    runtime_fn.restype = None
    source_desc = make_descriptor(values)
    output_desc = make_descriptor(runtime_out)

    def run_runtime():
        runtime_fn(ctypes.byref(source_desc), ctypes.byref(output_desc),
                   ctypes.c_int64(1), ctypes.c_bool(False))

    results = {
        "shape": [args.rows, args.cols],
        "thread_control": {
            "sar_runtime": args.threads,
            "mkl": args.threads if not args.no_mkl else None,
            "numpy": "implementation default",
        },
        "numpy": _time(lambda: np.fft.fft(values, axis=1), args.repeats),
        "sar_runtime": _time(run_runtime, args.repeats),
    }
    np.testing.assert_allclose(runtime_out, expected, rtol=1e-12, atol=1e-12)
    if not args.no_mkl:
        mkl_out = np.empty_like(values)
        create, execute, free = _mkl_library(Path("/tmp/sar-dsl-mkl-baseline"))
        plan = create(args.rows, args.cols, args.threads)
        if not plan:
            raise RuntimeError("MKL DFTI plan creation failed")

        def run_mkl():
            status = execute(plan, values.ctypes.data, mkl_out.ctypes.data)
            if status:
                raise RuntimeError(f"MKL DFTI failed with status {status}")

        try:
            results["mkl"] = _time(run_mkl, args.repeats)
            np.testing.assert_allclose(mkl_out,
                                       expected,
                                       rtol=1e-12,
                                       atol=1e-12)
        finally:
            free(plan)
    print(json.dumps(results, indent=2))
    if args.json:
        results["environment"] = result_environment(args.allow_dirty)
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
