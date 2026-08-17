"""Smoke tests for the benchmark runners.

Runs every runner at a tiny size to catch import errors, API drift and
broken cross-module references before the long benchmark jobs do. The
figure helpers are exercised through `tmp_path`, so nothing lands in
`benchmarks/assets/`.
"""

import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, requires_cpu, requires_hls

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import run_accuracy  # noqa: E402
import run_figures  # noqa: E402
import run_performance  # noqa: E402
import run_precision  # noqa: E402
import run_quality  # noqa: E402
import run_resources  # noqa: E402
from algorithms import ALL, STRIPMAP, load  # noqa: E402

N = 32


def test_runner_modules_expose_main():
    """Every runner imports cleanly and exposes a `main` entry point."""
    for module in (run_accuracy, run_quality, run_figures):
        assert callable(module.main), module.__name__


@pytest.mark.parametrize("script,flag", [
    ("run_accuracy.py", "--algs"),
    ("run_quality.py", "--algs"),
    ("run_figures.py", "--sva-n"),
])
def test_runner_cli_help(script, flag):
    """argparse smoke: `--help` must exit 0 and document the script's
    flags -- without running the heavy benchmark flow itself."""
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable,
         str(REPO_ROOT / "benchmarks" / script), "--help"],
        capture_output=True,
        text=True,
        env=env)
    assert result.returncode == 0, result.stderr
    assert flag in result.stdout
    assert "--n" in result.stdout


@requires_cpu
@pytest.mark.parametrize("name", ALL)
def test_chain_compiles_and_runs(name):
    chain = load(name, N)
    kernel = chain.compile_kernel()
    assert chain.run(kernel) is not None


@pytest.mark.parametrize("name", ALL)
def test_chain_exposes_its_kernel_arguments(name):
    """`Chain.args` must be the very list `run` passes to the kernel.

    `run_accuracy` drives the same kernel through an HLS testbench, and
    rebuilding the inputs there instead of reusing these would let the
    two legs drift apart silently.
    """
    chain = load(name, N)
    assert chain.args is not None
    kernel = chain.compile_kernel()
    assert len(chain.args) == len(kernel.arg_types)


def test_throughput_figure_writes_to_tmp(tmp_path):
    results = [
        dict(name=name, n=n, throughput=float(n * n), skipped=False)
        for name in ALL for n in (16, 32, 64)
    ]
    run_performance.throughput_figure(results, out_dir=tmp_path)
    assert (tmp_path / "throughput.png").exists()


def test_throughput_figure_handles_skipped_points(tmp_path):
    """A timed-out point must not break the figure or the slope fit."""
    results = [
        dict(name="wka", n=16, throughput=256.0, skipped=False),
        dict(name="wka", n=32, throughput=0.0, skipped=True),
    ]
    run_performance.throughput_figure(results, out_dir=tmp_path)
    assert (tmp_path / "throughput.png").exists()


@requires_hls
def test_resource_measure_reports_ports():
    chain = load("wka", N)
    design = chain.compile_kernel(backend="hls",
                                  axi_interface=True,
                                  on_chip_budget=1 << 20)
    stats = run_resources.measure(design.source(), design.name)
    assert stats["lines"] > 0
    assert stats["dram_mib"] >= 0.0


@requires_hls
def test_budget_sweep_and_figure(tmp_path):
    results = run_resources.sweep(["wka"], N, steps=2)
    assert len(results) == 3
    # The ladder must start at all-DRAM and end at the full working set.
    assert results[0]["budget"] == 0
    assert results[-1]["budget"] == results[-1]["full"]
    run_resources.budget_figure(results, N, out_dir=tmp_path)
    assert (tmp_path / "budget_sweep.png").exists()


@pytest.mark.parametrize("name", ALL)
def test_f32_build_parses_as_valid_ir(name):
    """Every chain built with `dtype=sar.c64` must produce IR the C++
    verifiers accept -- the option threads precision through annotations,
    casts and result types, and a missed spot surfaces here."""
    from sar.compiler.toolchain import find_tool

    ir = run_precision.narrow_ir(name)
    assert "f32" in ir
    result = subprocess.run([find_tool("sar-opt"), "-"],
                            input=ir,
                            capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", STRIPMAP)
def test_f32_build_narrows_the_kernel_boundary(name):
    """A stripmap chain built at f32 declares no double-precision planes:
    the echo stays c64 (no widening cast) and the window/axis vectors
    follow the data down."""
    kernel, mode = run_precision._build(name, 64, single=True)
    assert mode == "c64+f32"
    for t in kernel.arg_types + kernel.declared_result_types:
        assert t.dtype.name not in ("c128", "f64"), t


def test_dtype_must_be_a_complex_spec():
    """`build_kernel` rejects anything but sar.c128/sar.c64 up front."""
    import importlib

    import sar
    from common.params import synthetic_params

    algorithm = importlib.import_module("wka.algorithm")
    with pytest.raises(ValueError, match="c128 or sar.c64"):
        algorithm.build_kernel(64, synthetic_params(64), dtype=sar.f32)


@pytest.mark.parametrize("name", ALL)
def test_narrowed_ir_carries_single_precision_planes(name):
    """Narrowing must reach the complex planes, not just the annotations.

    Counts 2-D complex plane types in the narrowed IR; at least one must
    have come out as c64, otherwise the build parsed but bought nothing.
    """
    counts = run_precision.residual_double_planes(
        run_precision.narrow_ir(name))
    assert counts["complex<f32>"] > 0


def test_pfa_keeps_its_interpolation_axes_double():
    """PFA's collection axes stay f64 whatever the dtype: they feed
    `interp1d` positions, which the language requires to be double. The
    mode label says so, and the position planes in the narrowed IR are
    still f64."""
    kernel, mode = run_precision._build("pfa", 64, single=True)
    assert mode == "c64 only"
    counts = run_precision.residual_double_planes(kernel.to_mlir())
    assert counts["f64"] > 0  # the interpolation position planes
