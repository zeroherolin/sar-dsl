"""Smoke tests for the benchmark runners.

Runs every runner at a tiny size to catch import errors, API drift and
broken cross-module references before the long benchmark jobs do. The
figure helpers are exercised through `tmp_path`, so nothing lands in
`benchmarks/assets/`.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import sar
from sar.backends.hls import HLSConfig
from conftest import REPO_ROOT, requires_cpu, requires_hls

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import run_accuracy  # noqa: E402
import run_figures  # noqa: E402
import run_performance  # noqa: E402
import run_precision  # noqa: E402
import run_quality  # noqa: E402
import run_resources  # noqa: E402
import provenance  # noqa: E402
import metrics  # noqa: E402
from hls_reports import parse_csynth_xml, validate_constraints  # noqa: E402
from algorithms import ALL, STRIPMAP, load  # noqa: E402

N = 32


def test_provenance_handles_missing_git(monkeypatch):

    def unavailable(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(provenance.subprocess, "run", unavailable)
    data = provenance.environment()
    assert data["git_commit"] is None
    assert data["git_dirty"] is None


def test_accuracy_rejects_result_count_mismatch():
    with pytest.raises(ValueError, match="result count mismatch"):
        run_accuracy._max_abs((object(), ), ())


@pytest.mark.parametrize("parser", [
    run_performance._positive_int,
    run_resources._positive_int,
])
def test_benchmark_positive_integer_validation(parser):
    assert parser("3") == 3
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        parser("0")


def test_quality_metrics_reject_degenerate_inputs():
    with pytest.raises(ValueError, match="finite nonzero"):
        metrics.measure_cut(np.zeros(8))
    with pytest.raises(ValueError, match="non-empty 2-D"):
        metrics.measure_image(np.zeros(8))
    with pytest.raises(ValueError, match="at least tile"):
        metrics.urban_contrast(np.ones((8, 8)), tile=16)


def test_checked_in_hls_results_are_self_consistent():
    paths = sorted((REPO_ROOT / "benchmarks" / "results").glob("*.json"))
    assert paths
    for path in paths:
        data = json.loads(path.read_text())
        assert data["schema_version"] == 1
        constraints = data["constraints"]
        if "report" in data:
            report = data["report"]
            assert report["timing_constraint_met"] == (
                report["estimated_clock_ns"] <= constraints["clock_ns"])
            assert report["bram18k"] <= constraints["bram18k_budget"]
            assert report["uram"] <= constraints["uram_budget"]
            assert report["dsp"] <= constraints["dsp_budget"]
            assert len(report["report_sha256"]) == 64
        else:
            assert len(data["designs"]) == len(ALL)
            for design in data["designs"]:
                assert design["estimated_clock_ns"] <= constraints["clock_ns"]
                if "vitis_csim_passed" in design:
                    assert design["vitis_csim_passed"]
                if "source_sha256" in design:
                    assert len(design["source_sha256"]) == 64
                assert len(design["report_sha256"]) == 64


def test_runner_modules_expose_main():
    """Every runner imports cleanly and exposes a `main` entry point."""
    for module in (run_accuracy, run_quality, run_figures, run_performance,
                   run_precision, run_resources):
        assert callable(module.main), module.__name__


@pytest.mark.parametrize("script,flag", [
    ("run_accuracy.py", "--algs"),
    ("run_quality.py", "--algs"),
    ("run_figures.py", "--sva-n"),
    ("run_performance.py", "--sizes"),
    ("run_precision.py", "--algs"),
    ("run_resources.py", "--budget-sweep"),
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
    assert len(chain.args) == len(chain.kernel.arg_types)


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
                                  interface="axi",
                                  bram_bytes=1 << 22,
                                  uram_bytes=1 << 22,
                                  lutram_bytes=0)
    stats = run_resources.measure(design.source(), design.name)
    assert stats["lines"] > 0
    assert stats["external_footprint_mib"] >= 0.0
    assert stats["constant_rom_kib"] >= 0.0


def test_resource_measure_counts_vector_ports_and_constant_rom():
    source = """\
static const float table[16] = {0};
void vector_top(
  hls::vector<float, 16> in0[8],
  hls::vector<float, 16> out0[8]
) {
  #pragma HLS interface m_axi port=in0 bundle=gmem0
  #pragma HLS interface m_axi port=out0 bundle=gmem1
  float scratch[32];
}
"""
    stats = run_resources.measure(source, "vector_top")
    assert stats["external_footprint_mib"] == 1024 / 2**20
    assert stats["mutable_on_chip_kib"] == 128 / 2**10
    assert stats["constant_rom_kib"] == 64 / 2**10


def test_resource_measure_rejects_format_drift():
    with pytest.raises(ValueError, match="cannot parse top function"):
        run_resources.measure("void other() {}", "expected")
    malformed = "void top(\n  int unsupported\n) {\n}\n"
    with pytest.raises(ValueError, match="top-level declaration"):
        run_resources.measure(malformed, "top")


def test_vitis_report_parser_checks_constraints(tmp_path):
    report = tmp_path / "probe_csynth.xml"
    report.write_text("""\
<profile>
  <ReportVersion><Version>2022.2</Version></ReportVersion>
  <UserAssignments>
    <Part>xcvu13p-fhgb2104-2-i</Part>
    <TopModelName>probe</TopModelName>
    <TargetClockPeriod>4.00</TargetClockPeriod>
  </UserAssignments>
  <PerformanceEstimates>
    <PipelineType>no</PipelineType>
    <SummaryOfTimingAnalysis>
      <EstimatedClockPeriod>3.10</EstimatedClockPeriod>
    </SummaryOfTimingAnalysis>
    <SummaryOfOverallLatency>
      <Worst-caseLatency>100</Worst-caseLatency>
      <Interval-max>20</Interval-max>
    </SummaryOfOverallLatency>
    <SummaryOfLoopLatency>
      <LOOP_1>
        <TripCount>8</TripCount><Latency>16</Latency>
        <IterationLatency>2</IterationLatency>
      </LOOP_1>
    </SummaryOfLoopLatency>
  </PerformanceEstimates>
  <AreaEstimates>
    <Resources>
      <BRAM_18K>10</BRAM_18K><DSP>20</DSP><FF>30</FF>
      <LUT>40</LUT><URAM>1</URAM>
    </Resources>
    <AvailableResources>
      <BRAM_18K>5376</BRAM_18K><DSP>12288</DSP><FF>3456000</FF>
      <LUT>1728000</LUT><URAM>1280</URAM>
    </AvailableResources>
  </AreaEstimates>
  <InterfaceSummary>
    <RtlPorts>
      <name>m_axi_gmem_RDATA</name><Object>gmem</Object>
      <IOProtocol>m_axi</IOProtocol><Bits>512</Bits>
    </RtlPorts>
    <RtlPorts>
      <name>m_axi_gmem_WDATA</name><Object>gmem</Object>
      <IOProtocol>m_axi</IOProtocol><Bits>512</Bits>
    </RtlPorts>
  </InterfaceSummary>
</profile>
""")
    parsed = parse_csynth_xml(report)
    assert parsed["estimated_clock_ns"] == 3.1
    assert parsed["loop_latencies"] == [{
        "name": "LOOP_1",
        "trip_count": 8,
        "latency_cycles": 16,
        "iteration_latency": 2,
    }]
    assert parsed["interfaces"][0]["read_data_bits"] == 512
    assert len(parsed["xml_sha256"]) == 64
    assert not validate_constraints(parsed, HLSConfig.resolve())


@requires_hls
def test_budget_sweep_and_figure(tmp_path):
    results = run_resources.sweep(["wka"], N, steps=2)
    assert len(results) >= 2
    assert len({row["budget"] for row in results}) == len(results)
    # The ladder starts no lower than the primitive floor and ends at the
    # full working set. Infeasible and duplicate points are omitted.
    assert results[0]["budget"] == 8 * 36864
    assert results[-1]["budget"] >= results[-1]["full"]
    run_resources.budget_figure(results, N, out_dir=tmp_path)
    assert (tmp_path / "budget_sweep.png").exists()


@requires_hls
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
    counts = run_precision.complex_type_references(
        run_precision.narrow_ir(name))
    assert counts["complex<f32>"] > 0


def test_pfa_keeps_its_interpolation_axes_double():
    """PFA's collection axes stay f64 whatever the dtype: they feed
    `interp1d` positions, which the language requires to be double. The
    mode label says so, and the position planes in the narrowed IR are
    still f64."""
    kernel, mode = run_precision._build("pfa", 64, single=True)
    assert mode == "c64 only"
    counts = run_precision.complex_type_references(kernel.to_mlir())
    assert counts["f64"] > 0  # the interpolation position planes
