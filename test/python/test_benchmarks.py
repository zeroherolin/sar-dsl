"""Benchmark runner, report parser, and recorded-result tests.

Runner integration uses a bounded fixture size to catch import errors, API
drift, and broken cross-module references without re-running the published
measurements. Figure helpers write through `tmp_path`, so tests do not modify
`benchmarks/assets/`.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

import sar
from sar.backends.hls import HLSConfig
from sar.backends.hls.autotune import AUTO_OPTIONS
from conftest import REPO_ROOT, requires_cpu, requires_hls

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import run_cpu_hls_accuracy  # noqa: E402
import plot_cpu_impulse_response  # noqa: E402
import plot_cpu_hls_results  # noqa: E402
import run_hls_sweep  # noqa: E402
import run_cpu_performance  # noqa: E402
import run_cpu_precision  # noqa: E402
import run_cpu_quality  # noqa: E402
import run_hls_resources  # noqa: E402
import provenance  # noqa: E402
import metrics  # noqa: E402
import importlib.util  # noqa: E402
from hls_reports import (
    parse_csynth_xml,
    parse_vitis_warnings,  # noqa: E402
    timing_shortfall)
from hls_reports import validate_constraints  # noqa: E402
from algorithms import ALL, STRIPMAP, load  # noqa: E402

N = 32


def test_generated_documentation_references_are_current():
    path = REPO_ROOT / "docs/check_references.py"
    spec = importlib.util.spec_from_file_location("check_references", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check() == []


def test_vitis_warning_parser_groups_repeated_codes(tmp_path):
    log = tmp_path / "vitis_hls.log"
    log.write_text("INFO: ordinary line\n"
                   "WARNING: [HLS 200-885] first port conflict\n"
                   "WARNING: [HLS 200-885] second port conflict\n"
                   "WARNING: [SYN 201-103] legalized name\n")
    assert parse_vitis_warnings(log) == [{
        "code":
        "HLS 200-885",
        "count":
        2,
        "examples": ["first port conflict", "second port conflict"],
    }, {
        "code": "SYN 201-103",
        "count": 1,
        "examples": ["legalized name"],
    }]


def test_provenance_handles_missing_git(monkeypatch):

    def unavailable(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(provenance.subprocess, "run", unavailable)
    data = provenance.environment()
    assert data["git_commit"] is None
    assert data["git_dirty"] is None
    assert data["git_diff_sha256"] is None


def test_result_provenance_refuses_dirty_tree_without_opt_in(monkeypatch):
    dirty = {
        "git_dirty": True,
        "git_diff_sha256": "a" * 64,
    }
    monkeypatch.setattr(provenance, "environment", lambda: dict(dirty))
    with pytest.raises(RuntimeError, match="dirty worktree"):
        provenance.result_environment()
    with pytest.raises(RuntimeError, match="dirty worktree"):
        provenance.check_result_preconditions()
    assert provenance.result_environment(allow_dirty=True) == dirty


def test_result_provenance_treats_unknown_git_state_as_dirty(monkeypatch):
    unknown = {"git_dirty": None, "git_diff_sha256": None}
    monkeypatch.setattr(provenance, "environment", lambda: dict(unknown))
    with pytest.raises(RuntimeError, match="cannot determine"):
        provenance.result_environment()
    with pytest.raises(RuntimeError, match="cannot determine"):
        provenance.check_result_preconditions()
    assert provenance.result_environment(allow_dirty=True) == unknown


def test_dirty_tree_hash_tracks_uncommitted_content(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_REPO", tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"],
                   cwd=tmp_path,
                   check=True)
    subprocess.run(["git", "config", "user.name", "Test"],
                   cwd=tmp_path,
                   check=True)
    source = tmp_path / "source.txt"
    source.write_text("one\n")
    subprocess.run(["git", "add", "source.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    assert provenance._dirty_tree_sha256() is None
    source.write_text("two\n")
    first = provenance._dirty_tree_sha256()
    source.write_text("three\n")
    second = provenance._dirty_tree_sha256()
    assert first and second and first != second


def test_accuracy_rejects_result_count_mismatch():
    with pytest.raises(ValueError, match="result count mismatch"):
        run_cpu_hls_accuracy._max_abs((object(), ), ())


@pytest.mark.parametrize("parser", [
    run_cpu_performance._positive_int,
    run_hls_resources._positive_int,
    run_hls_sweep._positive_int,
])
def test_benchmark_positive_integer_validation(parser):
    assert parser("3") == 3
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        parser("0")


def test_cpu_timing_reports_robust_statistics():
    single = run_cpu_performance._timing_statistics([4.0])
    assert single == {
        "best": 4.0,
        "mean": 4.0,
        "median": 4.0,
        "p95": 4.0,
        "stdev": 0.0,
        "samples": [4.0],
    }

    times = [3.0, 1.0, 2.0, 5.0]
    timing = run_cpu_performance._timing_statistics(times)
    assert timing["best"] == 1.0
    assert timing["mean"] == pytest.approx(2.75)
    assert timing["median"] == pytest.approx(2.5)
    assert timing["p95"] == 5.0  # rank ceil(0.95 * 4) - 1 = 3
    assert timing["stdev"] == pytest.approx(math.sqrt(8.75 / 3))
    assert timing["samples"] == times

    # At 20 samples the p95 rank lands on the second-largest value, not
    # the maximum.
    descending = [float(value) for value in range(20, 0, -1)]
    assert run_cpu_performance._timing_statistics(descending)["p95"] == 19.0

    measured = run_cpu_performance._timed(lambda: None, 4)
    assert set(measured) == set(timing)
    assert len(measured["samples"]) == 4


def test_quality_metrics_reject_degenerate_inputs():
    with pytest.raises(ValueError, match="finite nonzero"):
        metrics.measure_cut(np.zeros(8))
    with pytest.raises(ValueError, match="non-empty 2-D"):
        metrics.measure_image(np.zeros(8))
    with pytest.raises(ValueError, match="at least tile"):
        metrics.urban_contrast(np.ones((8, 8)), tile=16)


def test_checked_in_hls_results_are_self_consistent():
    paths = sorted(
        (REPO_ROOT / "benchmarks" / "results").glob("hls_algorithms_*.json"))
    assert paths
    for path in paths:
        data = json.loads(path.read_text())
        assert data["schema_version"] == 1
        constraints = data["constraints"]
        timing_budget = constraints.get(
            "timing_budget_ns", constraints["clock_ns"] -
            constraints.get("clock_uncertainty_ns", 0.0))
        if "report" in data:
            report = data["report"]
            assert report["timing_constraint_met"] == (
                report["estimated_clock_ns"] <= timing_budget)
            assert report["bram18k"] <= constraints["bram18k_budget"]
            assert report["uram"] <= constraints["uram_budget"]
            assert report["dsp"] <= constraints["dsp_budget"]
            assert len(report["report_sha256"]) == 64
        else:
            assert len(data["designs"]) == len(ALL)
            for design in data["designs"]:
                # A design may miss timing while the work to close it is
                # tracked; the record has to say so rather than omit it.
                if "timing_constraint_met" in design:
                    assert design["timing_constraint_met"] == (
                        design["estimated_clock_ns"] <= timing_budget)
                else:
                    assert design["estimated_clock_ns"] <= timing_budget
                # Resource budgets are hard: exceeding one does not fit.
                for resource in ("bram18k", "uram", "dsp", "ff", "lut"):
                    budget = constraints.get(f"{resource}_budget")
                    if budget is not None and resource in design:
                        assert design[resource] <= budget
                if "hls_csim_passed" in design:
                    assert design["hls_csim_passed"]
                if "source_sha256" in design:
                    assert len(design["source_sha256"]) == 64
                if "header_sha256" in design:
                    assert len(design["header_sha256"]) == 64
                # Synthesis time is recorded as a measurement, so it only
                # has to be a positive number of seconds.
                assert design["csynth_elapsed_s"] > 0
                if "warning_counts" in design:
                    assert design["warning_counts"]
                    assert all(count > 0
                               for count in design["warning_counts"].values())
                assert len(design["report_sha256"]) == 64


def test_checked_in_fft_dse_supports_the_balanced_default():
    data = json.loads((REPO_ROOT / "benchmarks/results" /
                       "hls_fft_dse_c64_n128_vitis_2022_2.json").read_text())
    designs = data["designs"]
    balanced = next(row for row in designs
                    if (row["fft_parallel_rows"], row["fft_stage_group"],
                        row["fft_io_unroll"]) == (4, 2, 4))
    serial = next(row for row in designs if row["fft_parallel_rows"] == 0)
    wide = next(row for row in designs if row["fft_parallel_rows"] == 8)
    ungrouped = next(row for row in designs if row["fft_stage_group"] == 0)
    assert balanced["latency_cycles"] < serial["latency_cycles"]
    assert wide["bram18k"] > balanced["bram18k"] * 1.5
    assert ungrouped["latency_cycles"] == balanced["latency_cycles"]
    assert ungrouped["bram18k"] > balanced["bram18k"]


def test_checked_in_cpu_baseline_records_mkl_and_numa_scaling():
    data = json.loads((REPO_ROOT / "benchmarks/results" /
                       "cpu_numa_mkl_2026_08_27.json").read_text())
    fft = data["fft_c128_2048x2048_median_ms"]
    assert all(mkl < sar
               for mkl, sar in zip(fft["mkl_dfti"], fft["sar_runtime"]))
    rda = data["rda_c128_n2048_median_ms"]
    assert rda["both_interleaved_120"] < rda["node0_local_60"]
    assert rda["node0_smt_120"] > rda["node0_local_60"]


def test_checked_in_gather_dse_supports_complexity_guard():
    data = json.loads((REPO_ROOT / "benchmarks/results" /
                       "hls_gather_dse_c128_vitis_2022_2.json").read_text())
    small = [row for row in data["designs"] if row["shape"] == [8, 64]]
    band = next(row for row in small if row["strategy"] == "band")
    direct = next(row for row in small if row["strategy"] == "direct")
    assert band["compute_ii"] < direct["compute_ii"]
    assert band["latency_cycles"] < direct["latency_cycles"]
    wide_band = next(row for row in data["designs"]
                     if row["shape"] == [8, 128] and row["strategy"] == "band")
    wide_direct = next(
        row for row in data["designs"]
        if row["shape"] == [8, 128] and row["strategy"] == "direct")
    assert not wide_band["success"] and wide_band["timeout_s"] == 900
    assert wide_direct["success"]


def test_checked_in_hls_budget_sweep_is_self_consistent():
    data = json.loads(plot_cpu_hls_results._BUDGET_SWEEP.read_text())
    assert data["schema_version"] == 1
    assert data["benchmark"] == "hls_budget_sweep"
    assert data["size"] == 1024
    assert data[
        "memory_budget_bytes"] == run_hls_resources._DEFAULT_MEMORY_BYTES
    assert data[
        "configured_memory_caps"] == run_hls_resources._DEFAULT_MEMORY_CAPS
    assert len(data["results"]) == 10 * len(ALL)
    for name in ALL:
        points = [row for row in data["results"] if row["name"] == name]
        assert [row["cap_fraction"] for row in points
                ] == pytest.approx([step / 10 for step in range(1, 11)])
        for row in points:
            assert sum(row["memory_caps"].values()) == row["budget"]
            assert row["on_chip_kib"] > 0
            assert row["external_footprint_mib"] > 0
            assert set(row["strategy"]) == set(AUTO_OPTIONS)


def test_handwritten_wka_matches_generated_production_contract():
    data = json.loads(
        (REPO_ROOT / "benchmarks" / "results" /
         "hls_algorithms_c64_production_vitis_2022_2.json").read_text())
    generated = next(design for design in data["designs"]
                     if design["algorithm"] == "omega-K")
    config = HLSConfig.resolve({
        "axi_bus_bits":
        data["constraints"]["axi_bus_bits"],
        "axi_max_burst_length":
        64,
        "axi_max_outstanding":
        16,
        "external_vector_max_lanes":
        generated["external_vector_max_lanes"],
        "external_vector_compute_lanes":
        generated["external_vector_compute_lanes"],
        "interp_cache_copies":
        generated["interp_cache_copies"],
    })
    run_hls_sweep.check_handwritten_geometry()
    run_hls_sweep.check_handwritten_production_contract(
        generated["shape"][0], config)

    hand = json.loads((REPO_ROOT / "examples" / "wka" / "handwritten_hls" /
                       "reports" / "production_csynth.json").read_text())
    assert hand["design"]["shape"] == generated["shape"]
    assert hand["design"]["dtype"] == "complex64"
    assert hand["design"]["array_ports"] == generated["axi_ports"]
    assert hand["design"]["axi_master_bundles"] == 8
    assert hand["constraints"]["complex_stream_bits"] == data["constraints"][
        "axi_bus_bits"]
    assert hand["constraints"]["plane_port_bits"] == 256
    assert hand["constraints"]["part"] == data["constraints"]["part"]
    assert hand["constraints"]["clock_ns"] == data["constraints"]["clock_ns"]
    assert hand["constraints"]["clock_uncertainty_ns"] == data["constraints"][
        "clock_uncertainty_ns"]
    for resource in ("bram18k", "uram", "dsp", "ff", "lut"):
        assert hand["constraints"][f"{resource}_budget"] == data[
            "constraints"][f"{resource}_budget"]
    assert hand["design"]["stolt_compute_lanes"] == generated[
        "external_vector_compute_lanes"]
    assert hand["design"]["stolt_cache_copies"] == generated[
        "interp_cache_copies"]
    assert hand["report"]["latency_min_cycles"] == hand["report"][
        "latency_max_cycles"]


def test_summary_figures_are_drawn_from_the_recorded_measurements():
    """The summary figures must plot the checked-in numbers, not new ones.

    They exist so the README can show the synthesis and CPU results without
    Vitis or the reference host, which only holds while they read
    `benchmarks/results/`. A figure that re-measured would caption
    this machine's numbers with those claims.
    """
    constraints, designs = plot_cpu_hls_results._production_designs()
    recorded = json.loads(plot_cpu_hls_results._PRODUCTION.read_text())
    assert len(designs) == len(recorded["designs"])
    assert constraints == recorded["constraints"]
    # Every bar is a share of a budget the record actually carries.
    for _, budget, _ in plot_cpu_hls_results._RESOURCES:
        assert constraints[budget] > 0

    sweep = plot_cpu_hls_results._cpu_sweep()
    cpu_record = json.loads(plot_cpu_hls_results._CPU_PERFORMANCE.read_text())
    assert cpu_record["benchmark"] == "cpu_performance"
    assert cpu_record["dtype"] == "c128"
    assert sum(map(len, sweep.values())) == len(cpu_record["measurements"])
    assert set(sweep) == {"omega-K", "Range-Doppler", "Chirp Scaling", "PFA"}
    for points in sweep.values():
        assert len(points) >= 6
        assert points == sorted(points), "sizes must be plotted in order"
        assert all(warm > 0 for _, warm, _ in points)
        assert all(n * n / warm > 0 for n, warm, _ in points)
    budget = json.loads(plot_cpu_hls_results._BUDGET_SWEEP.read_text())
    assert budget["benchmark"] == "hls_budget_sweep"


def test_runner_modules_expose_main():
    """Every runner imports cleanly and exposes a `main` entry point."""
    for module in (run_cpu_hls_accuracy, run_cpu_quality, run_cpu_performance,
                   run_cpu_precision, run_hls_resources, run_hls_sweep,
                   plot_cpu_impulse_response, plot_cpu_hls_results):
        assert callable(module.main), module.__name__


@pytest.mark.parametrize("script,flag", [
    ("run_cpu_hls_accuracy.py", "--algs"),
    ("run_cpu_quality.py", "--algs"),
    ("plot_cpu_impulse_response.py", "--sva-n"),
    ("plot_cpu_hls_results.py", "usage"),
    ("run_cpu_performance.py", "--sizes"),
    ("run_cpu_precision.py", "--algs"),
    ("run_hls_resources.py", "--budget-sweep"),
    ("run_hls_sweep.py", "--baseline-only"),
])
def test_runner_cli_help(script, flag):
    """Every runner's `--help` exits successfully and documents its
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


def test_hls_sweep_variant_file_validation(tmp_path):
    variants = tmp_path / "variants.json"
    variants.write_text('[{"fft_parallel_rows": 8}]')
    assert run_hls_sweep._load_variants(str(variants)) == [{
        "fft_parallel_rows":
        8
    }]
    variants.write_text('{"fft_parallel_rows": 8}')
    with pytest.raises(ValueError, match="list of option objects"):
        run_hls_sweep._load_variants(str(variants))


def test_hls_sweep_interrupt_terminates_child(tmp_path, monkeypatch):

    class Process:
        pid = 99999999

        def poll(self):
            return None

        def wait(self):
            return -15

    process = Process()
    terminated = []
    monkeypatch.setattr(run_hls_sweep.subprocess, "Popen",
                        lambda *args, **kwargs: process)
    monkeypatch.setattr(run_hls_sweep, "_terminate_group",
                        lambda child: terminated.append(child))
    job = run_hls_sweep.Job("wka", 32, "alos", {}, tmp_path,
                            tmp_path / "probe.tcl", "probe", None, "key")
    stop_event = threading.Event()
    stop_event.set()
    result = run_hls_sweep._run_job(job, "vitis_hls", 60, 0, 0, stop_event)
    assert terminated == [process]
    assert result["failure"] == "interrupted"


@requires_cpu
@pytest.mark.parametrize("name", ALL)
def test_chain_compiles_and_runs(name):
    chain = load(name, N)
    kernel = chain.compile_kernel()
    assert chain.run(kernel) is not None


@pytest.mark.parametrize("name", ALL)
def test_chain_exposes_its_kernel_arguments(name):
    """`Chain.args` must be the very list `run` passes to the kernel.

    `run_cpu_hls_accuracy` drives the same kernel through an HLS testbench, and
    rebuilding the inputs there instead of reusing these would let the
    two legs drift apart silently.
    """
    chain = load(name, N)
    assert chain.args is not None
    assert len(chain.args) == len(chain.kernel.arg_types)


@pytest.mark.parametrize("name", STRIPMAP)
def test_stripmap_chains_build_for_either_geometry(name):
    """A production synthesis comparison has to build the same collection
    the reference it is compared against implements, so the geometry is a
    parameter and the two must not share a kernel symbol."""
    synthetic = load(name, N, geometry="synthetic")
    alos = load(name, N, geometry="alos")
    assert synthetic.kernel.name != alos.kernel.name
    assert synthetic.args[0].shape == alos.args[0].shape


def test_unknown_geometry_is_rejected():
    with pytest.raises(ValueError, match="unknown geometry"):
        load("wka", N, geometry="lunar")


def test_throughput_figure_writes_to_tmp(tmp_path):
    results = [
        dict(name=name, n=n, throughput=float(n * n), skipped=False)
        for name in ALL for n in (16, 32, 64)
    ]
    run_cpu_performance.throughput_figure(results, out_dir=tmp_path)
    assert (tmp_path / "cpu_throughput.png").exists()


def test_throughput_figure_handles_skipped_points(tmp_path):
    """A timed-out point must not break the figure or the slope fit."""
    results = [
        dict(name="wka", n=16, throughput=256.0, skipped=False),
        dict(name="wka", n=32, throughput=0.0, skipped=True),
    ]
    run_cpu_performance.throughput_figure(results, out_dir=tmp_path)
    assert (tmp_path / "cpu_throughput.png").exists()


def test_cpu_throughput_defaults_include_production_raster():
    assert 16384 in run_cpu_performance._DEFAULT_SIZES
    assert run_cpu_performance._MAX_INPUT_SIZE["pfa"] == 8192


@requires_hls
def test_resource_measure_reports_ports():
    chain = load("wka", N)
    design = chain.compile_kernel(backend="hls",
                                  interface="axi",
                                  bram_bytes=1 << 22,
                                  uram_bytes=1 << 22,
                                  lutram_bytes=0)
    stats = run_hls_resources.measure(design.source(), design.name)
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
    stats = run_hls_resources.measure(source, "vector_top")
    assert stats["external_footprint_mib"] == 1024 / 2**20
    assert stats["mutable_on_chip_kib"] == 128 / 2**10
    assert stats["constant_rom_kib"] == 64 / 2**10


def test_resource_measure_rejects_format_drift():
    with pytest.raises(ValueError, match="cannot parse top function"):
        run_hls_resources.measure("void other() {}", "expected")
    malformed = "void top(\n  int unsupported\n) {\n}\n"
    with pytest.raises(ValueError, match="top-level declaration"):
        run_hls_resources.measure(malformed, "top")


def test_vitis_report_parser_checks_constraints(tmp_path):
    report = tmp_path / "probe_csynth.xml"
    report.write_text("""\
<profile>
  <ReportVersion><Version>2022.2</Version></ReportVersion>
  <UserAssignments>
    <Part>xcvu13p-fhgb2104-2-i</Part>
    <TopModelName>probe</TopModelName>
    <TargetClockPeriod>4.00</TargetClockPeriod>
    <ClockUncertainty>0.50</ClockUncertainty>
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
        <PipelineII>1</PipelineII><PipelineDepth>3</PipelineDepth>
        <Slack>0.75</Slack>
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
    assert parsed["clock_uncertainty_ns"] == 0.5
    assert parsed["timing_budget_ns"] == 3.5
    assert parsed["loop_latencies"] == [{
        "name": "LOOP_1",
        "trip_count": 8,
        "latency_cycles": 16,
        "iteration_latency": 2,
        "pipeline_ii": 1,
        "pipeline_depth": 3,
        "slack_ns": 0.75,
    }]
    assert parsed["interfaces"][0]["read_data_bits"] == 512
    assert len(parsed["xml_sha256"]) == 64
    assert not validate_constraints(parsed, HLSConfig.resolve())
    assert timing_shortfall(parsed, HLSConfig.resolve()) is None


def test_timing_is_reported_and_resources_are_enforced():
    """The two kinds of constraint must not be conflated.

    A resource budget decides whether a design can be placed on the device
    at all, so overrunning one is a violation. The clock is the goal the
    compiler optimizes toward and the estimate is pre-route, so missing it
    is a number to weigh -- reported beside the result, never as a failure.
    """
    report = {
        "part": HLSConfig.resolve().part,
        "target_clock_ns": 4.0,
        "clock_uncertainty_ns": 0.5,
        "timing_budget_ns": 3.5,
        "estimated_clock_ns": 5.5,
        "resources": {
            "bram_18k": 1,
            "uram": 1,
            "dsp": 1,
            "ff": 1,
            "lut": 1
        },
    }
    config = HLSConfig.resolve()
    assert not validate_constraints(report, config), \
        "a missed clock is not a constraint violation"
    shortfall = timing_shortfall(report, config)
    assert shortfall["over_ns"] == pytest.approx(2.0)

    # The same report over one resource budget is a violation.
    over = dict(report, resources=dict(report["resources"], dsp=10**9))
    assert any("dsp" in v for v in validate_constraints(over, config))


@requires_hls
def test_budget_sweep_and_figure(tmp_path):
    results = run_hls_resources.sweep(["wka"], N, steps=2)
    assert results
    assert len({row["budget"] for row in results}) == len(results)
    # Every algorithm uses the same device-relative axis. Infeasible points
    # are omitted; the last point is the exact configured memory budget.
    assert results == sorted(results, key=lambda row: row["budget"])
    assert results[-1]["cap_fraction"] == 1.0
    assert results[-1]["budget"] == run_hls_resources._DEFAULT_MEMORY_BYTES
    chain = load("wka", N)
    for row in results:
        assert run_hls_resources._feasible(chain, row["budget"]) is not None
        assert sum(row["memory_caps"].values()) == row["budget"]
        assert set(row["strategy"]) == set(AUTO_OPTIONS)
    run_hls_resources.budget_figure(results, N, out_dir=tmp_path)
    assert (tmp_path / "hls_budget_sweep.png").exists()


@requires_hls
@pytest.mark.parametrize("name", ALL)
def test_f32_build_parses_as_valid_ir(name):
    """Every chain built with `dtype=sar.c64` must produce IR the C++
    verifiers accept -- the option threads precision through annotations,
    casts and result types, and a missed spot surfaces here."""
    from sar.compiler.toolchain import find_tool

    ir = run_cpu_precision.narrow_ir(name)
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
    kernel, mode = run_cpu_precision._build(name, 64, single=True)
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
    counts = run_cpu_precision.complex_type_references(
        run_cpu_precision.narrow_ir(name))
    assert counts["complex<f32>"] > 0


def test_pfa_keeps_its_interpolation_axes_double():
    """PFA's collection axes stay f64 whatever the dtype: they feed
    `interp1d` positions, which the language requires to be double. The
    mode label says so, and the position planes in the narrowed IR are
    still f64."""
    kernel, mode = run_cpu_precision._build("pfa", 64, single=True)
    assert mode == "c64 only"
    counts = run_cpu_precision.complex_type_references(kernel.to_mlir())
    assert counts["f64"] > 0  # the interpolation position planes


#: The production designs whose synthesis results are checked in, and the
#: geometry each was measured at. Keyed by the kernel symbol the recorded
#: results name.
_PRODUCTION = {
    "wka_c64_alos": ("wka", 16384, "alos"),
    "rda_c64_alos": ("rda", 16384, "alos"),
    "csa_c64_alos": ("csa", 16384, "alos"),
    "pfa_c64": ("pfa", 8192, "synthetic"),
}

#: The strategy keys the recorded results carry. These are what the design
#: was built with, so they are what a rebuild has to reproduce.
_RECORDED_STRATEGY = ("fft_parallel_rows", "fft_stage_group", "fft_io_unroll",
                      "external_vector_max_lanes",
                      "external_vector_compute_lanes",
                      "array_partition_max_factor",
                      "interp_complete_bank_max_elements",
                      "interp_cache_copies", "interp_full_row_max_bytes")


@requires_hls
@pytest.mark.parametrize("top", sorted(_PRODUCTION))
def test_production_designs_still_derive_their_recorded_strategy(top):
    """The checked-in synthesis results must stay reproducible.

    `benchmarks/results/` records latency and resources for designs the
    compiler derived a particular strategy for. Nothing else ties those
    numbers to the compiler: a change in `autotune` that picks
    another lane count or stage grouping leaves the tables describing a
    design this tree no longer emits, and both suites stay green while it
    happens. So the derivation is replayed here and held to what was
    measured.

    Re-deriving is cheap -- it reads the traced module, not a synthesis
    run -- which is what makes the guard affordable at production scale.
    """
    from sar.backends.hls import autotune
    from sar.backends.hls.config import HLSConfig

    recorded = {
        design["top"]: design
        for design in json.loads((
            REPO_ROOT / "benchmarks" / "results" /
            "hls_algorithms_c64_production_vitis_2022_2.json"
        ).read_text())["designs"]
    }
    assert top in recorded, f"{top} is not in the recorded results"
    design = recorded[top]

    name, n, geometry = _PRODUCTION[top]
    chain = load(name, n, dtype=sar.c64, geometry=geometry)
    assert chain.kernel.name == top, "the recorded design named another kernel"

    config = HLSConfig.resolve({
        "interface": design["interface"],
        "axi_bus_bits": design.get("axi_bus_bits", 512),
    })
    facts = autotune.measure_kernel(chain.kernel.to_mlir())
    plan = autotune.plan(config, facts)
    for key in _RECORDED_STRATEGY:
        if key not in design:
            continue
        assert plan.values[key] == design[key], (
            f"{top}: {key} is now {plan.values[key]!r}, but the checked-in "
            f"synthesis result was measured at {design[key]!r}; re-run the "
            "sweep or restore the derivation")
