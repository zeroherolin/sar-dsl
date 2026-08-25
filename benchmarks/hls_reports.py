"""Parsing and validation of Vitis HLS synthesis reports."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

_BRAM18_BYTES = 18 * 1024 // 8
_URAM_BYTES = 288 * 1024 // 8
_WARNING = re.compile(r"^WARNING: \[(?P<code>[^]]+)\] (?P<message>.*)$")


def _text(root, path: str) -> str:
    node = root.find(path)
    if node is None or node.text is None:
        raise ValueError(f"missing {path!r} in Vitis synthesis report")
    return node.text.strip()


def _optional_text(root, path: str):
    node = root.find(path)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loop_latencies(root) -> list:
    summary = root.find("./PerformanceEstimates/SummaryOfLoopLatency")
    if summary is None:
        return []
    loops = []
    for node in summary:
        entry = {"name": node.tag}
        for xml_name, key in (
            ("TripCount", "trip_count"),
            ("Latency", "latency_cycles"),
            ("IterationLatency", "iteration_latency"),
            ("PipelineII", "pipeline_ii"),
            ("PipelineDepth", "pipeline_depth"),
        ):
            value = _optional_text(node, xml_name)
            if value not in (None, ""):
                entry[key] = int(value)
        slack = _optional_text(node, "Slack")
        if slack not in (None, ""):
            entry["slack_ns"] = float(slack)
        loops.append(entry)
    return loops


def _interfaces(root) -> list:
    interfaces = {}
    for port in root.findall("./InterfaceSummary/RtlPorts"):
        protocol = _optional_text(port, "IOProtocol")
        obj = _optional_text(port, "Object")
        name = _optional_text(port, "name")
        bits = _optional_text(port, "Bits")
        if not protocol or not obj or not name or not bits:
            continue
        key = (obj, protocol)
        entry = interfaces.setdefault(
            key, {
                "object": obj,
                "protocol": protocol,
                "read_data_bits": 0,
                "write_data_bits": 0,
            })
        width = int(bits)
        if name.endswith(("_RDATA", "_TDATA")):
            entry["read_data_bits"] = max(entry["read_data_bits"], width)
        if name.endswith(("_WDATA", "_TDATA")):
            entry["write_data_bits"] = max(entry["write_data_bits"], width)
    return sorted(interfaces.values(),
                  key=lambda entry: (entry["protocol"], entry["object"]))


def parse_csynth_xml(path) -> dict:
    """Returns the stable summary fields from a Vitis `*_csynth.xml`."""
    path = Path(path)
    root = ET.parse(path).getroot()
    resources = {
        name.lower(): int(_text(root, f"./AreaEstimates/Resources/{name}"))
        for name in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
    }
    available = {
        name.lower():
        int(_text(root, f"./AreaEstimates/AvailableResources/{name}"))
        for name in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
    }
    target_clock = float(_text(root, "./UserAssignments/TargetClockPeriod"))
    uncertainty = float(_text(root, "./UserAssignments/ClockUncertainty"))
    return {
        "version":
        _text(root, "./ReportVersion/Version"),
        "part":
        _text(root, "./UserAssignments/Part"),
        "top":
        _text(root, "./UserAssignments/TopModelName"),
        "target_clock_ns":
        target_clock,
        "clock_uncertainty_ns":
        uncertainty,
        "timing_budget_ns":
        target_clock - uncertainty,
        "estimated_clock_ns":
        float(
            _text(
                root, "./PerformanceEstimates/SummaryOfTimingAnalysis/"
                "EstimatedClockPeriod")),
        "latency_cycles":
        int(
            _text(
                root, "./PerformanceEstimates/SummaryOfOverallLatency/"
                "Worst-caseLatency")),
        "interval_cycles":
        int(
            _text(
                root, "./PerformanceEstimates/SummaryOfOverallLatency/"
                "Interval-max")),
        "pipeline_type":
        _text(root, "./PerformanceEstimates/PipelineType"),
        "loop_latencies":
        _loop_latencies(root),
        "interfaces":
        _interfaces(root),
        "resources":
        resources,
        "available":
        available,
        "xml_sha256":
        _sha256(path),
    }


def parse_vitis_warnings(path) -> list:
    """Groups Vitis warnings by code, preserving first-seen order.

    RTL generation may repeat one diagnostic hundreds of times. A count and
    a few representative messages keep the report actionable without burying
    the synthesis metrics in duplicate text.
    """
    path = Path(path)
    if not path.is_file():
        return []
    grouped = OrderedDict()
    for line in path.read_text(errors="replace").splitlines():
        match = _WARNING.match(line)
        if match is None:
            continue
        code = match.group("code")
        entry = grouped.setdefault(code, {
            "code": code,
            "count": 0,
            "examples": [],
        })
        entry["count"] += 1
        message = match.group("message")
        if message not in entry["examples"] and len(entry["examples"]) < 3:
            entry["examples"].append(message)
    return list(grouped.values())


def parse_csynth_bundle(path) -> dict:
    """Parses a top report and the per-module reports beside it."""
    path = Path(path)
    top = parse_csynth_xml(path)
    modules = []
    for candidate in sorted(path.parent.glob("*_csynth.xml")):
        if candidate == path:
            continue
        report = parse_csynth_xml(candidate)
        modules.append({
            "top": report["top"],
            "estimated_clock_ns": report["estimated_clock_ns"],
            "clock_uncertainty_ns": report["clock_uncertainty_ns"],
            "timing_budget_ns": report["timing_budget_ns"],
            "latency_cycles": report["latency_cycles"],
            "interval_cycles": report["interval_cycles"],
            "pipeline_type": report["pipeline_type"],
            "loop_latencies": report["loop_latencies"],
            "resources": report["resources"],
            "xml_sha256": report["xml_sha256"],
        })

    elapsed = None
    manifest = None
    warnings = []
    for parent in list(path.parents)[:6]:
        elapsed_path = parent / f"{top['top']}_csynth_elapsed_s.txt"
        if elapsed is None and elapsed_path.is_file():
            elapsed = float(elapsed_path.read_text().strip())
        manifest_path = parent / "design_manifest.json"
        if manifest is None and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
        log_path = parent / "vitis_hls.log"
        if not warnings and log_path.is_file():
            warnings = parse_vitis_warnings(log_path)
    return {
        "top": top,
        "modules": modules,
        "csynth_elapsed_s": elapsed,
        "manifest": manifest,
        "warnings": warnings,
    }


def validate_constraints(report: dict, config) -> list:
    """Hard-constraint violations for a parsed report and HLS config.

    Only the constraints a design cannot be shipped against: the resource
    budgets, which decide whether it fits the device at all, and the
    identity of the target it was built for. Timing is a goal rather than a
    budget -- see `timing_shortfall`.
    """
    violations = []
    if report["part"] != config.part:
        violations.append(
            f"part {report['part']} does not match configured {config.part}")
    if report["target_clock_ns"] != float(config.clock_ns):
        violations.append("report target clock does not match configuration")
    expected_uncertainty = (float(config.clock_ns) *
                            float(config.clock_uncertainty_percent) / 100.0)
    if abs(report["clock_uncertainty_ns"] - expected_uncertainty) > 0.005:
        violations.append("report clock uncertainty does not match "
                          "configuration")

    resources = report["resources"]
    limits = {
        "bram_18k": int(config.bram_bytes) // _BRAM18_BYTES,
        "uram": int(config.uram_bytes) // _URAM_BYTES,
        "dsp": int(config.dsp),
        "ff": int(config.ff),
        "lut": int(config.lut),
    }
    for name, limit in limits.items():
        if resources[name] > limit:
            violations.append(
                f"{name} usage {resources[name]} exceeds budget {limit}")
    return violations


def timing_shortfall(report: dict, config):
    """How far the estimate misses the clock target, or None if it meets it.

    Separate from `validate_constraints` because it is a different kind of
    result. The period is what the compiler optimizes toward and the
    estimate is pre-route, so a miss is a number to weigh against the
    project's own margin -- not a design that cannot be placed.
    """
    estimated = report["estimated_clock_ns"]
    target = report.get("timing_budget_ns", config.effective_clock_ns())
    if estimated <= target:
        return None
    return {
        "estimated_clock_ns": estimated,
        "target_clock_ns": float(config.clock_ns),
        "clock_uncertainty_ns": (float(config.clock_ns) - target),
        "timing_budget_ns": target,
        "over_ns": estimated - target,
    }
