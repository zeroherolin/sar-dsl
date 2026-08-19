"""Parsing and validation of Vitis HLS synthesis reports."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

_BRAM18_BYTES = 18 * 1024 // 8
_URAM_BYTES = 288 * 1024 // 8


def _text(root, path: str) -> str:
    node = root.find(path)
    if node is None or node.text is None:
        raise ValueError(f"missing {path!r} in Vitis synthesis report")
    return node.text.strip()


def parse_csynth_xml(path) -> dict:
    """Returns the stable summary fields from a Vitis `*_csynth.xml`."""
    root = ET.parse(Path(path)).getroot()
    resources = {
        name.lower(): int(_text(root, f"./AreaEstimates/Resources/{name}"))
        for name in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
    }
    available = {
        name.lower():
        int(_text(root, f"./AreaEstimates/AvailableResources/{name}"))
        for name in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
    }
    return {
        "version":
        _text(root, "./ReportVersion/Version"),
        "part":
        _text(root, "./UserAssignments/Part"),
        "top":
        _text(root, "./UserAssignments/TopModelName"),
        "target_clock_ns":
        float(_text(root, "./UserAssignments/TargetClockPeriod")),
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
        "resources":
        resources,
        "available":
        available,
    }


def validate_constraints(report: dict, config) -> list:
    """Returns constraint violations for a parsed report and HLS config."""
    violations = []
    if report["part"] != config.part:
        violations.append(
            f"part {report['part']} does not match configured {config.part}")
    if report["target_clock_ns"] != float(config.clock_ns):
        violations.append("report target clock does not match configuration")
    if report["estimated_clock_ns"] > float(config.clock_ns):
        violations.append(
            f"estimated clock {report['estimated_clock_ns']:.3f} ns exceeds "
            f"target {float(config.clock_ns):.3f} ns")

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
