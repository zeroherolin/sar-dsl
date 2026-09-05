#!/usr/bin/env python3
"""Check documented defaults and result tables against their data sources."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_DISPLAY_NAME = {"Polar Format": "PFA"}


def _tables(markdown: str) -> dict[tuple[str, ...], list[list[str]]]:
    """Return Markdown tables keyed by their header cells."""
    lines = markdown.splitlines()
    tables = {}
    index = 0
    while index + 1 < len(lines):
        if not lines[index].startswith("|") or not re.fullmatch(
                r"\|(?:\s*:?-+:?\s*\|)+", lines[index + 1]):
            index += 1
            continue
        header = tuple(cell.strip()
                       for cell in lines[index].strip("|").split("|"))
        rows = []
        index += 2
        while index < len(lines) and lines[index].startswith("|"):
            rows.append(
                [cell.strip() for cell in lines[index].strip("|").split("|")])
            index += 1
        tables[header] = rows
    return tables


def _check_table(errors: list[str], name: str, actual: list[list[str]],
                 expected: list[list[str]]) -> None:
    if actual == expected:
        return
    for index, (got, want) in enumerate(zip(actual, expected), 1):
        if got != want:
            errors.append(
                f"{name} row {index}: docs have {got!r}, data says {want!r}")
    if len(actual) != len(expected):
        errors.append(
            f"{name}: docs have {len(actual)} rows, data has {len(expected)}")


def _shape(record: dict) -> str:
    source = record["shape"]
    output = record.get("output_shape", source)
    return f"{source[0]}² → {output[0]}²"


def _int(value: int) -> str:
    return f"{value:,}"


def check() -> list[str]:
    errors = []
    backend = (ROOT / "docs/backends.md").read_text()
    from sar.backends.hls.config import HLSConfig
    config = HLSConfig.resolve()
    defaults = {
        "bram_bytes": config.bram_bytes,
        "uram_bytes": config.uram_bytes,
        "lutram_bytes": config.lutram_bytes,
        "dsp": config.dsp,
        "ff": config.ff,
        "lut": config.lut,
        "bram_block_bytes": config.bram_block_bytes,
        "uram_block_bytes": config.uram_block_bytes,
        "axi_bus_bits": config.axi_bus_bits,
        "axi_max_burst_length": config.axi_max_burst_length,
        "axi_max_outstanding": config.axi_max_outstanding,
    }
    for key, expected in defaults.items():
        match = re.search(rf"\| `{key}` \| `([^`]+)` \|", backend)
        if not match:
            errors.append(f"docs/backends.md has no default row for {key}")
            continue
        actual = match.group(1)
        if actual != str(expected):
            errors.append(f"{key}: docs say {actual}, config says {expected}")

    report = (ROOT / "benchmarks/README.md").read_text()
    result_files = set(re.findall(r"results/([A-Za-z0-9_.-]+\.json)", report))
    for name in result_files:
        if not (ROOT / "benchmarks/results" / name).is_file():
            errors.append(f"benchmark report references missing result {name}")

    tables = _tables(report)
    cpu_header = ("Algorithm", "Input edge N", "Warm", "NumPy reference",
                  "Speedup")
    cpu = json.loads((ROOT / "benchmarks/results" /
                      "cpu_performance_c128_llvm_22.json").read_text())
    cpu_rows = []
    for row in cpu["measurements"]:
        speedup = (f"{row['numpy_s'] / row['warm_s']:.1f}x"
                   if row["numpy_s"] > 0 else "—")
        cpu_rows.append([
            row["algorithm"],
            str(row["n"]), f"{row['warm_s']:.3f} s", f"{row['numpy_s']:.2f} s",
            speedup
        ])
    _check_table(errors, "CPU reference table", tables.get(cpu_header, []),
                 cpu_rows)

    hls_header = ("Algorithm", "Input → output", "Estimated clock", "BRAM18K",
                  "URAM", "DSP", "FF", "LUT", "Latency cycles",
                  "Interval cycles")
    hls = json.loads((ROOT / "benchmarks/results" /
                      "hls_algorithms_c128_512_vitis_2022_2.json").read_text())
    hls_rows = [[
        _DISPLAY_NAME.get(row["algorithm"], row["algorithm"]),
        _shape(row), f"{row['estimated_clock_ns']:.3f} ns",
        _int(row["bram18k"]),
        _int(row["uram"]),
        _int(row["dsp"]),
        _int(row["ff"]),
        _int(row["lut"]),
        _int(row["latency_cycles"]),
        _int(row["interval_cycles"])
    ] for row in hls["designs"]]
    _check_table(errors, "c128 HLS table", tables.get(hls_header, []),
                 hls_rows)

    production = json.loads(
        (ROOT / "benchmarks/results" /
         "hls_algorithms_c64_production_vitis_2022_2.json").read_text())
    handwritten = json.loads((ROOT / "examples/wka/handwritten_hls/reports" /
                              "production_csynth.json").read_text())
    clock_ns = production["constraints"]["clock_ns"]
    production_latency = []
    production_resources = []
    for row in production["designs"]:
        name = _DISPLAY_NAME.get(row["algorithm"], row["algorithm"])
        label = f"{name}, generated"
        production_latency.append([
            label,
            _shape(row), f"{clock_ns:.3f} ns",
            f"{row['estimated_clock_ns']:.3f} ns",
            _int(row["latency_cycles"]),
            f"{row['latency_cycles'] * clock_ns / 1e9:.3f} s",
            f"{row['csynth_elapsed_s']:.0f} s"
        ])
        production_resources.append([
            label,
            _shape(row),
            _int(row["bram18k"]),
            _int(row["uram"]),
            _int(row["dsp"]),
            _int(row["ff"]),
            _int(row["lut"])
        ])
        if row["algorithm"] == "omega-K":
            hand_report = handwritten["report"]
            production_latency.append([
                "omega-K, hand-written",
                _shape(row), f"{clock_ns:.3f} ns",
                f"{hand_report['estimated_clock_ns']:.3f} ns",
                _int(hand_report["latency_min_cycles"]),
                f"{hand_report['latency_min_cycles'] * clock_ns / 1e9:.3f} s",
                f"{handwritten['tool']['csynth_elapsed_s']:.0f} s"
            ])
            production_resources.append([
                "omega-K, hand-written",
                _shape(row),
                _int(hand_report["bram18k"]),
                _int(hand_report["uram"]),
                _int(hand_report["dsp"]),
                _int(hand_report["ff"]),
                _int(hand_report["lut"])
            ])

    latency_header = ("Design", "Input → output", "Target clock",
                      "Estimated clock", "Latency cycles", "Latency time",
                      "C-synth time")
    resource_header = ("Design", "Input → output", "BRAM18K", "URAM", "DSP",
                       "FF", "LUT")
    _check_table(errors, "production HLS latency table",
                 tables.get(latency_header, []), production_latency)
    _check_table(errors, "production HLS resource table",
                 tables.get(resource_header, []), production_resources)
    return errors


if __name__ == "__main__":
    problems = check()
    if problems:
        raise SystemExit("\n".join(problems))
