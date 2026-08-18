#!/usr/bin/env python3
"""Collect Vitis HLS 2022.2 csynth XML files into stable JSON/CSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path


FIELDS = {
    "target_clock_ns": ".//TargetClockPeriod",
    "estimated_clock_ns": ".//EstimatedClockPeriod",
    "latency_min": ".//Best-caseLatency",
    "latency_max": ".//Worst-caseLatency",
    "interval_min": ".//Interval-min",
    "interval_max": ".//Interval-max",
}
RESOURCES = ("BRAM_18K", "URAM", "DSP", "FF", "LUT")


def value(root: ET.Element, path: str) -> str:
    node = root.find(path)
    return "" if node is None or node.text is None else node.text.strip()


def parse_report(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    row = {"report": str(path)}
    for key, path_expr in FIELDS.items():
        row[key] = value(root, path_expr)
    for resource in RESOURCES:
        used = value(root, f".//AreaEstimates/Resources/{resource}")
        available = value(root, f".//AreaEstimates/AvailableResources/{resource}")
        row[resource.lower()] = used
        row[f"{resource.lower()}_available"] = available
        try:
            row[f"{resource.lower()}_pct"] = f"{100.0 * float(used) / float(available):.3f}"
        except (ValueError, ZeroDivisionError):
            row[f"{resource.lower()}_pct"] = ""
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--output", type=Path, default=Path("reports/hls_summary"))
    args = parser.parse_args()

    reports = sorted(args.work.glob("**/syn/report/*_csynth.xml"))
    rows = [parse_report(path) for path in reports]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"collected {len(rows)} report(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
