#!/usr/bin/env python3
"""Resource benchmark: what each imaging chain costs on an FPGA.

Emits every chain through the HLS backend and reports what the design
asks of the device -- interfaces, DRAM traffic, on-chip memory -- as a
function of scene size and of the on-chip budget it was given.

These are the numbers that decide whether a design fits, and they come
from the emitted C++ rather than from synthesis, so the sweep is cheap
enough to run on every change. Vitis reports LUT/FF/DSP after synthesis;
those are not modelled here.

Usage:
    python benchmarks/run_resources.py [--algs wka rda csa pfa]
                                       [--sizes 512 4096 16384]
                                       [--budgets 4194304]
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms import ALL, LABELS, load  # noqa: E402

_ELEMENT_BYTES = {"float": 4, "double": 8}


def _top_signature(source: str, top: str) -> str:
    match = re.search(rf"^void {top}\(\n?(.*?)\n\) \{{", source, re.S | re.M)
    return match.group(1) if match else ""


def measure(source: str, top: str) -> dict:
    """Device-facing cost of an emitted design."""
    signature = _top_signature(source, top)

    dram_bytes = 0
    for line in signature.split(","):
        decl = re.search(r"(float|double) \w+((?:\[\d+\])*)", line)
        if not decl:
            continue
        elements = 1
        for dim in re.findall(r"\[(\d+)\]", decl.group(2)):
            elements *= int(dim)
        dram_bytes += elements * _ELEMENT_BYTES[decl.group(1)]

    on_chip_bytes = 0
    for decl in re.finditer(r"^\s+(float|double) \w+((?:\[\d+\])+);", source,
                            re.M):
        elements = 1
        for dim in re.findall(r"\[(\d+)\]", decl.group(2)):
            elements *= int(dim)
        on_chip_bytes += elements * _ELEMENT_BYTES[decl.group(1)]

    bundles = {
        line.split("bundle=")[1].split()[0]
        for line in source.splitlines() if "m_axi" in line
    }
    return {
        "ports": source.count("m_axi"),
        "bundles": len(bundles),
        "dram_mib": dram_bytes / 2**20,
        "on_chip_kib": on_chip_bytes / 2**10,
        "dataflow": source.count("#pragma HLS dataflow"),
        "lines": source.count("\n"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algs", nargs="+", default=list(ALL), choices=ALL)
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 4096])
    parser.add_argument("--budgets",
                        nargs="+",
                        type=int,
                        default=[4 * 1024 * 1024],
                        help="on-chip budgets, in bytes")
    args = parser.parse_args()

    print(f"{'chain':<14} {'size':>6} {'budget':>9} {'ports':>6} "
          f"{'bundles':>8} {'DRAM/MiB':>10} {'chip/KiB':>10} {'dataflow':>9}")
    print("-" * 78)
    for name in args.algs:
        for size in args.sizes:
            for budget in args.budgets:
                chain = load(name, size)
                design = chain.compile_kernel(backend="hls",
                                              axi_interface=True,
                                              on_chip_budget=budget)
                stats = measure(design.source(), design.name)
                print(f"{LABELS[name]:<14} {size:>6} {budget >> 20:>7} MiB "
                      f"{stats['ports']:>6} {stats['bundles']:>8} "
                      f"{stats['dram_mib']:>10.1f} "
                      f"{stats['on_chip_kib']:>10.1f} "
                      f"{stats['dataflow']:>9}")


if __name__ == "__main__":
    main()
