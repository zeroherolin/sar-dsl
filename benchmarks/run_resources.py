#!/usr/bin/env python3
"""Resource benchmark: what each imaging chain costs on an FPGA.

Emits every chain through the HLS backend and reports what the design
asks of the device -- interfaces, external buffer footprint, on-chip
memory -- as a function of scene size and of the on-chip budget it was
given.

These are the numbers that decide whether a design fits, and they come
from the emitted C++ rather than from synthesis, so the sweep is cheap
enough to run on every change. Vitis reports LUT/FF/DSP after synthesis;
those are not modelled here.

Usage:
    python benchmarks/run_resources.py [--algs wka rda csa pfa]
                                       [--sizes 512 4096 16384]
                                       [--budgets 4194304]
                                       [--budget-sweep [--sweep-size 512]
                                        [--sweep-steps 8]]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms import ALL, LABELS, load  # noqa: E402
from hls_reports import parse_csynth_xml, validate_constraints  # noqa: E402
from provenance import environment  # noqa: E402

from sar.backends.hls.config import HLSConfig, HLSConfigError  # noqa: E402

_ELEMENT_BYTES = {"float": 4, "double": 8}

ASSETS = Path(__file__).resolve().parent / "assets"
_DEFAULT_LUTRAM_BYTES = HLSConfig.resolve().lutram_bytes


def _top_signature(source: str, top: str) -> str:
    match = re.search(rf"^void {top}\(\n?(.*?)\n\) \{{", source, re.S | re.M)
    if not match:
        raise ValueError(f"cannot parse top function {top!r}")
    return match.group(1)


def _array_declarations(signature: str) -> list[tuple[str, str, int]]:
    arrays = []
    for line in signature.split(","):
        if not line.strip():
            continue
        decl = re.fullmatch(r"\s*(float|double)\s+(\w+)((?:\[\d+\])+)\s*",
                            line)
        if not decl:
            raise ValueError(f"cannot parse top-level declaration: {line!r}")
        elements = 1
        for dim in re.findall(r"\[(\d+)\]", decl.group(3)):
            elements *= int(dim)
        arrays.append((decl.group(1), decl.group(2), elements))
    return arrays


def measure(source: str, top: str) -> dict:
    """Device-facing cost of an emitted design."""
    signature = _top_signature(source, top)

    external_bytes = 0
    for dtype, _, elements in _array_declarations(signature):
        external_bytes += elements * _ELEMENT_BYTES[dtype]

    on_chip_bytes = 0
    for decl in re.finditer(r"^\s+(float|double) \w+((?:\[\d+\])+);", source,
                            re.M):
        elements = 1
        for dim in re.findall(r"\[(\d+)\]", decl.group(2)):
            elements *= int(dim)
        on_chip_bytes += elements * _ELEMENT_BYTES[decl.group(1)]

    # Every port takes its own bundle (shared ones serialize their bus
    # requests); a design that broke the invariant would misreport here.
    bundles = {
        line.split("bundle=")[1].split()[0]
        for line in source.splitlines() if "m_axi" in line
    }
    ports = source.count("m_axi")
    assert len(bundles) == ports, (ports, sorted(bundles))
    return {
        "ports": ports,
        "external_footprint_mib": external_bytes / 2**20,
        "on_chip_kib": on_chip_bytes / 2**10,
        "dataflow": source.count("#pragma HLS dataflow"),
        "lines": source.count("\n"),
    }


def _tier_caps(budget: int) -> dict:
    """Splits one total into the tier caps the sweep varies together:
    block RAM and UltraRAM half each, distributed RAM held at the
    shipped default (it serves banks, not planes)."""
    return {
        "bram_bytes": budget // 2,
        "uram_bytes": budget - budget // 2,
        "lutram_bytes": _DEFAULT_LUTRAM_BYTES,
    }


def working_set_bytes(name: str, size: int) -> int:
    """Full working set of a chain, measured rather than assumed.

    Compiled with device-scale caps nothing at these sizes can exhaust,
    every buffer stays resident; the emitted arrays plus the I/O ports
    then add up to the footprint the design would need to hold
    everything internally. This is the upper bound of the budget sweep
    -- past it, raising the caps cannot move anything.
    """
    chain = load(name, size)
    design = chain.compile_kernel(backend="hls",
                                  interface="axi",
                                  **_tier_caps(1 << 30))
    source = design.source()
    declarations = _array_declarations(_top_signature(source, design.name))
    io_ports = sum(2 if tensor.dtype.is_complex else 1
                   for tensor in chain.kernel.arg_types +
                   chain.kernel.declared_result_types)
    if any(elements > 1 for _, _, elements in declarations[io_ports:]):
        raise RuntimeError(
            f"{name} at N={size} still spills with device-scale caps; "
            "cannot establish the full working set")
    stats = measure(source, design.name)
    return int(stats["on_chip_kib"] * 2**10 +
               stats["external_footprint_mib"] * 2**20)


def sweep(names, size: int, steps: int) -> list:
    """Measures each chain across caps from one primitive's worth up to
    the full working set."""
    results = []
    for name in names:
        full = working_set_bytes(name, size)
        # Linear ladder up to the full working set. The floor is a
        # handful of URAM blocks rather than 0: the caps are hard, and a
        # tier set that cannot hold the resident tables refuses the
        # design instead of streaming harder.
        floor = 8 * 36864
        budgets = sorted(
            {max(floor, round(full * i / steps))
             for i in range(steps + 1)})
        chain = load(name, size)
        for budget in budgets:
            try:
                design = chain.compile_kernel(backend="hls",
                                              interface="axi",
                                              **_tier_caps(budget))
            except HLSConfigError:
                continue
            stats = measure(design.source(), design.name)
            stats.update(name=name, size=size, budget=budget, full=full)
            results.append(stats)
    return results


def budget_figure(results: list, size: int, out_dir=None) -> None:
    """Saves port count and on-chip usage against the budget.

    Pass a `pathlib.Path` as `out_dir` to redirect output (e.g. a pytest
    `tmp_path`) instead of writing into assets/.
    """
    import matplotlib.pyplot as plt

    from common.plot import PALETTE, apply_style

    apply_style()
    fig, (ax_p, ax_m) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    for idx, name in enumerate(ALL):
        pts = [r for r in results if r["name"] == name]
        if not pts:
            continue
        color = PALETTE[idx % len(PALETTE)]
        # Budget as a fraction of the chain's own working set, so chains
        # with very different footprints share one axis.
        x = [
            100.0 * p["budget"] / p["full"] if p["full"] else 0.0 for p in pts
        ]
        ax_p.plot(x, [p["ports"] for p in pts],
                  "o-",
                  color=color,
                  linewidth=1.5,
                  markersize=3.5,
                  label=LABELS[name])
        ax_m.plot(x, [p["on_chip_kib"] / 1024.0 for p in pts],
                  "o-",
                  color=color,
                  linewidth=1.5,
                  markersize=3.5,
                  label=LABELS[name])

    ax_p.set_ylabel("AXI master ports")
    ax_m.set_ylabel("On-chip memory (MiB)")
    for ax in (ax_p, ax_m):
        ax.set_xlabel("On-chip budget (% of working set)")
    ax_p.set_title(f"Interfaces vs budget  (N={size})")
    ax_m.set_title(f"On-chip usage vs budget  (N={size})")
    ax_p.legend(fontsize=8.5)

    dest = Path(out_dir) if out_dir is not None else ASSETS
    dest.mkdir(exist_ok=True)
    out = dest / "budget_sweep.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algs", nargs="+", default=list(ALL), choices=ALL)
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 4096])
    parser.add_argument("--budgets",
                        nargs="+",
                        type=int,
                        default=[4 * 1024 * 1024],
                        help="on-chip budgets, in bytes")
    parser.add_argument("--budget-sweep",
                        action="store_true",
                        help="sweep the on-chip budget from 0 to the full "
                        "working set and write budget_sweep.png")
    parser.add_argument("--sweep-size", type=int, default=512)
    parser.add_argument("--sweep-steps", type=int, default=8)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--reports",
                        nargs="+",
                        help="parse and validate Vitis *_csynth.xml reports")
    parser.add_argument("--json", help="write machine-readable results here")
    args = parser.parse_args()

    if args.reports:
        config = HLSConfig.resolve()
        reports = []
        failed = False
        for path in args.reports:
            report = parse_csynth_xml(path)
            violations = validate_constraints(report, config)
            report["violations"] = violations
            reports.append(report)
            failed |= bool(violations)
            status = "PASS" if not violations else "FAIL"
            top = report["top"]
            clock = report["estimated_clock_ns"]
            latency = report["latency_cycles"]
            print(f"{top:<14} {clock:>7.3f} ns "
                  f"{latency:>12} cycles {status}")
            for violation in violations:
                print(f"  {violation}", file=sys.stderr)
        if args.json:
            Path(args.json).write_text(
                json.dumps(
                    {
                        "environment": environment(),
                        "benchmark": "vitis_csynth",
                        "reports": reports,
                    },
                    indent=2) + "\n")
        if failed:
            raise SystemExit(1)
        return

    if args.budget_sweep:
        results = sweep(args.algs, args.sweep_size, args.sweep_steps)
        print(f"{'chain':<14} {'size':>6} {'budget/KiB':>11} {'%WS':>6} "
              f"{'ports':>6} {'ext/MiB':>10} {'chip/KiB':>10} "
              f"{'dataflow':>9}")
        print("-" * 78)
        for r in results:
            pct = 100.0 * r["budget"] / r["full"] if r["full"] else 0.0
            print(f"{LABELS[r['name']]:<14} {r['size']:>6} "
                  f"{r['budget'] / 2**10:>11.0f} {pct:>6.0f} "
                  f"{r['ports']:>6} "
                  f"{r['external_footprint_mib']:>10.1f} "
                  f"{r['on_chip_kib']:>10.1f} {r['dataflow']:>9}")
        if not args.no_figure:
            budget_figure(results, args.sweep_size)
        if args.json:
            Path(args.json).write_text(
                json.dumps(
                    {
                        "environment": environment(),
                        "benchmark": "hls_budget_sweep",
                        "results": results,
                    },
                    indent=2) + "\n")
        return

    print(f"{'chain':<14} {'size':>6} {'budget':>9} {'ports':>6} "
          f"{'ext/MiB':>10} {'chip/KiB':>10} {'dataflow':>9}")
    print("-" * 70)
    results = []
    for name in args.algs:
        for size in args.sizes:
            for budget in args.budgets:
                chain = load(name, size)
                design = chain.compile_kernel(backend="hls",
                                              interface="axi",
                                              **_tier_caps(budget))
                stats = measure(design.source(), design.name)
                stats.update(name=name, size=size, budget=budget)
                results.append(stats)
                print(f"{LABELS[name]:<14} {size:>6} {budget >> 20:>7} MiB "
                      f"{stats['ports']:>6} "
                      f"{stats['external_footprint_mib']:>10.1f} "
                      f"{stats['on_chip_kib']:>10.1f} "
                      f"{stats['dataflow']:>9}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "environment": environment(),
                    "benchmark": "hls_resources",
                    "results": results,
                },
                indent=2) + "\n")


if __name__ == "__main__":
    main()
