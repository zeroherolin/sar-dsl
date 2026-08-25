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
    python benchmarks/run_hls_resources.py [--algs wka rda csa pfa]
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
from hls_reports import parse_csynth_bundle, timing_shortfall  # noqa: E402
from hls_reports import validate_constraints  # noqa: E402
from provenance import environment  # noqa: E402

from sar.backends.hls.autotune import AUTO_OPTIONS  # noqa: E402
from sar.backends.hls.config import HLSConfig, HLSConfigError  # noqa: E402
from sar.errors import CompilationError  # noqa: E402

_ELEMENT_BYTES = {"float": 4, "double": 8}
_TYPE = r"(?:float|double|hls::vector<(?:float|double),\s*\d+>)"

ASSETS = Path(__file__).resolve().parent / "assets"
_DEFAULT_CONFIG = HLSConfig.resolve()
_DEFAULT_MEMORY_CAPS = {
    "bram_bytes": int(_DEFAULT_CONFIG.bram_bytes),
    "uram_bytes": int(_DEFAULT_CONFIG.uram_bytes),
    "lutram_bytes": int(_DEFAULT_CONFIG.lutram_bytes),
}
_DEFAULT_MEMORY_BYTES = sum(_DEFAULT_MEMORY_CAPS.values())


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _top_signature(source: str, top: str) -> str:
    match = re.search(rf"^void {top}\(\n?(.*?)\n\) \{{", source, re.S | re.M)
    if not match:
        raise ValueError(f"cannot parse top function {top!r}")
    return match.group(1)


def _type_bytes(type_name: str) -> int:
    if type_name in _ELEMENT_BYTES:
        return _ELEMENT_BYTES[type_name]
    match = re.fullmatch(r"hls::vector<(float|double),\s*(\d+)>", type_name)
    if not match:
        raise ValueError(f"unsupported HLS element type: {type_name!r}")
    return _ELEMENT_BYTES[match.group(1)] * int(match.group(2))


def _parameters(signature: str) -> list[str]:
    """Splits a C++ parameter list without splitting template arguments."""
    parameters = []
    start = 0
    angle_depth = 0
    for index, char in enumerate(signature):
        if char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth -= 1
        elif char == "," and angle_depth == 0:
            parameters.append(signature[start:index])
            start = index + 1
    parameters.append(signature[start:])
    return parameters


def _array_declarations(signature: str) -> list[tuple[str, str, int, int]]:
    arrays = []
    for line in _parameters(signature):
        if not line.strip():
            continue
        decl = re.fullmatch(rf"\s*({_TYPE})\s+(\w+)((?:\[\d+\])+)\s*", line)
        if not decl:
            raise ValueError(f"cannot parse top-level declaration: {line!r}")
        elements = 1
        for dim in re.findall(r"\[(\d+)\]", decl.group(3)):
            elements *= int(dim)
        arrays.append((decl.group(1), decl.group(2), elements,
                       _type_bytes(decl.group(1))))
    return arrays


def _on_chip_storage(source: str) -> tuple[int, int]:
    """Logical mutable storage and constant ROM bytes in emitted C++."""
    mutable = 0
    constant = 0
    declarations = re.finditer(
        rf"^(?P<indent>\s*)(?P<qual>(?:(?:static|const|constexpr)\s+)*)"
        rf"(?P<type>{_TYPE})\s+\w+"
        rf"(?P<dims>(?:\[\d+\])+)\s*(?:=|;)", source, re.M)
    for declaration in declarations:
        elements = 1
        for dim in re.findall(r"\[(\d+)\]", declaration.group("dims")):
            elements *= int(dim)
        size = elements * _type_bytes(declaration.group("type"))
        qualifiers = declaration.group("qual").split()
        if "const" in qualifiers or "constexpr" in qualifiers:
            constant += size
        else:
            mutable += size

    for table in re.finditer(
            r"^static constexpr LinearTable<(float|double),\s*(\d+)>\s+\w+\(",
            source, re.M):
        constant += _ELEMENT_BYTES[table.group(1)] * int(table.group(2))
    return mutable, constant


def measure(source: str, top: str) -> dict:
    """Device-facing cost of an emitted design."""
    signature = _top_signature(source, top)

    external_bytes = 0
    for _, _, elements, element_bytes in _array_declarations(signature):
        external_bytes += elements * element_bytes

    mutable_bytes, constant_bytes = _on_chip_storage(source)

    # Ports and masters are counted separately: a full-size plane takes its
    # own master, while compact read-only tables share one the way a
    # hand-written design would. Both matter to an integrator -- the ports
    # are the design's ABI, the bundles are the physical channels a board
    # has to provide.
    bundles = {
        line.split("bundle=")[1].split()[0]
        for line in source.splitlines()
        if "#pragma HLS interface m_axi" in line
    }
    ports = sum("#pragma HLS interface m_axi" in line
                for line in source.splitlines())
    assert len(bundles) <= ports, (ports, sorted(bundles))
    return {
        "ports": ports,
        "bundles": len(bundles),
        "external_footprint_mib": external_bytes / 2**20,
        "on_chip_kib": (mutable_bytes + constant_bytes) / 2**10,
        "mutable_on_chip_kib": mutable_bytes / 2**10,
        "constant_rom_kib": constant_bytes / 2**10,
        "dataflow": source.count("#pragma HLS dataflow"),
        "lines": source.count("\n"),
    }


def _tier_caps(budget: int) -> dict:
    """Scales the default BRAM, URAM, and LUTRAM caps to ``budget`` bytes."""
    bram = budget * _DEFAULT_MEMORY_CAPS["bram_bytes"] // _DEFAULT_MEMORY_BYTES
    uram = budget * _DEFAULT_MEMORY_CAPS["uram_bytes"] // _DEFAULT_MEMORY_BYTES
    return {
        "bram_bytes": bram,
        "uram_bytes": uram,
        "lutram_bytes": budget - bram - uram,
    }


def _feasible(chain, budget: int):
    """The design at `budget`, or None when the caps refuse it.

    A budget below the design's minimum is reported either by the
    backend's own retry ladder giving up or by a pass refusing to place a
    buffer; both mean the same thing here.
    """
    try:
        return chain.compile_kernel(backend="hls",
                                    interface="axi",
                                    **_tier_caps(budget))
    except (HLSConfigError, CompilationError):
        return None


def sweep(names, size: int, steps: int) -> list:
    """Measures each chain from 1/``steps`` to the configured memory caps."""
    results = []
    for name in names:
        chain = load(name, size)
        for step in range(1, steps + 1):
            budget = round(_DEFAULT_MEMORY_BYTES * step / steps)
            caps = _tier_caps(budget)
            design = _feasible(chain, budget)
            if design is None:
                continue
            stats = measure(design.source(), design.name)
            stats.update(
                name=name,
                size=size,
                budget=budget,
                cap_fraction=step / steps,
                memory_caps=caps,
                strategy={key: design.config[key]
                          for key in AUTO_OPTIONS})
            results.append(stats)
    return results


def budget_figure(results: list, size: int, out_dir=None) -> None:
    """Saves the selected interface and logical storage against memory caps.

    Pass a `pathlib.Path` as `out_dir` to redirect output (e.g. a pytest
    `tmp_path`) instead of writing into assets/.
    """
    import matplotlib.pyplot as plt

    from common.plot import PALETTE, apply_style

    apply_style()
    fig, (ax_p, ax_m, ax_e) = plt.subplots(1, 3, figsize=(12.6, 3.8))

    for idx, name in enumerate(ALL):
        pts = [r for r in results if r["name"] == name]
        if not pts:
            continue
        color = PALETTE[idx % len(PALETTE)]
        x = [100.0 * p["cap_fraction"] for p in pts]
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
        ax_e.plot(x, [p["external_footprint_mib"] for p in pts],
                  "o-",
                  color=color,
                  linewidth=1.5,
                  markersize=3.5,
                  label=LABELS[name])

    ax_p.set_ylabel("AXI array ports")
    ax_m.set_ylabel("Local arrays (logical MiB)")
    ax_e.set_ylabel("Top-level arrays (logical MiB)")
    for ax in (ax_p, ax_m, ax_e):
        ax.set_xlabel("Memory caps (% of configured device budget)")
        ax.set_xlim(0.0, 102.0)
    ax_p.set_title(f"AXI ports selected  (input N={size})")
    ax_m.set_title(f"Local storage selected  (input N={size})")
    ax_e.set_title(f"External footprint selected  (input N={size})")
    handles, labels = ax_p.get_legend_handles_labels()
    fig.legend(handles,
               labels,
               ncol=len(labels),
               loc="lower center",
               bbox_to_anchor=(0.5, -0.015),
               fontsize=8.5)
    fig.subplots_adjust(bottom=0.25, wspace=0.34)

    dest = Path(out_dir) if out_dir is not None else ASSETS
    dest.mkdir(exist_ok=True)
    out = dest / "hls_budget_sweep.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algs", nargs="+", default=list(ALL), choices=ALL)
    parser.add_argument("--sizes",
                        nargs="+",
                        type=_positive_int,
                        default=[512, 4096])
    parser.add_argument("--budgets",
                        nargs="+",
                        type=_positive_int,
                        default=[8 * 1024 * 1024],
                        help="on-chip budgets, in bytes")
    parser.add_argument("--budget-sweep",
                        action="store_true",
                        help="sweep proportional BRAM/URAM/LUTRAM caps up to "
                        "the configured device budget and write "
                        "hls_budget_sweep.png")
    parser.add_argument("--sweep-size", type=_positive_int, default=512)
    parser.add_argument("--sweep-steps", type=_positive_int, default=8)
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
            bundle = parse_csynth_bundle(path)
            report = bundle["top"]
            violations = validate_constraints(report, config)
            report["violations"] = violations
            # Missing the clock is reported, never fatal: the period is a
            # goal the compiler optimizes toward and the estimate is
            # pre-route, unlike a resource budget the device cannot exceed.
            shortfall = timing_shortfall(report, config)
            report["timing_shortfall"] = shortfall
            reports.append(bundle)
            failed |= bool(violations)
            status = ("FAIL" if violations else
                      "PASS (timing warning)" if shortfall else "PASS")
            top = report["top"]
            target = report["target_clock_ns"]
            timing_budget = report["timing_budget_ns"]
            clock = report["estimated_clock_ns"]
            latency = report["latency_cycles"]
            # Wall-clock latency is quoted at the target period: that is the
            # constraint the design closed against and the rate a board
            # clocks it at. The estimated period is timing margin, and
            # dividing by it instead would credit a design for slack the
            # deployed clock never converts into throughput. Both are shown
            # so the margin is visible next to the rate it does not set.
            seconds = latency * target * 1e-9
            print(f"{top:<14} target {target:>6.3f} ns "
                  f"budget {timing_budget:>6.3f} ns "
                  f"est {clock:>6.3f} ns "
                  f"{latency:>12} cycles {seconds:>9.3f} s {status}")
            if shortfall:
                print(
                    f"  timing: estimate misses the target by "
                    f"{shortfall['over_ns']:.3f} ns after uncertainty",
                    file=sys.stderr)
            for warning in bundle["warnings"]:
                print(
                    f"  warning {warning['code']}: "
                    f"{warning['count']} occurrence(s)",
                    file=sys.stderr)
            for violation in violations:
                print(f"  {violation}", file=sys.stderr)
        if args.json:
            Path(args.json).write_text(
                json.dumps(
                    {
                        "environment": environment(),
                        "benchmark": "vitis_csynth",
                        "command": [Path(sys.executable).name, *sys.argv],
                        "reports": reports,
                    },
                    indent=2) + "\n")
        if failed:
            raise SystemExit(1)
        return

    if args.budget_sweep:
        results = sweep(args.algs, args.sweep_size, args.sweep_steps)
        print(f"{'chain':<14} {'size':>6} {'budget/KiB':>11} {'%cap':>6} "
              f"{'ports':>6} {'ext/MiB':>10} {'chip/KiB':>10} "
              f"{'dataflow':>9}")
        print("-" * 78)
        for r in results:
            pct = 100.0 * r["cap_fraction"]
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
                        "schema_version": 1,
                        "environment": environment(),
                        "benchmark": "hls_budget_sweep",
                        "size": args.sweep_size,
                        "memory_budget_bytes": _DEFAULT_MEMORY_BYTES,
                        "configured_memory_caps": _DEFAULT_MEMORY_CAPS,
                        "command": [Path(sys.executable).name, *sys.argv],
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
                    "command": [Path(sys.executable).name, *sys.argv],
                    "results": results,
                },
                indent=2) + "\n")


if __name__ == "__main__":
    main()
