"""Artifact set of the ALOS-scale HLS example runners.

The three stripmap runners emit the same two things and differ only in
which imaging chain they trace, so the emission lives here and each
runner stays a short script.

Two designs come out of one geometry because no single design can be
both. At the full 16384 x 16384 raster the planes have to stream from
DRAM, which means `interface="axi"`: kernel I/O becomes AXI masters and
internal spilled planes use compiler-managed scratch arenas. The full-raster
arrays are too large for a practical host simulation, so that design is
emitted for synthesis. Simulation uses a smaller `ap_memory` design.

The reduced design is not a different radar: `alos_params` keeps `fc`,
`fs`, PRF, `Vr`, `R0`, `Kr` and `Tp`, and the frequency axes span
+/- fs/2 and +/- PRF/2 at every raster size, so the phase functions,
the Stolt map and the migration corrections are evaluated with the
acquisition's own constants. What shrinks is the sampled window and the
dwell: the point target focuses, at coarser azimuth resolution than the
full aperture would give.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .params import alos_params
from .simulate import demo_scene

__all__ = ["emit_alos_artifacts"]


def _axi_ports(source: str) -> int:
    """m_axi ports the design presents (each on its own bundle)."""
    return source.count("m_axi")


def _annotate_csim_script(path: Path, algorithm: str, csim_n: int,
                          n: int) -> None:
    """Records in the csim script which raster it simulates and why."""
    fallback = (
        "# Without Vitis HLS the package runs through any C++ compiler:\n"
        f"#   c++ -O2 -I stubs {algorithm}_alos.cpp "
        f"{algorithm}_alos_tb.cpp \\\n"
        "#       -o csim -pthread && ./csim\n")
    header = (
        f"# C simulation of the {algorithm.upper()} chain at ALOS-1 radar\n"
        f"# parameters on a {csim_n} x {csim_n} raster.\n"
        "#\n"
        f"# The design emitted for the {n} x {n} scene is\n"
        f"# {algorithm}_alos_axi.cpp, and it is not this one: it compiles\n"
        "# with interface='axi', where internal off-chip buffers use\n"
        "# compiler-managed scratch arenas. Its host arrays are too large\n"
        "# for practical simulation, so it is synthesized instead.\n"
        "# This raster\n"
        "# keeps the planes on chip, so the top function is the kernel's\n"
        "# own signature and the testbench can drive every port. The radar\n"
        "# parameters are identical at both sizes; the sampled window and\n"
        "# the dwell are what shrink.\n"
        "#\n" + fallback + "\n")
    path.write_text(header + path.read_text())


def emit_alos_artifacts(algorithm: str,
                        build_kernel,
                        make_inputs,
                        processor,
                        n: int,
                        csim_n: int,
                        out: Path,
                        testbench: bool = True) -> dict:
    """Emits the full artifact set of an ALOS-scale HLS example.

    `build_kernel(n, params, name=...)` and `make_inputs(n, params)` come
    from the algorithm's `algorithm.py`, `processor(n, params).process`
    from its `reference.py`. Writes into `out`:

        <algorithm>_alos_axi.cpp         the `n x n` design, AXI ports
        <algorithm>_alos_axi_csynth.tcl  Vitis HLS synthesis script
        <algorithm>_alos.cpp             the `csim_n x csim_n` design
        <algorithm>_alos_tb.cpp          testbench against the reference
        <algorithm>_alos_csim.tcl        Vitis HLS csim script
        <algorithm>_alos_csynth.tcl      synthesis script (csim raster)
        <algorithm>_alos_tb_data/        golden data, one file per port
        stubs/                           Vitis header stand-ins
    """
    out.mkdir(parents=True, exist_ok=True)
    top = f"{algorithm}_alos"
    result = {}

    print(f"[1/3] Emitting the {n}x{n} {algorithm.upper()} design "
          "(AXI, for synthesis) ...")
    started = time.time()
    axi = build_kernel(n, alos_params(n),
                       name=f"{top}_axi").compile(backend="hls",
                                                  options={"interface": "axi"})
    source = axi.source()
    axi.write_synthesis_script(out)  # writes {top}_axi.cpp + its csynth.tcl
    ports = _axi_ports(source)
    result.update(axi_lines=source.count("\n"), axi_ports=ports)
    print(f"      {source.count(chr(10))} lines, {ports} AXI master "
          f"ports, {time.time() - started:.1f} s")

    print(f"[2/3] Emitting the {csim_n}x{csim_n} {algorithm.upper()} design "
          "(csim package) ...")
    params = alos_params(csim_n)
    design = build_kernel(csim_n, params, name=top).compile(backend="hls")
    source = design.source()
    result.update(csim_lines=source.count("\n"))
    print(f"      {source.count(chr(10))} lines")

    if not testbench:
        (out / f"{top}.cpp").write_text(source)
        print(f"[3/3] Saved {out} (design only)")
        return result

    print("[3/3] Simulating the scene and running the NumPy reference ...")
    raw, targets = demo_scene(csim_n, params)
    golden = processor(csim_n, params).process(raw)
    design.write_testbench([raw, *make_inputs(csim_n, params)], [golden], out)
    uses_streams = "hls::stream" in source
    _annotate_csim_script(out / f"{top}_csim.tcl", algorithm, csim_n, n)
    result.update(targets=len(targets),
                  golden_peak=float(golden.max()),
                  uses_streams=uses_streams)
    print(f"      {len(targets)} point targets, golden peak "
          f"{golden.max():.1f}, {np.count_nonzero(golden):d} nonzero samples")
    print(f"      saved {out}")
    return result
