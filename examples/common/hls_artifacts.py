"""Artifact set of the ALOS-scale HLS example runners.

The three stripmap runners emit the same two things and differ only in
which imaging chain they trace, so the emission lives here and each
runner stays a short script.

Two designs come out of one geometry because no single design can be
both. At the full 16384 x 16384 raster the planes have to stream from
DRAM, which means `axi_interface=True`: every streamed buffer becomes
its own AXI master port, and a testbench has no data to drive the
promoted intermediates with -- golden data exists for the kernel's own
inputs and results, not for its scratch. That design is therefore
emitted for synthesis and not simulated. Simulation happens on a second
design at a raster small enough to keep every plane on chip, so the top
function is the kernel's own signature and the testbench can drive it.

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


def _axi_summary(source: str):
    """(ports, bundles) of the m_axi interfaces the design presents."""
    ports = source.count("m_axi")
    bundles = {
        line.split("bundle=")[1].split()[0]
        for line in source.splitlines()
        if "m_axi" in line and "bundle=" in line
    }
    return ports, len(bundles)


def _annotate_csim_script(path: Path, algorithm: str, csim_n: int, n: int,
                          uses_streams: bool) -> None:
    """Records in the csim script which raster it simulates and why."""
    if uses_streams:
        fallback = (
            "# This design uses hls::stream for its dataflow handshakes, so\n"
            "# it needs the real Vitis headers: the `stubs/` stand-ins cover\n"
            "# ap_int and the math headers but not hls::stream. Run the csim\n"
            "# through vitis_hls, or shrink --csim-n until the dataflow\n"
            "# regions disappear if you want the plain-C++ fallback.\n")
    else:
        fallback = (
            "# Without Vitis HLS the same package runs through any C++\n"
            "# compiler:\n"
            f"#   c++ -O2 -I stubs {algorithm}_alos.cpp "
            f"{algorithm}_alos_tb.cpp \\\n"
            "#       -o csim -pthread && ./csim\n")
    header = (
        f"# C simulation of the {algorithm.upper()} chain at ALOS-1 radar\n"
        f"# parameters on a {csim_n} x {csim_n} raster.\n"
        "#\n"
        f"# The design emitted for the {n} x {n} scene is\n"
        f"# {algorithm}_alos_axi.cpp, and it is not this one: it compiles\n"
        "# with axi_interface=True, where every buffer the compiler moves\n"
        "# off chip becomes its own AXI master port. Those promoted ports\n"
        "# are kernel scratch, and no golden data exists for scratch, so\n"
        "# that design is synthesized rather than simulated. This raster\n"
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
    axi = build_kernel(n, alos_params(n), name=f"{top}_axi").compile(
        backend="hls", options={"axi_interface": True})
    source = axi.source()
    axi.write_synthesis_script(out)  # writes {top}_axi.cpp + its csynth.tcl
    ports, bundles = _axi_summary(source)
    result.update(axi_lines=source.count("\n"),
                  axi_ports=ports,
                  axi_bundles=bundles)
    print(f"      {source.count(chr(10))} lines, {ports} AXI ports on "
          f"{bundles} bundle(s), {time.time() - started:.1f} s")

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
    _annotate_csim_script(out / f"{top}_csim.tcl", algorithm, csim_n, n,
                          uses_streams)
    result.update(targets=len(targets),
                  golden_peak=float(golden.max()),
                  uses_streams=uses_streams)
    print(f"      {len(targets)} point targets, golden peak "
          f"{golden.max():.1f}, {np.count_nonzero(golden):d} nonzero samples")
    if uses_streams:
        print("      note: the design uses hls::stream, so csim needs the "
              "real Vitis headers")
    print(f"      saved {out}")
    return result
