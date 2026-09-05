"""Single-design artifact packages for the ALOS HLS example runners.

Each runner compiles one kernel at the requested raster and writes one
self-contained package. The interface and all other HLS choices follow the
normal backend configuration chain: shipped `hls_config.yaml`, an optional
project config, then Python compile options. Design, testbench, golden data,
and Tcl scripts therefore describe the same top function, port set, and size.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .params import alos_params
from .simulate import demo_scene

__all__ = ["emit_alos_artifacts"]


def _port_count(design) -> int:
    """Number of physical ports in the generated top-level schema."""
    return len(design.interface_schema())


def _replace_artifact_directory(staging: Path, output: Path) -> None:
    """Atomically installs a completed package over an older package."""
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            output.unlink()
        else:
            shutil.rmtree(output)
    os.replace(staging, output)


def emit_alos_artifacts(algorithm: str,
                        build_kernel,
                        make_inputs,
                        processor,
                        n: int,
                        out: Path,
                        options: Mapping | None = None) -> dict:
    """Emits one ALOS-geometry design and its matching validation package.

    `options` is passed directly to `compile(backend="hls")`. If it is None,
    the backend uses the shipped or project-level HLS configuration unchanged.
    The output directory contains only artifacts for `<algorithm>_alos` at
    `n x n`: declarations, implementation, optional ROM tables, manifest,
    testbench, golden data, stubs, C-sim/C-synth/C-RTL scripts, and the
    portable C++ fallback script.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    output = Path(out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    top = f"{algorithm}_alos"
    params = alos_params(n)

    print(f"[1/3] Compiling the {n}x{n} {algorithm.upper()} HLS design ...")
    started = time.time()
    design = build_kernel(n, params, name=top).compile(backend="hls",
                                                       options=options)
    source = design.source()
    interface = design.config.interface
    ports = _port_count(design)
    print(f"      interface={interface}, {ports} physical ports, "
          f"{source.count(chr(10))} lines, {time.time() - started:.1f} s")

    print("[2/3] Simulating the scene and running the NumPy reference ...")
    raw, targets = demo_scene(n, params)
    golden = processor(n, params).process(raw)
    print(f"      {len(targets)} point targets, golden peak "
          f"{golden.max():.1f}, {np.count_nonzero(golden):d} nonzero samples")

    print("[3/3] Writing the matching design and validation package ...")
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.",
                                     dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        design.write_testbench([raw, *make_inputs(n, params)], [golden],
                               staging,
                               max_bytes=0)
        _replace_artifact_directory(staging, output)

    print(f"      saved {output}")
    return {
        "top": top,
        "n": n,
        "interface": interface,
        "ports": ports,
        "lines": source.count("\n"),
        "targets": len(targets),
        "golden_peak": float(golden.max()),
    }
