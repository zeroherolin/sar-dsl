"""Registry of the imaging chains under benchmark.

Wraps each example algorithm behind one interface so the runners do not
repeat the import and setup dance. `pfa` is a spotlight collection with
its own geometry, hence the separate loader.

The stripmap chains take a `geometry`. `synthetic` derives a
self-consistent airborne collection in which a point target focuses to
about one resolution cell, which is what the image-quality and accuracy
runners measure against. `alos` is the ALOS-1 acquisition the examples
process and the one the hand-written HLS reference implements, so it is
what a production-scale synthesis comparison has to use: the geometry
sets how far Stolt remapping displaces a sample, and with it whether the
resampling reads a bounded band or a whole row.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "python"), str(_REPO / "examples")]

import sar  # noqa: E402

from common.params import alos_params, synthetic_params  # noqa: E402
from common.simulate import single_target_scene  # noqa: E402

__all__ = [
    "STRIPMAP", "ALL", "GEOMETRIES", "LABELS", "load", "focus_point_target"
]

#: Stripmap chains sharing the RadarParams geometry.
STRIPMAP = ("wka", "rda", "csa")
#: Every benchmarked chain, in reporting order.
ALL = STRIPMAP + ("pfa", )
#: Collection geometries the stripmap chains can be built for.
GEOMETRIES = {"synthetic": synthetic_params, "alos": alos_params}

LABELS = {
    "wka": "omega-K",
    "rda": "Range-Doppler",
    "csa": "Chirp Scaling",
    "pfa": "PFA"
}


@dataclass
class Chain:
    """One benchmarkable imaging chain, set up for a given size."""

    name: str
    #: (backend="cpu", **options) -> compiled kernel or emitted design
    compile_kernel: Callable
    run: Callable  # (kernel) -> image(s)
    run_reference: Callable  # () -> numpy reference image(s)
    #: The exact argument list `run` passes to the kernel, `[raw, *inputs]`.
    #: Exposed so a caller that has to drive the same kernel by another
    #: route -- an HLS testbench, say -- uses the very same arrays rather
    #: than rebuilding them and risking a silent mismatch.
    args: List[np.ndarray] = None
    #: The underlying (untraced) kernel, for its declared argument and
    #: result types.
    kernel: object = None


def _chain(name: str, kernel, raw, inputs, run_reference) -> Chain:
    """Assembles a Chain around a built (untraced) kernel: the argument
    arrays are cast to whatever the kernel declares, so one code path
    serves both precisions."""
    values = [raw, *inputs]
    if len(values) != len(kernel.arg_types):
        raise ValueError(f"{name}: expected {len(kernel.arg_types)} kernel "
                         f"arguments, got {len(values)}")
    args = [
        np.asarray(value, dtype=tensor_type.dtype.to_numpy())
        for value, tensor_type in zip(values, kernel.arg_types)
    ]
    return Chain(name,
                 lambda backend="cpu", **opts: kernel.compile(
                     backend=backend, options=opts or None),
                 lambda k: k(*args),
                 run_reference,
                 args,
                 kernel)


def _stripmap(name: str, n: int, dtype, geometry: str) -> Chain:
    processor = {
        "wka": "WKAProcessor",
        "rda": "RDAProcessor",
        "csa": "CSAProcessor"
    }[name]
    alg = __import__(f"{name}.algorithm",
                     fromlist=["build_kernel", "make_inputs"])
    ref = getattr(__import__(f"{name}.reference", fromlist=[processor]),
                  processor)

    params = GEOMETRIES[geometry](n)
    suffix = "" if dtype is sar.c128 else "_c64"
    if geometry != "synthetic":
        suffix += f"_{geometry}"
    kernel = alg.build_kernel(n, params, name=f"{name}{suffix}", dtype=dtype)
    return _chain(
        name, kernel, single_target_scene(n,
                                          params), alg.make_inputs(n, params),
        lambda: ref(n, params).process(single_target_scene(n, params)))


def _pfa(name: str, n: int, dtype, geometry: str) -> Chain:
    from pfa.algorithm import build_kernel, make_inputs
    from pfa.geometry import Geometry
    from pfa.reference import PFAProcessor

    collection = Geometry(n)
    kernel = build_kernel(n,
                          collection,
                          name=name if dtype is sar.c128 else f"{name}_c64",
                          dtype=dtype)
    raw = collection.simulate(collection.demo_targets())
    return _chain(name, kernel, raw, make_inputs(n, collection),
                  lambda: PFAProcessor(n, collection).process(raw))


def load(name: str, n: int, dtype=sar.c128, geometry: str = "synthetic"):
    """Sets up `name` for an `n`-sized scene (polar-grid edge for PFA).

    `dtype` selects the working precision the kernel is built with;
    `geometry` the stripmap collection it is built for (PFA carries its
    own spotlight geometry and ignores it).
    """
    if name not in ALL:
        raise ValueError(f"unknown algorithm {name!r}; known: {ALL}")
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; "
                         f"known: {tuple(GEOMETRIES)}")
    return (_pfa if name == "pfa" else _stripmap)(name, n, dtype, geometry)


def focus_point_target(
        n: int,
        algs: Tuple[str, ...] = STRIPMAP) -> List[Tuple[str, np.ndarray]]:
    """Focuses a single scene-center scatterer with the selected stripmap
    chains (all of them by default)."""
    images = []
    for name in algs:
        chain = load(name, n)
        kernel = chain.compile_kernel()
        images.append((name, np.asarray(chain.run(kernel), dtype=np.float64)))
    return images
