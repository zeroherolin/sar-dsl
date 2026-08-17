"""Registry of the imaging chains under benchmark.

Wraps each example algorithm behind one interface so the runners do not
repeat the import and setup dance. `pfa` is a spotlight collection with
its own geometry, hence the separate loader.
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

from common.params import synthetic_params  # noqa: E402
from common.simulate import single_target_scene  # noqa: E402

__all__ = ["STRIPMAP", "ALL", "LABELS", "load", "focus_point_target"]

#: Stripmap chains sharing the RadarParams geometry.
STRIPMAP = ("wka", "rda", "csa")
#: Every benchmarked chain, in reporting order.
ALL = STRIPMAP + ("pfa", )

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
    args = [
        np.asarray(a, dtype=t.dtype.to_numpy())
        for a, t in zip([raw, *inputs], kernel.arg_types)
    ]
    return Chain(name,
                 lambda backend="cpu", **opts: kernel.compile(
                     backend=backend, options=opts or None),
                 lambda k: k(*args),
                 run_reference,
                 args,
                 kernel)


def _stripmap(name: str, n: int, dtype) -> Chain:
    processor = {
        "wka": "WKAProcessor",
        "rda": "RDAProcessor",
        "csa": "CSAProcessor"
    }[name]
    alg = __import__(f"{name}.algorithm",
                     fromlist=["build_kernel", "make_inputs"])
    ref = getattr(__import__(f"{name}.reference", fromlist=[processor]),
                  processor)

    params = synthetic_params(n)
    kernel = alg.build_kernel(
        n,
        params,
        name=name if dtype is sar.c128 else f"{name}_c64",
        dtype=dtype)
    return _chain(
        name, kernel, single_target_scene(n,
                                          params), alg.make_inputs(n, params),
        lambda: ref(n, params).process(single_target_scene(n, params)))


def _pfa(name: str, n: int, dtype) -> Chain:
    from pfa.algorithm import build_kernel, make_inputs
    from pfa.geometry import Geometry
    from pfa.reference import PFAProcessor

    geometry = Geometry(n)
    kernel = build_kernel(n,
                          geometry,
                          name=name if dtype is sar.c128 else f"{name}_c64",
                          dtype=dtype)
    raw = geometry.simulate(geometry.demo_targets())
    return _chain(name, kernel, raw, make_inputs(n, geometry),
                  lambda: PFAProcessor(n, geometry).process(raw))


def load(name: str, n: int, dtype=sar.c128) -> Chain:
    """Sets up `name` for an `n`-sized scene (polar-grid edge for PFA).
    `dtype` selects the working precision the kernel is built with."""
    if name not in ALL:
        raise ValueError(f"unknown algorithm {name!r}; known: {ALL}")
    return (_pfa if name == "pfa" else _stripmap)(name, n, dtype)


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
