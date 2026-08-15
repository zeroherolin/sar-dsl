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


def _stripmap(name: str, n: int) -> Chain:
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
    raw = single_target_scene(n, params)
    inputs = alg.make_inputs(n, params)
    return Chain(
        name,
        lambda backend="cpu", **opts: alg.build_kernel(n, params).compile(
            backend=backend, options=opts or None),
        lambda kernel: kernel(raw, *inputs),
        lambda: ref(n, params).process(raw))


def _pfa(name: str, n: int) -> Chain:
    from pfa.algorithm import build_kernel, make_inputs
    from pfa.geometry import Geometry
    from pfa.reference import PFAProcessor

    geometry = Geometry(n)
    raw = geometry.simulate(geometry.demo_targets())
    inputs = make_inputs(n, geometry)
    return Chain(
        name,
        lambda backend="cpu", **opts: build_kernel(n, geometry).compile(
            backend=backend, options=opts or None),
        lambda kernel: kernel(raw, *inputs),
        lambda: PFAProcessor(n, geometry).process(raw))


def load(name: str, n: int) -> Chain:
    """Sets up `name` for an `n`-sized scene (polar-grid edge for PFA)."""
    if name not in ALL:
        raise ValueError(f"unknown algorithm {name!r}; known: {ALL}")
    return (_pfa if name == "pfa" else _stripmap)(name, n)


def focus_point_target(n: int) -> List[Tuple[str, np.ndarray]]:
    """Focuses a single scene-center scatterer with every stripmap chain."""
    images = []
    for name in STRIPMAP:
        chain = load(name, n)
        kernel = chain.compile_kernel()
        images.append((name, np.asarray(chain.run(kernel), dtype=np.float64)))
    return images
