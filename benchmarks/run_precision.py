#!/usr/bin/env python3
"""Precision benchmark: what single precision costs an image.

Focuses the same point target through each chain twice -- once with the
spectral processing in f64, once built with `dtype=sar.c64` -- and
reports the image-quality metrics side by side. Declared dtypes fix the
data path on every backend, so this is the number that says whether a
design can halve its memory and its arithmetic without losing
resolution. What narrows, and why geometry stays double, is described in
benchmarks/README.md ("Precision"); the `c128/c64 refs` column reports
textual 2-D complex-type references in the traced IR.

Usage:
    python benchmarks/run_precision.py [--n 512] [--algs wka rda csa pfa]
"""

import argparse
import collections
import importlib
import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO / "examples"))
sys.path.insert(0, str(_REPO / "python"))

import sar  # noqa: E402

from algorithms import ALL, LABELS  # noqa: E402
from metrics import measure_image  # noqa: E402


def _geometry(name: str, n: int):
    """The second `build_kernel` argument for a chain at size `n`."""
    if name == "pfa":
        from pfa.geometry import Geometry
        return Geometry(n)
    from common.params import synthetic_params
    return synthetic_params(n)


def _build(name: str, n: int, single: bool):
    """(kernel, mode label) for a chain at the requested precision.

    The mode is read back from the built kernel's own signature: the
    stripmap chains narrow their window/axis vectors along with the data
    (`c64+f32`), PFA keeps its collection axes f64 because they feed
    `interp1d` positions (`c64 only`).
    """
    algorithm = importlib.import_module(f"{name}.algorithm")
    dtype = sar.c64 if single else sar.c128
    kernel = algorithm.build_kernel(n,
                                    _geometry(name, n),
                                    name=f"{name}_{dtype.dtype.name}",
                                    dtype=dtype)
    if not single:
        return kernel, None
    vectors = [t for t in kernel.arg_types if len(t.shape) == 1]
    narrowed = any(t.dtype.name == "f32" for t in vectors)
    return kernel, "c64+f32" if narrowed else "c64 only"


def complex_type_references(ir: str) -> dict:
    """Counts textual 2-D complex-type references in narrowed IR.

    Geometry may deliberately compute a c128 phase factor and cast it to
    c64 before touching the data. This is a composition indicator, not a
    count of distinct allocations; cross-backend accuracy decides whether
    the resulting data path meets its precision contract.
    """
    planes = re.findall(r"tensor<\d+x\d+x(complex<f64>|complex<f32>|f64|f32)>",
                        ir)
    return collections.Counter(planes)


def narrow_ir(name: str, n: int = 64) -> str:
    """MLIR text of the narrowed (f32) kernel, before backend lowering."""
    kernel, _ = _build(name, n, single=True)
    return kernel.to_mlir()


def _focus(name: str, n: int, single: bool):
    """Focuses one scatterer, optionally with the chain built at f32.

    Returns (image, mode); PFA produces two images and the uniform one is
    taken.  `mode` is None for the f64 baseline.
    """
    kernel, mode = _build(name, n, single)
    compiled = kernel.compile("cpu")

    geometry = _geometry(name, n)
    algorithm = importlib.import_module(f"{name}.algorithm")
    inputs = algorithm.make_inputs(n, geometry)
    if name == "pfa":
        raw = geometry.simulate(geometry.demo_targets())
    else:
        from common.simulate import single_target_scene
        raw = single_target_scene(n, geometry)

    # Match every argument to what the kernel declares; the dtype option
    # is what moved those declarations, so reading them back keeps the two
    # runs consistent without a second list of dtypes here.
    typed = [
        np.asarray(a, dtype=t.dtype.to_numpy())
        for a, t in zip([raw, *inputs], kernel.arg_types)
    ]
    result = compiled(*typed)
    if isinstance(result, (tuple, list)):
        result = result[0]
    return np.asarray(result, dtype=np.float64), mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--algs", nargs="+", default=list(ALL), choices=ALL)
    args = parser.parse_args()

    print(f"{'chain':<14} {'precision':>15} {'IRW':>7} {'PSLR/dB':>9} "
          f"{'ISLR/dB':>9} {'rel. err':>10} {'c64/c128 refs':>17}")
    print("-" * 86)
    for name in args.algs:
        reference = None
        for single in (False, True):
            image, mode = _focus(name, args.n, single)
            metrics = measure_image(image)["range"]
            if reference is None:
                reference = image
                error = 0.0
                planes = ""
            else:
                peak = reference.max()
                error = float(np.abs(image - reference).max() / peak)
                counts = complex_type_references(narrow_ir(name, args.n))
                planes = (f"{counts['complex<f32>']} / "
                          f"{counts['complex<f64>']}")
            label = f"f32 ({mode})" if single else "f64"
            print(f"{LABELS[name]:<14} {label:>15} "
                  f"{metrics['irw']:>7.2f} {metrics['pslr']:>9.2f} "
                  f"{metrics['islr']:>9.2f} {error:>10.2e} {planes:>17}")

    print("\nc64/c128 refs: textual 2-D complex-type references in the "
          "narrowed IR,\nnot distinct allocations. A c128 phase may be cast "
          "before it multiplies\nc64 data; use the accuracy columns to judge "
          "the data path.")


if __name__ == "__main__":
    main()
