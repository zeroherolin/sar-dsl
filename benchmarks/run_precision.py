#!/usr/bin/env python3
"""Precision benchmark: what single precision costs an image.

Focuses the same point target through each chain twice -- once with the
spectral processing in f64, once in f32 -- and reports the image-quality
metrics side by side. Declared dtypes fix the data path on every backend,
so this is the number that says whether a design can halve its memory and
its arithmetic without losing resolution.

Only the data path narrows. Geometry -- the frequency axes a chain
computes on the host -- stays double: it enters the interpolation
positions, where the error is a fraction of a resampling bin rather than
of a sample value, and omega-K loses about 4 dB of PSLR when it narrows
too.

Usage:
    python benchmarks/run_precision.py [--n 512] [--algs wka rda csa]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms import LABELS, STRIPMAP  # noqa: E402
from metrics import measure_image  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _focus(name: str, n: int, single: bool) -> np.ndarray:
    """Focuses one scatterer, optionally with the chain narrowed to f32.

    The narrowing is a source rewrite rather than an option because the
    working precision is part of the kernel's declared type -- which is
    the contract the backends honour.
    """
    source = (_REPO / "examples" / name / "algorithm.py").read_text()
    if single:
        # The chains take c64 echoes and widen to c128 for the spectral
        # work; narrowing means dropping that widening. Window arguments
        # follow so the kernel stays single-precision throughout.
        source = (source.replace("sar.cast(raw, sar.c128)", "raw").replace(
            "sar.f64[N]", "sar.f32[N]"))
    namespace = {"__name__": f"{name}_precision"}
    exec(compile(source, f"<{name}:f32>" if single else f"<{name}>", "exec"),
         namespace)

    from common.params import synthetic_params
    from common.simulate import single_target_scene

    params = synthetic_params(n)
    raw = single_target_scene(n, params)
    inputs = namespace["make_inputs"](n, params)
    kernel = namespace["build_kernel"](n, params).compile("cpu")
    # Match every argument to what the kernel declares; the rewrite above
    # is what moved those declarations, so reading them back keeps the two
    # runs consistent without a second list of dtypes here.
    arrays = [raw, *inputs]
    typed = [
        np.asarray(a, dtype=t.dtype.to_numpy())
        for a, t in zip(arrays, kernel.arg_types)
    ]
    return np.asarray(kernel(*typed), dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--algs",
                        nargs="+",
                        default=list(STRIPMAP),
                        choices=STRIPMAP)
    args = parser.parse_args()

    print(f"{'chain':<14} {'precision':>10} {'IRW':>7} {'PSLR/dB':>9} "
          f"{'ISLR/dB':>9} {'rel. err':>10}")
    print("-" * 62)
    for name in args.algs:
        reference = None
        for single in (False, True):
            image = _focus(name, args.n, single)
            metrics = measure_image(image)["range"]
            if reference is None:
                reference = image
                error = 0.0
            else:
                peak = reference.max()
                error = float(np.abs(image - reference).max() / peak)
            print(f"{LABELS[name]:<14} {'f32' if single else 'f64':>10} "
                  f"{metrics['irw']:>7.2f} {metrics['pslr']:>9.2f} "
                  f"{metrics['islr']:>9.2f} {error:>10.2e}")


if __name__ == "__main__":
    main()
