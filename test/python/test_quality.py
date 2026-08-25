"""Point-target image-quality gates for every imaging algorithm.

Locks in the focusing quality measured by `benchmarks/run_cpu_quality.py` so a
lowering or numerics regression shows up as a failing bound rather than a
slightly blurrier picture.

Bounds sit roughly 1.5 dB / 0.1 samples below the values measured at
N = 256, which is tight enough to catch a real regression and loose enough
to absorb last-bit variation between toolchains. The reference points:

  * Hann theory line is IRW = 1.44 bins / 0.70 = 2.06 samples; the chains
    measure 2.09..2.13.
  * PSLR lands near -38 dB, better than the -31.5 dB window bound because
    the LFM spectrum rolls off at the band edges.
  * The RDA *range* axis is the worst case on both PSLR (-35.0) and ISLR
    (-20.6): basic RDA has no secondary range compression, so residual
    range-azimuth coupling raises its sidelobes (see
    examples/rda/README.md). It gets its own bounds instead of loosening
    everyone else's.
  * PFA is gated on its uniform-weighting output; the SVA comparison
    (sidelobe suppression at constant mainlobe width) stays in
    test_pfa.py, next to the SVA reference.
"""

import sys

import numpy as np
import pytest

from conftest import REPO_ROOT, requires_cpu

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from algorithms import STRIPMAP, load  # noqa: E402
from metrics import measure_image  # noqa: E402

pytestmark = requires_cpu

N = 256

#: Per-axis bounds: (max IRW in samples, max PSLR dB, max ISLR dB).
_DEFAULT_BOUNDS = (2.25, -36.0, -22.0)

#: Axes that legitimately sit above the default; see the module docstring.
_BOUNDS = {
    ("rda", "range"): (2.25, -33.5, -19.0),
}


def _check(name, axis, cut):
    irw_max, pslr_max, islr_max = _BOUNDS.get((name, axis), _DEFAULT_BOUNDS)
    assert cut["irw"] < irw_max, (name, axis, cut)
    assert cut["pslr"] < pslr_max, (name, axis, cut)
    assert cut["islr"] < islr_max, (name, axis, cut)


@pytest.mark.parametrize("name", STRIPMAP)
def test_point_target_quality(name):
    chain = load(name, N)
    image = np.asarray(chain.run(chain.compile_kernel()), dtype=np.float64)
    metrics = measure_image(image, expected_peak=(N // 2, N // 2))

    assert metrics["peak_error"] == (0, 0), metrics
    for axis in ("range", "azimuth"):
        _check(name, axis, metrics[axis])


def test_pfa_point_target_quality():
    """PFA is spotlight-mode with a 2x oversampled output grid, so it
    carries its own bounds: one resolution cell spans two pixels (measured
    IRW 1.81 pixels = 0.91 cells) and uniform aperture weighting pins the
    first sidelobe at the -13.26 dB theory line."""
    chain = load("pfa", N)
    uniform, _sva = chain.run(chain.compile_kernel())
    image = np.asarray(uniform, dtype=np.float64)

    # The 2x oversampled grid is 2N x 2N, so a centered target peaks at (N, N).
    metrics = measure_image(image, expected_peak=(N, N))
    assert metrics["peak_error"] == (0, 0), metrics

    for axis in ("range", "azimuth"):
        cut = metrics[axis]
        assert cut["irw"] < 2.0, (axis, cut)
        assert cut["pslr"] < -12.5, (axis, cut)
