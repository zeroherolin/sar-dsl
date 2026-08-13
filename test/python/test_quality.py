"""Point-target image-quality gates for all imaging algorithms.

Locks in the focusing quality measured by `benchmarks/point_target_quality`:
impulse response width, peak sidelobe ratio and peak position. Bounds are
set with margin below the measured values (IRW 1.62 samples, PSLR ~-26 dB
with Hanning windows on a 70%-bandwidth chirp).
"""

import sys

import numpy as np
import pytest

from conftest import REPO_ROOT, requires_cpu

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from common.params import synthetic_params       # noqa: E402
from common.simulate import single_target_scene  # noqa: E402
from point_target_quality import measure_image   # noqa: E402

pytestmark = requires_cpu

N = 256

_ALGORITHMS = ["wka", "rda", "csa"]


@pytest.fixture(scope="module")
def scene():
    params = synthetic_params(N)
    return params, single_target_scene(N, params)


@pytest.mark.parametrize("name", _ALGORITHMS)
def test_point_target_quality(name, scene):
    params, raw = scene
    if name == "wka":
        from wka.algorithm import build_kernel, make_inputs
    elif name == "rda":
        from rda.algorithm import build_kernel, make_inputs
    else:
        from csa.algorithm import build_kernel, make_inputs

    kernel = build_kernel(N, params).compile("cpu")
    image = kernel(raw, *make_inputs(N, params)).astype(np.float64)
    metrics = measure_image(image, expected_peak=(N // 2, N // 2))

    assert metrics["peak_error"] == (0, 0), metrics
    for axis in ("range", "azimuth"):
        cut = metrics[axis]
        assert cut["irw"] < 2.0, (name, axis, cut)
        assert cut["pslr"] < -20.0, (name, axis, cut)
        assert cut["islr"] < -15.0, (name, axis, cut)
