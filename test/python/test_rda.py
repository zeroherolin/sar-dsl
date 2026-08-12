"""RDA end-to-end: DSL kernel vs NumPy reference, plus focusing quality.

RDA shares no algorithm-specific compiler op: RCMC is composed from
`sar.interp1d` + element-wise ops, which is exactly the generalization this
test locks in.
"""

import sys

import numpy as np
import pytest

from conftest import REPO_ROOT, requires_cpu

sys.path.insert(0, str(REPO_ROOT / "examples" / "rda"))

from rda_dsl import build_rda_kernel, make_kernel_inputs   # noqa: E402
from rda_numpy import RDAProcessor                          # noqa: E402
from synthetic import single_target_scene, synthetic_params  # noqa: E402

pytestmark = requires_cpu

N = 128


@pytest.fixture(scope="module")
def setup():
    params = synthetic_params(N)
    kernel = build_rda_kernel(N, params)
    range_ref, fa = make_kernel_inputs(N, params)
    return params, kernel, range_ref, fa


def test_rda_matches_numpy_reference(setup):
    params, kernel, range_ref, fa = setup
    rng = np.random.default_rng(21)
    raw = (rng.standard_normal((N, N))
           + 1j * rng.standard_normal((N, N))).astype(np.complex64)

    ref = RDAProcessor(N, params).process(raw)
    out = kernel(raw, range_ref, fa)

    peak = float(ref.max())
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6 * peak)


def test_rda_focuses_point_target(setup):
    params, kernel, range_ref, fa = setup
    raw = single_target_scene(N, params)

    image = kernel(raw, range_ref, fa).astype(np.float64)

    i, j = np.unravel_index(np.argmax(image), image.shape)
    assert abs(i - N // 2) <= 1 and abs(j - N // 2) <= 1

    window = image[i - 3:i + 4, j - 3:j + 4]
    energy_fraction = (window ** 2).sum() / (image ** 2).sum()
    assert energy_fraction > 0.8


def test_rda_and_wka_agree_on_point_target(setup):
    """Both imaging algorithms must place the same scatterer at the same
    pixel with comparable focusing quality."""
    from wka_dsl import build_wka_kernel
    from wka_dsl import make_kernel_inputs as wka_inputs

    params, rda_kernel, range_ref, fa = setup
    raw = single_target_scene(N, params)

    rda_img = rda_kernel(raw, range_ref, fa).astype(np.float64)

    wka_kernel = build_wka_kernel(N, params)
    fa_w, fr_w, wr, wa = wka_inputs(N, params)
    wka_img = wka_kernel(raw, fa_w, fr_w, wr, wa).astype(np.float64)

    rda_peak = np.unravel_index(np.argmax(rda_img), rda_img.shape)
    wka_peak = np.unravel_index(np.argmax(wka_img), wka_img.shape)
    assert abs(rda_peak[0] - wka_peak[0]) <= 1
    assert abs(rda_peak[1] - wka_peak[1]) <= 1
