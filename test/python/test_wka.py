"""End-to-end WKA validation: the DSL kernel must reproduce the numpy
reference implementation, and it must actually focus point targets."""

import numpy as np
import pytest

from conftest import requires_cpu

from synthetic import demo_scene, single_target_scene, synthetic_params
from wka_dsl import build_wka_kernel, make_kernel_inputs
from wka_numpy import ALOS_PARAMS, WKAProcessor

pytestmark = requires_cpu

N = 128


@pytest.fixture(scope="module")
def compiled_kernel():
    params = synthetic_params(N)
    kernel = build_wka_kernel(N, params)
    return kernel, params


def test_wka_matches_numpy_reference(compiled_kernel):
    kernel, params = compiled_kernel
    raw, _ = demo_scene(N, params)

    ref = WKAProcessor(N, params).process(raw)
    fa, fr, wr, wa = make_kernel_inputs(N, params)
    out = kernel(raw, fa, fr, wr, wa)

    peak = float(ref.max())
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6 * peak)


def test_wka_focuses_point_target(compiled_kernel):
    kernel, params = compiled_kernel
    raw = single_target_scene(N, params)

    fa, fr, wr, wa = make_kernel_inputs(N, params)
    image = kernel(raw, fa, fr, wr, wa).astype(np.float64)

    # The scene-center scatterer must focus at the raster center...
    i, j = np.unravel_index(np.argmax(image), image.shape)
    assert abs(i - N // 2) <= 1 and abs(j - N // 2) <= 1

    # ...and concentrate most of the image energy in a small neighborhood.
    window = image[i - 3:i + 4, j - 3:j + 4]
    energy_fraction = (window ** 2).sum() / (image ** 2).sum()
    assert energy_fraction > 0.9

    # The raw (unfocused) data has no such concentration.
    raw_mag = np.abs(raw.astype(np.complex128))
    ri, rj = np.unravel_index(np.argmax(raw_mag), raw_mag.shape)
    raw_window = raw_mag[max(ri - 3, 0):ri + 4, max(rj - 3, 0):rj + 4]
    raw_fraction = (raw_window ** 2).sum() / (raw_mag ** 2).sum()
    assert raw_fraction < 0.05


def test_wka_with_alos_parameters():
    """The ALOS parameter set must run through the pipeline as well (spaceborne
    geometry does not focus at N=128, so only algebraic equivalence with the
    reference is checked)."""
    n = 64
    kernel = build_wka_kernel(n, ALOS_PARAMS)
    fa, fr, wr, wa = make_kernel_inputs(n, ALOS_PARAMS)

    rng = np.random.default_rng(11)
    raw = (rng.standard_normal((n, n))
           + 1j * rng.standard_normal((n, n))).astype(np.complex64)

    ref = WKAProcessor(n, ALOS_PARAMS).process(raw)
    out = kernel(raw, fa, fr, wr, wa)
    peak = float(ref.max())
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6 * peak)
