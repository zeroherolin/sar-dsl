"""RDA end-to-end: DSL kernel vs NumPy reference, plus focusing quality.

RDA shares no algorithm-specific compiler op: RCMC is composed from
`sar.interp1d` + element-wise ops, which is exactly the generalization this
test locks in.
"""

import numpy as np
import pytest

from conftest import requires_cpu, requires_hls

from common.params import synthetic_params
from common.simulate import single_target_scene
from rda.algorithm import build_kernel as build_rda_kernel
from rda.algorithm import make_inputs as rda_inputs
from rda.reference import RDAProcessor

N = 128


@pytest.fixture(scope="module")
def setup():
    params = synthetic_params(N)
    kernel = build_rda_kernel(N, params)
    inputs = rda_inputs(N, params)
    return params, kernel, inputs


@requires_cpu
def test_rda_matches_numpy_reference(setup):
    params, kernel, inputs = setup
    rng = np.random.default_rng(21)
    raw = (rng.standard_normal((N, N)) + 1j * rng.standard_normal(
        (N, N))).astype(np.complex64)

    ref = RDAProcessor(N, params).process(raw)
    out = kernel(raw, *inputs)

    peak = float(ref.max())
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6 * peak)


@requires_cpu
def test_rda_focuses_point_target(setup):
    params, kernel, inputs = setup
    raw = single_target_scene(N, params)

    image = kernel(raw, *inputs).astype(np.float64)

    i, j = np.unravel_index(np.argmax(image), image.shape)
    assert abs(i - N // 2) <= 1 and abs(j - N // 2) <= 1

    window = image[i - 3:i + 4, j - 3:j + 4]
    energy_fraction = (window**2).sum() / (image**2).sum()
    # Looser than WKA's 0.9 gate: residual RCMC interpolation error
    # leaves some energy outside the 7x7 window.
    assert energy_fraction > 0.8


@requires_cpu
def test_rda_and_wka_agree_on_point_target(setup):
    """Both imaging algorithms must place the same scatterer at the same
    pixel with comparable focusing quality."""
    from wka.algorithm import build_kernel as build_wka_kernel
    from wka.algorithm import make_inputs as wka_inputs

    params, rda_kernel, inputs = setup
    raw = single_target_scene(N, params)

    rda_img = rda_kernel(raw, *inputs).astype(np.float64)

    wka_kernel = build_wka_kernel(N, params)
    wr, wa = wka_inputs(N, params)
    wka_img = wka_kernel(raw, wr, wa).astype(np.float64)

    rda_peak = np.unravel_index(np.argmax(rda_img), rda_img.shape)
    wka_peak = np.unravel_index(np.argmax(wka_img), wka_img.shape)
    assert abs(rda_peak[0] - wka_peak[0]) <= 1
    assert abs(rda_peak[1] - wka_peak[1]) <= 1


@requires_hls
def test_rda_emits_hls_design():
    """The full range-Doppler chain must emit as a single design."""
    import re

    kernel = build_rda_kernel(N, synthetic_params(N))
    design = kernel.compile(backend="hls")
    source = design.source()
    assert "void rda" in source
    assert "#pragma HLS" in source
    assert not re.findall(r"\b(malloc|free|printf|std::cout)\b", source)
