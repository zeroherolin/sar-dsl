"""CSA end-to-end: DSL vs NumPy reference, focusing quality, agreement
with the other imaging algorithms, and HLS emission.

CSA is interpolation-free (chirp scaling replaces RCMC interpolation), so
the full chain also exercises the pure phase-multiply + FFT path.
"""

import numpy as np
import pytest

from conftest import requires_cpu, requires_hls

from common.params import synthetic_params
from common.simulate import demo_scene, single_target_scene
from csa.algorithm import build_kernel as build_csa_kernel
from csa.algorithm import make_inputs as csa_inputs
from csa.reference import CSAProcessor

N = 128


@pytest.fixture(scope="module")
def setup():
    params = synthetic_params(N)
    kernel = build_csa_kernel(N, params)
    return params, kernel, csa_inputs(N, params)


@requires_cpu
def test_csa_matches_numpy_reference(setup):
    params, kernel, inputs = setup
    rng = np.random.default_rng(31)
    raw = (rng.standard_normal((N, N)) + 1j * rng.standard_normal(
        (N, N))).astype(np.complex64)

    ref = CSAProcessor(N, params).process(raw)
    out = kernel(raw, *inputs)
    peak = float(ref.max())
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6 * peak)


@requires_cpu
def test_csa_focuses_point_target(setup):
    params, kernel, inputs = setup
    raw = single_target_scene(N, params)

    image = kernel(raw, *inputs).astype(np.float64)
    i, j = np.unravel_index(np.argmax(image), image.shape)
    assert abs(i - N // 2) <= 1 and abs(j - N // 2) <= 1

    window = image[i - 3:i + 4, j - 3:j + 4]
    energy_fraction = (window**2).sum() / (image**2).sum()
    # Looser than WKA's 0.9 gate: residual chirp-scaling (bulk RCMC)
    # error leaves some energy outside the 7x7 window.
    assert energy_fraction > 0.8


@requires_cpu
def test_csa_and_wka_agree_on_point_target(setup):
    from wka.algorithm import build_kernel as build_wka_kernel
    from wka.algorithm import make_inputs as wka_inputs

    params, csa_kernel, csa_in = setup
    raw, _ = demo_scene(N, params)

    csa_img = csa_kernel(raw, *csa_in).astype(np.float64)

    wka_kernel = build_wka_kernel(N, params)
    wr, wa = wka_inputs(N, params)
    wka_img = wka_kernel(raw, wr, wa).astype(np.float64)

    csa_peak = np.unravel_index(np.argmax(csa_img), csa_img.shape)
    wka_peak = np.unravel_index(np.argmax(wka_img), wka_img.shape)
    assert abs(csa_peak[0] - wka_peak[0]) <= 1
    assert abs(csa_peak[1] - wka_peak[1]) <= 1


@requires_hls
def test_csa_emits_hls_design():
    """The full chirp-scaling chain must emit as a single design."""
    import re

    n = 64
    design = build_csa_kernel(n, synthetic_params(n)).compile(backend="hls")
    source = design.source()
    assert "void csa" in source
    assert "#pragma HLS" in source
    assert not re.findall(r"\b(malloc|free|printf|std::cout)\b", source)
