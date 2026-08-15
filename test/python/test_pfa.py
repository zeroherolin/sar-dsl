"""PFA + SVA end-to-end: DSL kernel vs NumPy reference, focusing quality
and HLS emission.

The whole chain is built from Python-defined operators (`@sar.op`)
over built-in ops -- including SVA, a data-dependent per-pixel weight
selection written with the `sar.where` / comparison primitives -- so
this locks in that user-level composition covers algorithm logic beyond
plain op chaining and reaches both backends.
"""

import sys

import numpy as np
import pytest

from conftest import REPO_ROOT, requires_cpu, requires_hls

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from pfa.algorithm import build_kernel, make_inputs  # noqa: E402
from pfa.geometry import Geometry  # noqa: E402
from pfa.reference import PFAProcessor  # noqa: E402
from metrics import measure_cut  # noqa: E402

N = 128


@pytest.fixture(scope="module")
def setup():
    geometry = Geometry(N)
    return geometry, build_kernel(N, geometry), make_inputs(N, geometry)


@requires_cpu
def test_pfa_matches_numpy_reference(setup):
    geometry, kernel, inputs = setup
    rng = np.random.default_rng(41)
    raw = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))

    ref_uniform, ref_sva = PFAProcessor(N, geometry).process(raw)
    got_uniform, got_sva = kernel(raw, *inputs)
    peak = float(ref_uniform.max())
    np.testing.assert_allclose(got_uniform,
                               ref_uniform,
                               rtol=1e-6,
                               atol=1e-9 * peak)
    np.testing.assert_allclose(got_sva, ref_sva, rtol=1e-6, atol=1e-9 * peak)


@requires_cpu
def test_pfa_focuses_and_sva_suppresses_sidelobes(setup):
    geometry, kernel, inputs = setup
    targets = geometry.demo_targets()
    uniform, filtered = kernel(geometry.simulate(targets), *inputs)

    for x, y in targets:
        row, col = N + int(round(2 * y)), N + int(round(2 * x))
        window = filtered[row - 2:row + 3, col - 2:col + 3]
        assert window.max() == filtered[row, col], (x, y)

    # SVA must clearly beat the -13 dB uniform-weighting sidelobes
    # without widening the mainlobe.
    i, _ = np.unravel_index(np.argmax(uniform), uniform.shape)
    plain = measure_cut(uniform[i, :])
    sva = measure_cut(filtered[i, :])
    assert plain["pslr"] > -15.0
    assert sva["pslr"] < -20.0
    assert sva["irw"] < plain["irw"] * 1.05


@requires_hls
def test_pfa_emits_hls_design():
    """The full PFA + SVA chain must emit as a single design."""
    import re

    n = 32
    design = build_kernel(n, Geometry(n)).compile(backend="hls")
    source = design.source()
    assert "void pfa" in source
    assert "#pragma HLS" in source
    assert not re.findall(r"\b(malloc|free|printf|std::cout)\b", source)
