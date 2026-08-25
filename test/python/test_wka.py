"""End-to-end WKA validation: the DSL kernel must reproduce the numpy
reference implementation, and it must actually focus point targets."""

import importlib.util
import json
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT, requires_cpu

from common.params import ALOS_PARAMS, synthetic_params
from common.simulate import demo_scene, single_target_scene
from wka.algorithm import build_kernel, make_inputs
from wka.reference import WKAProcessor

N = 128


def test_handwritten_hls_reference_and_report(tmp_path, monkeypatch):
    root = REPO_ROOT / "examples" / "wka" / "handwritten_hls"
    reference_path = root / "reference" / "wka_reference.py"
    spec = importlib.util.spec_from_file_location("handwritten_wka_reference",
                                                  reference_path)
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    raw = reference.synthetic_input(64)
    image = reference.wka_reference(raw)
    assert image.shape == (64, 64)
    assert image.dtype == np.float32
    assert np.all(np.isfinite(image))

    candidate = tmp_path / "candidate.bin"
    output = tmp_path / "reference.bin"
    image.tofile(candidate)
    monkeypatch.setattr(sys, "argv", [
        str(reference_path), "--n", "64", "--output",
        str(output), "--compare",
        str(candidate)
    ])
    assert reference.main() == 0

    np.full_like(image, np.nan).tofile(candidate)
    monkeypatch.setattr(sys, "argv", [
        str(reference_path), "--n", "64", "--output",
        str(output), "--compare",
        str(candidate)
    ])
    assert reference.main() == 1

    report = json.loads(
        (root / "reports" / "production_csynth.json").read_text())
    assert report["design"]["shape"] == [16384, 16384]
    timing_budget = (report["constraints"]["clock_ns"] -
                     report["constraints"].get("clock_uncertainty_ns", 0.0))
    assert report["report"]["estimated_clock_ns"] <= (timing_budget)
    assert report["report"]["latency_min_cycles"] == report["report"][
        "latency_average_cycles"] == report["report"]["latency_max_cycles"]
    assert report["design"]["array_ports"] == 9
    assert report["design"]["axi_master_bundles"] == 8
    for resource in ("bram18k", "uram", "dsp", "ff", "lut"):
        assert report["report"][resource] <= report["constraints"][
            f"{resource}_budget"]


@pytest.fixture(scope="module")
def setup():
    params = synthetic_params(N)
    kernel = build_kernel(N, params)
    return kernel, params


@requires_cpu
def test_wka_matches_numpy_reference(setup):
    kernel, params = setup
    raw, _ = demo_scene(N, params)

    ref = WKAProcessor(N, params).process(raw)
    wr, wa = make_inputs(N, params)
    out = kernel(raw, wr, wa)

    peak = float(ref.max())
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6 * peak)


@requires_cpu
def test_wka_focuses_point_target(setup):
    kernel, params = setup
    raw = single_target_scene(N, params)

    wr, wa = make_inputs(N, params)
    image = kernel(raw, wr, wa).astype(np.float64)

    # The scene-center scatterer must focus at the raster center...
    i, j = np.unravel_index(np.argmax(image), image.shape)
    assert abs(i - N // 2) <= 1 and abs(j - N // 2) <= 1

    # ...and concentrate most of the image energy in a small neighborhood.
    window = image[i - 3:i + 4, j - 3:j + 4]
    energy_fraction = (window**2).sum() / (image**2).sum()
    assert energy_fraction > 0.9

    # The raw (unfocused) data has no such concentration.
    raw_mag = np.abs(raw.astype(np.complex128))
    ri, rj = np.unravel_index(np.argmax(raw_mag), raw_mag.shape)
    raw_window = raw_mag[max(ri - 3, 0):ri + 4, max(rj - 3, 0):rj + 4]
    raw_fraction = (raw_window**2).sum() / (raw_mag**2).sum()
    assert raw_fraction < 0.05


@requires_cpu
def test_wka_with_alos_parameters():
    """The ALOS parameter set must run through the pipeline as well (spaceborne
    geometry does not focus at N=128, so only algebraic equivalence with the
    reference is checked)."""
    n = 64
    kernel = build_kernel(n, ALOS_PARAMS)
    wr, wa = make_inputs(n, ALOS_PARAMS)

    rng = np.random.default_rng(11)
    raw = (rng.standard_normal((n, n)) + 1j * rng.standard_normal(
        (n, n))).astype(np.complex64)

    ref = WKAProcessor(n, ALOS_PARAMS).process(raw)
    out = kernel(raw, wr, wa)
    peak = float(ref.max())
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6 * peak)
