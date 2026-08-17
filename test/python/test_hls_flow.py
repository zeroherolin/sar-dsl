"""End-to-end numerical validation of the HLS (affine) flow.

The exact IR handed to the HLS pipeline is compiled for the CPU through
`sar-affine-to-llvm-pipeline` and compared against the NumPy reference,
for every complete algorithm (wka, rda, csa, pfa). HLS *emission* tests
live in the per-algorithm files (test_wka.py, test_rda.py, ...).
"""

import numpy as np
import pytest

from conftest import requires_cpu
from conftest import compile_split_kernel as _compile_split_kernel
from conftest import run_split as _run_split

from common.params import synthetic_params
from common.simulate import demo_scene
from csa.algorithm import build_kernel as build_csa_kernel
from csa.algorithm import make_inputs as csa_inputs
from csa.reference import CSAProcessor
from pfa.algorithm import build_kernel as build_pfa_kernel
from pfa.geometry import Geometry
from pfa.reference import PFAProcessor
from rda.algorithm import build_kernel as build_rda_kernel
from rda.algorithm import make_inputs as rda_inputs
from rda.reference import RDAProcessor
from wka.algorithm import build_kernel as build_wka_kernel
from wka.algorithm import make_inputs as wka_inputs
from wka.reference import WKAProcessor

N = 128


@pytest.fixture(scope="module")
def scene():
    params = synthetic_params(N)
    raw, _ = demo_scene(N, params)
    return params, raw


# --------------------------------------------------------------------- #
# Interpolation ops through the affine path
# --------------------------------------------------------------------- #


@requires_cpu
@pytest.mark.parametrize("attrs,kwargs", [
    ("", {}),
    ('{kernel = "nearest"}', dict(kernel="nearest")),
    ('{kernel = "linear"}', dict(kernel="linear")),
    ('{kernel = "cubic"}', dict(kernel="cubic")),
    ('{taps = 16 : i64, window = "kaiser", beta = 4.0 : f64}',
     dict(taps=16, window="kaiser", beta=4.0)),
    ('{boundary = "edge"}', dict(boundary="edge")),
    ('{boundary = "reflect"}', dict(boundary="reflect")),
    ('{kernel = "nearest", boundary = "edge"}',
     dict(kernel="nearest", boundary="edge")),
    ('{kernel = "nearest", boundary = "reflect"}',
     dict(kernel="nearest", boundary="reflect")),
    ('{kernel = "linear", boundary = "edge"}',
     dict(kernel="linear", boundary="edge")),
    ('{kernel = "cubic", boundary = "reflect"}',
     dict(kernel="cubic", boundary="reflect")),
])
def test_affine_interp1d_matches_runtime(tmp_path, attrs, kwargs):
    n, m = 8, 32
    mlir = f"""
func.func @ip(%d: tensor<{n}x{m}xcomplex<f64>>, %p: tensor<{n}x{m}xf64>)
    -> tensor<{n}x{m}xcomplex<f64>> {{
  %0 = sar.interp1d %d, %p {attrs}
      : (tensor<{n}x{m}xcomplex<f64>>, tensor<{n}x{m}xf64>)
      -> (tensor<{n}x{m}xcomplex<f64>>)
  return %0 : tensor<{n}x{m}xcomplex<f64>>
}}
"""
    lib, fn = _compile_split_kernel(mlir,
                                    "ip",
                                    tmp_path,
                                    pipeline="--sar-affine-to-llvm-pipeline")

    rng = np.random.default_rng(6)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    positions = rng.uniform(-2.0, m + 1.0, size=(n, m))
    re, im = np.ascontiguousarray(data.real), np.ascontiguousarray(data.imag)

    out_re, out_im = _run_split(fn, [re, im, positions], [(n, m), (n, m)],
                                np.float64)

    import sar

    @sar.func
    def runtime_kernel(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
        return sar.interp1d(d, p, **kwargs)

    want = runtime_kernel(data, positions)
    np.testing.assert_allclose(out_re + 1j * out_im,
                               want,
                               rtol=1e-11,
                               atol=1e-11)


# --------------------------------------------------------------------- #
# Complete algorithms
# --------------------------------------------------------------------- #


@requires_cpu
def test_wka_affine_ir_matches_numpy(scene, tmp_path):
    """The full omega-K chain in HLS-flavored IR, executed on CPU."""
    params, raw = scene
    kernel = build_wka_kernel(N, params)
    lib, fn = _compile_split_kernel(kernel.to_mlir(),
                                    "wka",
                                    tmp_path,
                                    pipeline="--sar-affine-to-llvm-pipeline")

    wr, wa = wka_inputs(N, params)
    re = np.ascontiguousarray(raw.real)
    im = np.ascontiguousarray(raw.imag)
    (out, ) = _run_split(fn, [re, im, wr, wa], [(N, N)], np.float32)

    ref = WKAProcessor(N, params).process(raw)
    peak = float(ref.max())
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6 * peak)


@requires_cpu
def test_rda_affine_ir_matches_numpy(scene, tmp_path):
    """The full range-Doppler chain in HLS-flavored IR, executed on CPU."""
    params, raw = scene
    kernel = build_rda_kernel(N, params)
    lib, fn = _compile_split_kernel(kernel.to_mlir(),
                                    "rda",
                                    tmp_path,
                                    pipeline="--sar-affine-to-llvm-pipeline")

    range_ref, fa, tau, win_a = rda_inputs(N, params)
    re = np.ascontiguousarray(raw.real)
    im = np.ascontiguousarray(raw.imag)
    # Decomplexify splits each complex argument into adjacent re/im planes.
    rr_re = np.ascontiguousarray(range_ref.real)
    rr_im = np.ascontiguousarray(range_ref.imag)
    (out, ) = _run_split(fn, [re, im, rr_re, rr_im, fa, tau, win_a], [(N, N)],
                         np.float32)

    ref = RDAProcessor(N, params).process(raw)
    peak = float(ref.max())
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6 * peak)


@requires_cpu
def test_csa_affine_ir_matches_numpy(scene, tmp_path):
    """The full chirp-scaling chain in HLS-flavored IR, executed on CPU."""
    params, raw = scene
    kernel = build_csa_kernel(N, params)
    lib, fn = _compile_split_kernel(kernel.to_mlir(),
                                    "csa",
                                    tmp_path,
                                    pipeline="--sar-affine-to-llvm-pipeline")

    fa, fr, tau, win_r, win_a = csa_inputs(N, params)
    re = np.ascontiguousarray(raw.real)
    im = np.ascontiguousarray(raw.imag)
    (out, ) = _run_split(fn, [re, im, fa, fr, tau, win_r, win_a], [(N, N)],
                         np.float32)

    ref = CSAProcessor(N, params).process(raw)
    peak = float(ref.max())
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6 * peak)


@requires_cpu
def test_pfa_affine_ir_matches_numpy(tmp_path):
    """The full PFA + SVA chain in HLS-flavored IR, executed on CPU."""
    n = 64  # PFA is the heaviest chain (2n x 2n image); keep it snappy
    geometry = Geometry(n)
    kernel = build_pfa_kernel(n, geometry)
    lib, fn = _compile_split_kernel(kernel.to_mlir(),
                                    "pfa",
                                    tmp_path,
                                    pipeline="--sar-affine-to-llvm-pipeline")

    rng = np.random.default_rng(42)
    raw = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    re = np.ascontiguousarray(raw.real)
    im = np.ascontiguousarray(raw.imag)
    out_uniform, out_sva = _run_split(fn, [re, im], [(2 * n, 2 * n),
                                                     (2 * n, 2 * n)],
                                      np.float64)

    ref_uniform, ref_sva = PFAProcessor(n, geometry).process(raw)
    peak = float(ref_uniform.max())
    np.testing.assert_allclose(out_uniform,
                               ref_uniform,
                               rtol=1e-6,
                               atol=1e-9 * peak)
    np.testing.assert_allclose(out_sva, ref_sva, rtol=1e-6, atol=1e-9 * peak)
