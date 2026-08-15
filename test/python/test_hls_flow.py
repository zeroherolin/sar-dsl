"""End-to-end validation of the HLS (affine) flow on complete algorithms.

Two complementary checks per algorithm:
1. numerical: the exact IR handed to ScaleHLS-the HLS pipeline is compiled for the CPU
   through `sar-affine-to-llvm-pipeline` and compared against the NumPy
   reference;
2. emission: the ScaleHLS backend produces Vitis HLS C++ for the full
   kernel (skipped when the ScaleHLS toolchain is absent).
"""

import numpy as np
import pytest

from conftest import requires_cpu, requires_hls
from conftest import compile_split_kernel as _compile_split_kernel
from conftest import run_split as _run_split

from common.params import synthetic_params
from common.simulate import demo_scene
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


@requires_hls
def test_wka_emits_hls_design(scene):
    """The headline: the complete omega-K kernel becomes one HLS design."""
    params, _ = scene
    n = 64  # keep the the HLS pipeline optimization time in check
    design = build_wka_kernel(n,
                              synthetic_params(n)).compile(backend="hls")
    source = design.source()
    assert "void wka" in source
    assert "#pragma HLS" in source


@requires_hls
def test_rda_emits_hls_design():
    from rda.algorithm import build_kernel as build_rda_kernel

    n = 64
    design = build_rda_kernel(n,
                              synthetic_params(n)).compile(backend="hls")
    source = design.source()
    assert "void rda" in source
    assert "#pragma HLS" in source
