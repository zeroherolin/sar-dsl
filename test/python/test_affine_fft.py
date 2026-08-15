"""Numerical validation of the split-complex affine (Stockham) FFT path.

The `sar-affine-to-llvm-pipeline` compiles the same IR the ScaleHLS backend
hands to HIDA, so passing here validates the HLS lowering's arithmetic
without needing an HLS simulator. The decomplexified ABI splits every
complex tensor into (re, im) float planes, in order, inputs before results.
"""

import numpy as np
import pytest

from conftest import requires_cpu, compile_split_kernel, run_split

pytestmark = requires_cpu

# Backwards-compatible aliases (the helpers moved to conftest.py so other
# test modules can share them without private cross-file imports).
_compile_split_kernel = compile_split_kernel
_run_split = run_split


@pytest.mark.parametrize("n,m,dim", [(4, 8, 1), (8, 16, 0), (16, 16, 1)])
def test_affine_fft_matches_numpy(n, m, dim, tmp_path):
    mlir = f"""
func.func @k(%x: tensor<{n}x{m}xcomplex<f64>>) -> tensor<{n}x{m}xcomplex<f64>> {{
  %0 = sar.fft %x {{dim = {dim} : i64}} : tensor<{n}x{m}xcomplex<f64>>
  return %0 : tensor<{n}x{m}xcomplex<f64>>
}}
"""
    lib, fn = _compile_split_kernel(mlir, "k", tmp_path)

    rng = np.random.default_rng(9)
    x = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    re, im = np.ascontiguousarray(x.real), np.ascontiguousarray(x.imag)

    out_re, out_im = _run_split(fn, [re, im], [(n, m), (n, m)], np.float64)
    ref = np.fft.fft(x, axis=dim)
    np.testing.assert_allclose(out_re + 1j * out_im,
                               ref,
                               rtol=1e-12,
                               atol=1e-12)


def test_affine_ifft_roundtrip(tmp_path):
    n, m = 8, 32
    mlir = f"""
func.func @rt(%x: tensor<{n}x{m}xcomplex<f64>>) -> tensor<{n}x{m}xcomplex<f64>> {{
  %0 = sar.fft %x {{dim = 1 : i64}} : tensor<{n}x{m}xcomplex<f64>>
  %1 = sar.ifft %0 {{dim = 1 : i64}} : tensor<{n}x{m}xcomplex<f64>>
  return %1 : tensor<{n}x{m}xcomplex<f64>>
}}
"""
    lib, fn = _compile_split_kernel(mlir, "rt", tmp_path)

    rng = np.random.default_rng(10)
    x = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    re, im = np.ascontiguousarray(x.real), np.ascontiguousarray(x.imag)

    out_re, out_im = _run_split(fn, [re, im], [(n, m), (n, m)], np.float64)
    np.testing.assert_allclose(out_re + 1j * out_im, x, rtol=1e-12, atol=1e-12)


def test_affine_rank1_fft(tmp_path):
    n = 64
    mlir = f"""
func.func @k1(%x: tensor<{n}xcomplex<f32>>) -> tensor<{n}xcomplex<f32>> {{
  %0 = sar.fft %x {{dim = 0 : i64}} : tensor<{n}xcomplex<f32>>
  return %0 : tensor<{n}xcomplex<f32>>
}}
"""
    lib, fn = _compile_split_kernel(mlir, "k1", tmp_path)

    rng = np.random.default_rng(11)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    re = np.ascontiguousarray(x.real, dtype=np.float32)
    im = np.ascontiguousarray(x.imag, dtype=np.float32)

    out_re, out_im = _run_split(fn, [re, im], [(n, ), (n, )], np.float32)
    ref = np.fft.fft(x)
    np.testing.assert_allclose(out_re + 1j * out_im, ref, rtol=1e-4, atol=1e-3)


def test_affine_complex_access_ops(tmp_path):
    """conj/angle through the split-complex path (angle lowers to the
    sar.atan2 the HLS emitter must also handle)."""
    n, m = 8, 16
    mlir = f"""
func.func @ca(%z: tensor<{n}x{m}xcomplex<f64>>) -> tensor<{n}x{m}xf64> {{
  %0 = sar.conj %z : tensor<{n}x{m}xcomplex<f64>>
  %re = sar.real %0 : tensor<{n}x{m}xcomplex<f64>> -> tensor<{n}x{m}xf64>
  %im = sar.imag %0 : tensor<{n}x{m}xcomplex<f64>> -> tensor<{n}x{m}xf64>
  %1 = sar.atan2 %im, %re : tensor<{n}x{m}xf64>
  return %1 : tensor<{n}x{m}xf64>
}}
"""
    lib, fn = _compile_split_kernel(mlir, "ca", tmp_path)

    rng = np.random.default_rng(13)
    z = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    re, im = np.ascontiguousarray(z.real), np.ascontiguousarray(z.imag)

    (out, ) = _run_split(fn, [re, im], [(n, m)], np.float64)
    np.testing.assert_allclose(out,
                               np.angle(np.conj(z)),
                               rtol=1e-12,
                               atol=1e-12)


def test_affine_full_stage_with_elementwise(tmp_path):
    """A decomplexified bulk-compression-style stage: expj multiply then
    FFT, all through the affine path."""
    n, m = 8, 16
    mlir = f"""
func.func @stage(%d: tensor<{n}x{m}xcomplex<f64>>, %p: tensor<{n}x{m}xf64>)
    -> tensor<{n}x{m}xcomplex<f64>> {{
  %pc = sar.cos %p : tensor<{n}x{m}xf64>
  %ps = sar.sin %p : tensor<{n}x{m}xf64>
  %0 = sar.complex %pc, %ps : tensor<{n}x{m}xf64> -> tensor<{n}x{m}xcomplex<f64>>
  %1 = sar.mul %d, %0 : tensor<{n}x{m}xcomplex<f64>>
  %2 = sar.fft %1 {{dim = 1 : i64}} : tensor<{n}x{m}xcomplex<f64>>
  return %2 : tensor<{n}x{m}xcomplex<f64>>
}}
"""
    lib, fn = _compile_split_kernel(mlir, "stage", tmp_path)

    rng = np.random.default_rng(12)
    d = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    p = rng.standard_normal((n, m))
    re, im = np.ascontiguousarray(d.real), np.ascontiguousarray(d.imag)

    out_re, out_im = _run_split(fn, [re, im, np.ascontiguousarray(p)],
                                [(n, m), (n, m)], np.float64)
    ref = np.fft.fft(d * np.exp(1j * p), axis=1)
    np.testing.assert_allclose(out_re + 1j * out_im,
                               ref,
                               rtol=1e-12,
                               atol=1e-12)


@pytest.mark.parametrize("n", [3, 5, 6, 12, 15, 33, 100])
def test_affine_bluestein_matches_numpy(n, tmp_path):
    """Non-power-of-two sizes take the Bluestein path in the affine (HLS)
    lowering, matching numpy to double precision.

    The chirp and the kernel spectrum are folded at compile time, so the
    emitted code is two padded Stockham transforms plus element-wise
    passes -- everything affine, nothing size-dependent at runtime.
    """
    lines = 4
    mlir = f"""
module {{
  func.func @bz(%re: tensor<{lines}x{n}xf64>, %im: tensor<{lines}x{n}xf64>)
      -> (tensor<{lines}x{n}xf64>, tensor<{lines}x{n}xf64>) {{
    %r, %i = sar.fft_split %re, %im {{dim = 1 : i64}} : tensor<{lines}x{n}xf64>
    return %r, %i : tensor<{lines}x{n}xf64>, tensor<{lines}x{n}xf64>
  }}
}}"""
    lib, fn = _compile_split_kernel(mlir, "bz", tmp_path)
    rng = np.random.default_rng(n)
    re = rng.standard_normal((lines, n))
    im = rng.standard_normal((lines, n))
    out_re, out_im = _run_split(fn, [re, im], [(lines, n)] * 2, np.float64)

    ref = np.fft.fft(re + 1j * im, axis=1)
    np.testing.assert_allclose(out_re + 1j * out_im,
                               ref,
                               rtol=1e-11,
                               atol=1e-11)


def test_affine_bluestein_scaling_is_exact(tmp_path):
    """A unit impulse transforms to all-ones.

    This pins the Bluestein scaling specifically: `emitStockham` emits the
    butterflies only (its `inverse` flag picks the twiddle conjugation, it
    does not scale), so the 1/M the convolution theorem needs has to be
    applied by the caller. Getting that wrong leaves the whole spectrum
    off by a factor of M with the phases still correct -- which an
    impulse makes obvious and a random-input tolerance check can hide.
    """
    n = 6
    mlir = f"""
module {{
  func.func @imp(%re: tensor<1x{n}xf64>, %im: tensor<1x{n}xf64>)
      -> (tensor<1x{n}xf64>, tensor<1x{n}xf64>) {{
    %r, %i = sar.fft_split %re, %im {{dim = 1 : i64}} : tensor<1x{n}xf64>
    return %r, %i : tensor<1x{n}xf64>, tensor<1x{n}xf64>
  }}
}}"""
    lib, fn = _compile_split_kernel(mlir, "imp", tmp_path)
    re = np.zeros((1, n))
    re[0, 0] = 1.0
    out_re, out_im = _run_split(fn, [re, np.zeros((1, n))], [(1, n)] * 2,
                                np.float64)
    np.testing.assert_allclose(out_re, np.ones((1, n)), atol=1e-12)
    np.testing.assert_allclose(out_im, np.zeros((1, n)), atol=1e-12)


def test_affine_bluestein_roundtrip(tmp_path):
    """ifft(fft(x)) == x through the Bluestein path (both directions)."""
    n, lines = 12, 2
    mlir = f"""
module {{
  func.func @rt(%re: tensor<{lines}x{n}xf64>, %im: tensor<{lines}x{n}xf64>)
      -> (tensor<{lines}x{n}xf64>, tensor<{lines}x{n}xf64>) {{
    %fr, %fi = sar.fft_split %re, %im {{dim = 1 : i64}}
        : tensor<{lines}x{n}xf64>
    %r, %i = sar.fft_split %fr, %fi {{dim = 1 : i64, inverse}}
        : tensor<{lines}x{n}xf64>
    return %r, %i : tensor<{lines}x{n}xf64>, tensor<{lines}x{n}xf64>
  }}
}}"""
    lib, fn = _compile_split_kernel(mlir, "rt", tmp_path)
    rng = np.random.default_rng(7)
    re = rng.standard_normal((lines, n))
    im = rng.standard_normal((lines, n))
    out_re, out_im = _run_split(fn, [re, im], [(lines, n)] * 2, np.float64)
    np.testing.assert_allclose(out_re, re, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(out_im, im, rtol=1e-11, atol=1e-11)
