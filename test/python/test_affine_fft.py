"""Numerical validation of the split-complex affine (Stockham) FFT path.

The `sar-affine-to-llvm-pipeline` compiles the same IR the ScaleHLS backend
hands to HIDA, so passing here validates the HLS lowering's arithmetic
without needing an HLS simulator. The decomplexified ABI splits every
complex tensor into (re, im) float planes, in order, inputs before results.
"""

import ctypes
import subprocess

import numpy as np
import pytest

from sar.compiler.toolchain import find_tool
from sar.runtime import _make_descriptor

from conftest import requires_cpu

pytestmark = requires_cpu


def _compile_split_kernel(mlir_text: str, name: str, tmp_path,
                          pipeline: str = "--sar-affine-to-llvm-pipeline"):
    """Compiles a module through the split-complex affine path into a
    shared library and returns the `_mlir_ciface_<name>` symbol."""
    llvm_mlir = subprocess.run(
        [find_tool("sar-opt"), pipeline, "-"],
        input=mlir_text, capture_output=True, text=True, check=True).stdout
    llvm_ir = subprocess.run(
        [find_tool("mlir-translate"), "--mlir-to-llvmir", "-"],
        input=llvm_mlir, capture_output=True, text=True, check=True).stdout
    ll = tmp_path / "kernel.ll"
    ll.write_text(llvm_ir)
    so = tmp_path / "kernel.so"
    subprocess.run(
        [find_tool("clang"), "-O2", "-shared", "-fPIC", str(ll), "-o",
         str(so), "-lm", "-Wno-override-module"], check=True)
    lib = ctypes.CDLL(str(so))
    fn = getattr(lib, f"_mlir_ciface_{name}")
    fn.restype = None
    return lib, fn


def _run_split(fn, inputs, out_shapes, dtype):
    outs = [np.empty(s, dtype=dtype) for s in out_shapes]
    descriptors = [_make_descriptor(a) for a in list(inputs) + outs]
    fn(*[ctypes.byref(d) for d in descriptors])
    return outs


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
    np.testing.assert_allclose(out_re + 1j * out_im, ref, rtol=1e-12,
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
    np.testing.assert_allclose(out_re + 1j * out_im, x, rtol=1e-12,
                               atol=1e-12)


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

    out_re, out_im = _run_split(fn, [re, im], [(n,), (n,)], np.float32)
    ref = np.fft.fft(x)
    np.testing.assert_allclose(out_re + 1j * out_im, ref, rtol=1e-4,
                               atol=1e-3)


def test_affine_full_stage_with_elementwise(tmp_path):
    """A decomplexified bulk-compression-style stage: expj multiply then
    FFT, all through the affine path."""
    n, m = 8, 16
    mlir = f"""
func.func @stage(%d: tensor<{n}x{m}xcomplex<f64>>, %p: tensor<{n}x{m}xf64>)
    -> tensor<{n}x{m}xcomplex<f64>> {{
  %0 = sar.expj %p : tensor<{n}x{m}xf64> -> tensor<{n}x{m}xcomplex<f64>>
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
    np.testing.assert_allclose(out_re + 1j * out_im, ref, rtol=1e-12,
                               atol=1e-12)
