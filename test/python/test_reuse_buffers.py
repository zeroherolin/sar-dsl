"""What buffer sharing must not change: the answers.

Which allocations `sar-reuse-buffers` collapses, and into what, is pinned by
`test/Dialect/SAR/reuse-buffers.mlir`. These tests cover the part FileCheck
cannot reach -- running the two forms and comparing them -- plus the pipeline
property that the affine hand-off never emits a `memref.view`, which the C++
emitter has no way to express.
"""

import re
import subprocess

import numpy as np

import sar

from conftest import requires_cpu

pytestmark = requires_cpu

RNG = np.random.default_rng(20260815)


def _opt(mlir_text: str, *args: str) -> str:
    """Runs sar-opt over `mlir_text` and returns the resulting IR."""
    from sar.compiler.toolchain import find_tool

    return subprocess.run([find_tool("sar-opt"), *args, "-"],
                          input=mlir_text,
                          capture_output=True,
                          text=True,
                          check=True).stdout


# A mixed-precision chain: a real plane is built and consumed, and only then
# does a complex plane of half the element count -- the same footprint in
# bytes -- come to life. Written as memref IR directly so the test states the
# lifetimes it means to test rather than depending on what bufferization
# happens to emit.
def _allocations(ir: str):
    """Every `memref.alloc` in `ir`, as the text of its result type."""
    return re.findall(
        r"memref\.alloc\(\)\s*(?:\{[^}]*\})?\s*:\s*(memref<.*?>)\s*$", ir,
        re.MULTILINE)


def test_affine_path_never_retypes():
    """The HLS hand-off keeps every buffer typed: an on-chip array's element
    width drives banking and partitioning, and the C++ emitter has no
    reinterpretation to emit. A `memref.view` reaching it is a hard error,
    so the affine pipeline must not produce one."""

    @sar.func
    def chain(x: sar.c64[64, 64]) -> sar.c64[64, 64]:
        return sar.ifft(sar.fft(x * 2.0, axis=1) + 1.0, axis=1)

    ir = _opt(chain.to_mlir(), "--sar-to-affine-pipeline")
    assert "memref.view" not in ir


@requires_cpu
def test_mixed_precision_kernel_is_numerically_unchanged():
    """The whole point of the conservative side of this pass: sharing a
    buffer must not change an answer. A chain that mixes an f32 stage with a
    c64 stage runs through the CPU pipeline, which shares across types, and
    is checked against numpy."""
    N = 64

    @sar.func
    def kernel(x: sar.f32[N]) -> sar.c64[N]:
        scaled = x * 2.0
        spectrum = sar.fft(sar.cast(scaled, sar.c64), axis=0)
        return sar.ifft(spectrum * 3.0, axis=0)

    x = RNG.standard_normal(N).astype(np.float32)
    scaled = (x * np.float32(2.0)).astype(np.complex64)
    reference = np.fft.ifft(np.fft.fft(scaled) * 3.0)

    np.testing.assert_allclose(kernel(x), reference, rtol=1e-4, atol=1e-4)


@requires_cpu
def test_sharing_is_bit_exact_against_disabled_sharing(tmp_path):
    """Differential gate over the scariest rewrite the pass performs: two
    buffers of different element types retire into one byte-addressed
    backing viewed as both. The same module is compiled and executed with
    and without the rewrite, and the results must be bit-identical --
    sharing decides where values live, never what they are, so any
    divergence is an aliasing bug."""
    from conftest import compile_split_kernel

    module = """
func.func @diff(%in: memref<64xf32>, %tmp: memref<64xf32>,
                %out: memref<32xf64>) {
  %a = memref.alloc() : memref<64xf32>
  %b = memref.alloc() : memref<32xf64>
  affine.for %i = 0 to 64 {
    %v = affine.load %in[%i] : memref<64xf32>
    %s = arith.mulf %v, %v : f32
    affine.store %s, %a[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %a[%i] : memref<64xf32>
    affine.store %v, %tmp[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 32 {
    %v0 = affine.load %tmp[2 * %i] : memref<64xf32>
    %v1 = affine.load %tmp[2 * %i + 1] : memref<64xf32>
    %s = arith.addf %v0, %v1 : f32
    %w = arith.extf %s : f32 to f64
    affine.store %w, %b[%i] : memref<32xf64>
  }
  affine.for %i = 0 to 32 {
    %v = affine.load %b[%i] : memref<32xf64>
    %d = arith.mulf %v, %v : f64
    affine.store %d, %out[%i] : memref<32xf64>
  }
  return
}
"""
    shared = _opt(module,
                  "--sar-reuse-buffers=min-elements=0 allow-retype=true")
    # The gate is only meaningful when the rewrite actually fired: the two
    # typed allocations must have collapsed into one byte backing.
    assert _allocations(shared) == ["memref<256xi8>"]
    assert "memref.view" in shared

    rng = np.random.default_rng(7)
    x = rng.standard_normal(64).astype(np.float32)

    results = []
    for tag, ir in (("original", module), ("shared", shared)):
        work = tmp_path / tag
        work.mkdir()
        _, fn = compile_split_kernel(ir, "diff", work)
        tmp = np.empty(64, dtype=np.float32)
        out = np.empty(32, dtype=np.float64)
        from sar.runtime import make_descriptor
        import ctypes
        descriptors = [make_descriptor(a) for a in (x, tmp, out)]
        fn(*[ctypes.byref(d) for d in descriptors])
        results.append(out.copy())

    assert np.array_equal(results[0], results[1])
