"""ScaleHLS backend tests: HLS C++ emission and subset diagnostics."""

import pytest

import sar

from conftest import requires_scalehls

pytestmark = requires_scalehls

N = 16


def test_emit_elementwise_kernel():
    @sar.jit
    def phase_mul(re: sar.f32[N, N], im: sar.f32[N, N], cosp: sar.f32[N, N],
                  sinp: sar.f32[N, N]) -> (sar.f32[N, N], sar.f32[N, N]):
        return re * cosp - im * sinp, re * sinp + im * cosp

    design = phase_mul.compile(backend="scalehls")
    source = design.source()
    assert "void phase_mul" in source
    assert "#pragma HLS" in source


def test_hls_design_is_not_executable():
    @sar.jit
    def k(a: sar.f32[N, N]) -> sar.f32[N, N]:
        return a * 2.0

    design = k.compile(backend="scalehls")
    with pytest.raises(RuntimeError, match="cannot be executed"):
        design()


def test_fft_kernel_emits_via_affine_flow():
    """Complex kernels with FFTs go through decomplexify + Stockham affine
    lowering and the HIDA C++ entry point."""
    @sar.jit
    def spectrum(a: sar.c64[N, N], p: sar.f32[N, N]) -> sar.c64[N, N]:
        return sar.fftshift(sar.fft(a * sar.expj(p), dim=1), dim=1)

    design = spectrum.compile(backend="scalehls")
    assert design.flow == "affine"
    source = design.source()
    assert "void spectrum" in source
    assert "#pragma HLS" in source
    # The Stockham twiddle tables become constant arrays.
    assert "twiddle" in source or "float v" in source


def test_interp_kernel_emits_via_affine_flow():
    @sar.jit
    def k(d: sar.c128[N, N], p: sar.f64[N, N]) -> sar.c128[N, N]:
        return sar.interp1d(d, p)

    design = k.compile(backend="scalehls")
    assert design.flow == "affine"
    assert "void k" in design.source()
