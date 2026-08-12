"""Frontend-only tests: tracing, IR emission and error reporting.

These run without any compiled toolchain.
"""

import numpy as np
import pytest

import sar
from sar.language import TraceError


def test_trace_emits_generic_ops():
    N = 16

    @sar.jit
    def k(x: sar.f32[N, N], y: sar.f32[N, N]) -> sar.f32[N, N]:
        return x * 2.0 + y

    text = k.to_mlir()
    assert '"sar.mul_scalar"' in text
    assert '"sar.add"' in text
    assert "tensor<16x16xf32>" in text
    assert "func.func @k" in text


def test_trace_signal_ops():
    N = 32

    @sar.jit
    def k(x: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.ifftshift(sar.ifft(sar.fftshift(sar.fft(x, dim=0),
                                                   dim=0), dim=1), dim=1)

    text = k.to_mlir()
    assert '"sar.fft"' in text and '"sar.ifft"' in text
    assert "inverse" in text  # the ifftshift unit attribute


def test_shape_mismatch_raises_at_trace_time():
    @sar.jit
    def k(x: sar.f32[4, 4], y: sar.f32[8, 8]) -> sar.f32[4, 4]:
        return x + y

    with pytest.raises(TraceError, match="operand types differ"):
        k.to_mlir()


def test_dtype_mismatch_suggests_cast():
    @sar.jit
    def k(x: sar.c64[4], y: sar.c128[4]) -> sar.c128[4]:
        return x * y

    with pytest.raises(TraceError, match="sar.cast"):
        k.to_mlir()


def test_fft_requires_power_of_two():
    @sar.jit
    def k(x: sar.c64[12]) -> sar.c64[12]:
        return sar.fft(x, dim=0)

    with pytest.raises(TraceError, match="power of two"):
        k.to_mlir()


def test_fft_requires_complex():
    @sar.jit
    def k(x: sar.f32[16]) -> sar.f32[16]:
        return sar.fft(x, dim=0)

    with pytest.raises(TraceError, match="complex"):
        k.to_mlir()


def test_result_type_checked_against_annotation():
    @sar.jit
    def k(x: sar.f32[4, 4]) -> sar.f64[4, 4]:
        return x + 1.0

    with pytest.raises(TraceError, match="declares"):
        k.to_mlir()


def test_missing_annotation_rejected():
    with pytest.raises(TraceError, match="type annotation"):
        @sar.jit
        def k(x) -> sar.f32[4]:
            return x


def test_constant_from_numpy_array():
    @sar.jit
    def k(x: sar.f64[4]) -> sar.f64[4]:
        return x * sar.constant(np.array([1.0, 2.0, 3.0, 4.0]))

    text = k.to_mlir()
    assert '"sar.constant"' in text and "dense<[" in text


def test_ops_outside_kernel_rejected():
    with pytest.raises(TraceError, match="traced"):
        sar.Tensor.__add__  # attribute exists
        # Building a constant requires an active tracing context.
        sar.constant(np.zeros((2, 2)))
