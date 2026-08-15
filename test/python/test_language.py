"""Frontend-only tests: tracing, IR emission and error reporting.

These run without any compiled toolchain.
"""

import numpy as np
import pytest

import sar
from sar.language import TraceError


def test_trace_emits_generic_ops():
    N = 16

    @sar.func
    def k(x: sar.f32[N, N], y: sar.f32[N, N]) -> sar.f32[N, N]:
        return x * 2.0 + y

    text = k.to_mlir()
    assert '"sar.mul_scalar"' in text
    assert '"sar.add"' in text
    assert "tensor<16x16xf32>" in text
    assert "func.func @k" in text


def test_trace_signal_ops():
    N = 32

    @sar.func
    def k(x: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.ifftshift(sar.ifft(sar.fftshift(sar.fft(x, dim=0), dim=0),
                                      dim=1),
                             dim=1)

    text = k.to_mlir()
    assert '"sar.fft"' in text and '"sar.ifft"' in text
    assert "inverse" in text  # the ifftshift unit attribute


def test_shape_mismatch_raises_at_trace_time():

    @sar.func
    def k(x: sar.f32[4, 4], y: sar.f32[8, 8]) -> sar.f32[4, 4]:
        return x + y

    with pytest.raises(TraceError, match="operand shapes differ"):
        k.to_mlir()


def test_mixed_dtypes_promote_numpy_style():

    @sar.func
    def k(x: sar.c64[4], y: sar.c128[4]) -> sar.c128[4]:
        return x * y

    text = k.to_mlir()
    assert '"sar.cast"' in text  # c64 operand promoted to c128
    assert text.count("complex<f64>") > text.count("complex<f32>")


def test_fft_accepts_any_size_geq_two():

    @sar.func
    def k(x: sar.c64[12]) -> sar.c64[12]:
        return sar.fft(x, dim=0)

    assert '"sar.fft"' in k.to_mlir()

    @sar.func
    def too_small(x: sar.c64[4, 1]) -> sar.c64[4, 1]:
        return sar.fft(x, dim=1)

    with pytest.raises(TraceError, match="at least 2"):
        too_small.to_mlir()


def test_fft_requires_complex():

    @sar.func
    def k(x: sar.f32[16]) -> sar.f32[16]:
        return sar.fft(x, dim=0)

    with pytest.raises(TraceError, match="complex"):
        k.to_mlir()


def test_result_type_checked_against_annotation():

    @sar.func
    def k(x: sar.f32[4, 4]) -> sar.f64[4, 4]:
        return x + 1.0

    with pytest.raises(TraceError, match="declares"):
        k.to_mlir()


def test_unannotated_kernels_specialize_per_call():

    @sar.func
    def k(x):
        return x * 2.0

    assert isinstance(k, sar.language.GenericKernel)
    variant = k.specialize(sar.f64[4])
    assert "tensor<4xf64>" in variant.to_mlir()


def test_partially_annotated_kernel_rejected():
    with pytest.raises(TraceError, match="type annotation"):

        @sar.func
        def k(x, y: sar.f32[4]) -> sar.f32[4]:
            return x + y


def test_constant_from_numpy_array():

    @sar.func
    def k(x: sar.f64[4]) -> sar.f64[4]:
        return x * sar.constant(np.array([1.0, 2.0, 3.0, 4.0]))

    text = k.to_mlir()
    assert '"sar.constant"' in text and "dense<[" in text


def test_ops_outside_kernel_rejected():
    with pytest.raises(TraceError, match="traced"):
        sar.Tensor.__add__  # attribute exists
        # Building a constant requires an active tracing context.
        sar.constant(np.zeros((2, 2)))
