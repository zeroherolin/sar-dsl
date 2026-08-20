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
    assert 'loc("' in text and "test_language.py" in text


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


def test_fft_promotes_real_input_and_checks_declared_result():

    @sar.func
    def promoted(x: sar.f32[16]) -> sar.c64[16]:
        return sar.fft(x, dim=0)

    text = promoted.to_mlir()
    assert '"sar.cast"' in text
    assert '"sar.fft"' in text

    @sar.func
    def conflicting(x: sar.f32[16]) -> sar.f32[16]:
        return sar.fft(x, dim=0)

    with pytest.raises(TraceError, match="declares"):
        conflicting.to_mlir()


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


def test_dense_hex_constants_are_little_endian():
    values = np.arange(65, dtype=">f8")

    @sar.func
    def k(x: sar.f64[65]) -> sar.f64[65]:
        return x + sar.constant(values)

    expected = values.astype("<f8").tobytes().hex().upper()
    assert f'"0x{expected}"' in k.to_mlir()


def test_numpy_scalars_preserve_shape_and_dtype():

    @sar.func
    def k(x: sar.f32[4]) -> sar.f32[4]:
        bias = sar.constant(np.float32(1.0), shape=(4, ))
        return x * np.float32(2.0) + bias

    text = k.to_mlir()
    assert '"sar.mul_scalar"' in text
    assert "tensor<4xf32>" in text


def test_scalar_operations_do_not_allocate_full_tensors(monkeypatch):

    def reject_full(*args, **kwargs):
        raise AssertionError("scalar tracing allocated a host tensor")

    monkeypatch.setattr(np, "full", reject_full)

    @sar.func
    def k(x: sar.f32[1024, 1024]) -> sar.f32[1024, 1024]:
        divided = x / 0.0 + 2.0 / x
        return sar.where(divided > 1.0, divided, 0.0)

    text = k.to_mlir()
    assert text.count('"sar.div"') == 2
    assert '"sar.cmp"' in text
    assert '"sar.where"' in text


def test_dense_splat_preserves_signed_zero():
    values = np.array([0.0, -0.0], dtype=np.float32)

    @sar.func
    def k(x: sar.f32[2]) -> sar.f32[2]:
        return x + sar.constant(values)

    assert "dense<[" in k.to_mlir()


def test_array_constant_rejects_a_conflicting_shape():

    @sar.func
    def k(x: sar.f32[2]) -> sar.f32[2]:
        return x + sar.constant(np.ones(2, dtype=np.float32), shape=(1, 2))

    with pytest.raises(TraceError, match="does not match array shape"):
        k.to_mlir()


def test_ops_outside_kernel_rejected():
    with pytest.raises(TraceError, match="traced"):
        sar.Tensor.__add__  # attribute exists
        # Building a constant requires an active tracing context.
        sar.constant(np.zeros((2, 2)))


def test_cumsum_emits_ir():
    N = 8

    @sar.func
    def k(x: sar.f64[N, N]) -> sar.f64[N, N]:
        return sar.cumsum(x, dim=1)

    text = k.to_mlir()
    assert '"sar.cumsum"' in text
    assert 'dim = 1' in text


def test_cumsum_complex_emits_ir():
    N = 8

    @sar.func
    def k(x: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.cumsum(x, dim=0)

    text = k.to_mlir()
    assert '"sar.cumsum"' in text


def test_cumsum_int_rejected():
    N = 8
    from sar.language import TraceError

    @sar.func
    def k(x: sar.i64[N, N]) -> sar.i64[N, N]:
        return sar.cumsum(x, dim=1)

    with pytest.raises(TraceError, match="integer"):
        k.to_mlir()


def test_rank_filter_emits_ir():
    N = 16

    @sar.func
    def k(x: sar.f32[N, N]) -> sar.f32[N, N]:
        return sar.rank_filter(x, window=5, rank=2, dim=1)

    text = k.to_mlir()
    assert '"sar.rank_filter"' in text
    assert 'window = 5' in text


def test_median_filter_emits_ir():
    N = 16

    @sar.func
    def k(x: sar.f32[N, N]) -> sar.f32[N, N]:
        return sar.median_filter(x, window=7, dim=0)

    text = k.to_mlir()
    assert '"sar.rank_filter"' in text
    assert 'rank = 3' in text  # window//2 = 7//2 = 3


def test_rank_filter_even_window_rejected():
    N = 16

    @sar.func
    def k(x: sar.f32[N, N]) -> sar.f32[N, N]:
        return sar.rank_filter(x, window=4, rank=1, dim=1)

    with pytest.raises(TraceError, match="odd"):
        k.to_mlir()


def test_rank_filter_complex_rejected():
    N = 8

    @sar.func
    def k(x: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.rank_filter(x, window=3, rank=1, dim=1)

    with pytest.raises(TraceError, match="float"):
        k.to_mlir()


def test_where_rejects_integer_branches():
    """Integer branches raise a TraceError that identifies `sar.where`."""
    N = 8

    @sar.func
    def k(x: sar.f32[N], a: sar.i32[N], b: sar.i32[N]) -> sar.i32[N]:
        return sar.where(x > 0.0, a, b)

    with pytest.raises(TraceError, match="integer branch"):
        k.to_mlir()


def test_integer_division_rejects_fractional_and_zero_scalars():

    @sar.func
    def valid(x: sar.i32[4]) -> sar.i32[4]:
        return x / 2

    assert '"sar.div"' in valid.to_mlir()

    @sar.func
    def fractional(x: sar.i32[4]) -> sar.i32[4]:
        return x / 2.5

    with pytest.raises(TraceError, match="integer scalar"):
        fractional.to_mlir()

    @sar.func
    def zero(x: sar.i32[4]) -> sar.i32[4]:
        return x / 0

    with pytest.raises(TraceError, match="nonzero"):
        zero.to_mlir()


@pytest.mark.parametrize("axis", [True, 1.5, "1"])
def test_axis_must_be_an_integer(axis):

    @sar.func
    def kernel(x: sar.f32[4, 4]) -> sar.f32[4]:
        return sar.sum(x, axis=axis)

    with pytest.raises(TraceError, match="must be an integer"):
        kernel.to_mlir()


def test_argmax_rejects_rank2_axis_out_of_range():

    @sar.func
    def kernel(x: sar.f32[4, 4]) -> sar.i64[4]:
        return sar.argmax(x, axis=2)

    with pytest.raises(TraceError, match="out of range for rank 2"):
        kernel.to_mlir()


def test_unresolvable_annotation_reports_kernel_and_text():
    """A string annotation that fails to evaluate must name the kernel
    and the annotation, not surface as a bare NameError."""
    with pytest.raises(TraceError, match=r"kernel 'bad'.*undefined_dim"):

        @sar.func
        def bad(x: "sar.f32[undefined_dim]") -> "sar.f32[4]":  # noqa: F821
            return x


def test_generic_kernel_dtype_errors_are_specific():

    @sar.func
    def k(x):
        return x * 2.0

    # Not an array at all: say what specialization needs.
    with pytest.raises(TraceError, match="numpy array"):
        k([1.0, 2.0])
    # An array of an unsupported dtype: name it and list the supported.
    with pytest.raises(TraceError, match=r"float16.*float32"):
        k(np.zeros(4, dtype=np.float16))


def test_to_mlir_reuses_the_trace():
    """Tracing is deterministic per kernel, so repeated to_mlir calls
    reuse the first trace instead of re-running the kernel body."""
    body_runs = []
    N = 4

    @sar.func
    def k(x: sar.f32[N]) -> sar.f32[N]:
        body_runs.append(1)
        return x + 1.0

    first = k.to_mlir()
    second = k.to_mlir()
    assert first == second
    assert len(body_runs) == 1


def test_renaming_a_traced_kernel_invalidates_its_ir():

    @sar.func
    def original(x: sar.f32[4]) -> sar.f32[4]:
        return x + 1.0

    assert "func.func @original" in original.to_mlir()
    original.name = "renamed"
    text = original.to_mlir()
    assert "func.func @renamed" in text
    assert "func.func @original" not in text


def test_explicit_retrace_invalidates_compiled_launchers():
    scale = [1.0]

    @sar.func
    def kernel(x: sar.f32[4]) -> sar.f32[4]:
        return x * scale[0]

    before = kernel.to_mlir()
    kernel._compiled[("test", ())] = object()
    scale[0] = 2.0
    after = kernel.trace()
    assert before != after
    assert not kernel._compiled


def test_trace_error_is_a_first_class_export():
    from sar.errors import TraceError as from_errors

    assert sar.TraceError is from_errors
    assert sar.language.TraceError is from_errors
    assert "TraceError" in sar.__all__
    assert issubclass(sar.TraceError, sar.SARError)
    # Existing TypeError handlers also catch tracing failures.
    assert issubclass(sar.TraceError, TypeError)
