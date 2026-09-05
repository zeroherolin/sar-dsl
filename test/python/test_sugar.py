"""NumPy-style ergonomics: array lifting, dtype promotion, broadcasting,
operator/method sugar, annotation-free kernels and Python-defined
operators (@sar.op).
"""

import numpy as np
import pytest

import sar

from conftest import requires_cpu

pytestmark = requires_cpu

N, M = 8, 16
RNG = np.random.default_rng(2078)


def _complex(n, m):
    return RNG.standard_normal((n, m)) + 1j * RNG.standard_normal((n, m))


def test_numpy_lift_promotion_broadcast_and_axis():
    z = _complex(N, M)
    win = np.hanning(M)

    @sar.func
    def k(z):
        spec = sar.fft(z, axis=1)  # axis= alias
        spec = spec * win  # ndarray lift + f64 -> c128 + bcast
        return abs(sar.ifft(spec, axis=1))**2

    ref = np.abs(np.fft.ifft(np.fft.fft(z, axis=1) * win, axis=1))**2
    np.testing.assert_allclose(k(z), ref, rtol=1e-12, atol=1e-12)


def test_reflected_operators_with_arrays():
    x = RNG.standard_normal((N, M))
    bias = RNG.standard_normal((N, M))

    @sar.func
    def k(x):
        return bias - x, 1.0 / (abs(x) + 1.0)

    diff, inv = k(x)
    np.testing.assert_allclose(diff, bias - x, rtol=1e-12)
    np.testing.assert_allclose(inv, 1.0 / (np.abs(x) + 1.0), rtol=1e-12)


def test_methods_and_properties():
    z = _complex(N, M)

    @sar.func
    def k(z):
        c = z.conj()
        return (c.real + c.imag).sum(axis=1), abs(z).max()

    row_sum, peak = k(z)
    np.testing.assert_allclose(row_sum, (z.real - z.imag).sum(axis=1),
                               rtol=1e-12)
    np.testing.assert_allclose(peak, [np.abs(z).max()], rtol=1e-12)


def test_fft_promotes_float_input():
    x = RNG.standard_normal((N, M))

    @sar.func
    def k(x):
        return abs(sar.fft(x, axis=1))

    np.testing.assert_allclose(k(x), np.abs(np.fft.fft(x, axis=1)), rtol=1e-12)


def test_annotation_free_kernel_specializes_per_shape():

    @sar.func
    def scale(x):
        return x * 2.0

    for shape in ((4, 4), (8, 2)):
        x = RNG.standard_normal(shape)
        np.testing.assert_allclose(scale(x), 2.0 * x, rtol=0, atol=0)
    x32 = RNG.standard_normal((4, 4)).astype(np.float32)
    assert scale(x32).dtype == np.float32


def test_function_defines_python_operator():

    @sar.op
    def range_compress(data, replica):
        return sar.ifft(sar.fft(data, axis=1) * replica, axis=1)

    z = _complex(N, M)
    replica = np.exp(-1j * np.linspace(0.0, 3.0, M))
    ref = np.fft.ifft(np.fft.fft(z, axis=1) * replica, axis=1)

    # Eagerly on arrays (auto-JIT) ...
    np.testing.assert_allclose(range_compress(z, replica),
                               ref,
                               rtol=1e-12,
                               atol=1e-12)

    # ... and inlined inside another kernel, like a built-in op.
    @sar.func
    def chain(z):
        return abs(range_compress(z, replica))

    np.testing.assert_allclose(chain(z), np.abs(ref), rtol=1e-12, atol=1e-12)


def test_function_specializes_for_emission_backends():

    @sar.op
    def phase_only(z, phase):
        return z * sar.expj(phase)

    kernel = phase_only.func.specialize(sar.c64[N, M], sar.f32[N, M])
    text = kernel.to_mlir()
    assert '"sar.cos"' in text and '"sar.complex"' in text  # expj composed


def test_numeric_vocabulary_matches_numpy():
    """argmin / sign / floor / ceil / round compositions vs numpy."""
    n = 128
    rng = np.random.default_rng(21)
    a = rng.uniform(-50.0, 50.0, n)
    a[:8] = [-2.5, -2.0, -0.5, 0.0, 0.5, 1.5, 2.0, 2.5]  # edge cases

    @sar.func
    def k(x):
        return (sar.cast(sar.argmin(x), sar.f64), sar.sign(x), sar.floor(x),
                sar.ceil(x), sar.round(x))

    got_argmin, got_sign, got_floor, got_ceil, got_round = k(a)
    np.testing.assert_allclose(got_argmin, [np.argmin(a)], atol=0)
    np.testing.assert_allclose(got_sign, np.sign(a), atol=0)
    np.testing.assert_allclose(got_floor, np.floor(a), atol=0)
    np.testing.assert_allclose(got_ceil, np.ceil(a), atol=0)
    # round: half away from zero (Matlab), not numpy's half-to-even.
    want_round = np.sign(a) * np.floor(np.abs(a) + 0.5)
    np.testing.assert_allclose(got_round, want_round, atol=0)


def test_sign_propagates_nan():
    values = np.array([np.nan, -0.0, 0.0, -2.0, 3.0])

    @sar.func
    def kernel(x):
        return sar.sign(x)

    np.testing.assert_allclose(kernel(values), np.sign(values), equal_nan=True)


def test_floor_ceil_round_define_nonfinite_and_large_inputs():
    values = np.array(
        [np.nan, -np.inf, np.inf, -0.0, 0.0, -(2.0**63), 2.0**63, -2.5, 2.5])

    @sar.func
    def kernel(x):
        return sar.floor(x), sar.ceil(x), sar.round(x)

    floor, ceil, rounded = kernel(values)
    np.testing.assert_array_equal(floor, np.floor(values))
    np.testing.assert_array_equal(ceil, np.ceil(values))
    expected = np.sign(values) * np.floor(np.abs(values) + 0.5)
    np.testing.assert_array_equal(rounded, expected)
    for result in (floor, ceil, rounded):
        assert np.signbit(result[3])  # preserve negative zero


def test_fft_norm_conventions():
    """fft/ifft norm= matches numpy for all three conventions."""
    z = _complex(N, M)
    for norm in ("backward", "ortho", "forward"):

        @sar.func
        def k(x: sar.c128[N, M]) -> sar.c128[N, M]:
            return sar.ifft(sar.fft(x, axis=1, norm=norm), axis=1, norm=norm)

        np.testing.assert_allclose(k(z), z, rtol=1e-12, atol=1e-12)

    @sar.func
    def fwd(x: sar.c128[N, M]) -> sar.c128[N, M]:
        return sar.fft(x, axis=1, norm="ortho")

    np.testing.assert_allclose(fwd(z),
                               np.fft.fft(z, axis=1, norm="ortho"),
                               rtol=1e-12,
                               atol=1e-12)


def test_method_and_alias_symmetry():
    """x.mean() / x.astype() methods and the sar.abs alias."""
    rng = np.random.default_rng(33)
    a = rng.standard_normal((N, M))

    @sar.func
    def k(x):
        return (x.mean(axis=0), x.astype(sar.f32), sar.abs(x))

    mu, cast32, mag = k(a)
    np.testing.assert_allclose(mu, a.mean(axis=0), rtol=1e-12)
    assert cast32.dtype == np.float32
    np.testing.assert_allclose(mag, np.abs(a), rtol=1e-12)


def test_numpy_spelling_aliases():
    """Names numpy exposes under two spellings work under both here too:
    `concat` (numpy 2.0 / Array API) beside `concatenate`, `abs` beside
    `absolute`, `conjugate` beside `conj`, `db` beside `mag2db`."""
    assert sar.concat is sar.concatenate
    assert sar.abs is sar.absolute
    assert sar.conjugate is sar.conj
    assert sar.db is sar.mag2db

    rng = np.random.default_rng(43)
    a = rng.standard_normal((3, 4))

    @sar.func
    def k(x):
        return sar.concat([x, x], axis=1)

    np.testing.assert_allclose(k(a), np.concatenate([a, a], axis=1))


def test_every_op_with_a_method_form_matches_its_free_function():
    """The numpy-style method surface: every method or property a numpy
    user reaches for delegates to the free function with the same name.
    Pinned pairwise so a new op with a natural method form is added to
    both spellings or fails here."""
    rng = np.random.default_rng(35)
    a = rng.standard_normal((N, M))
    z = (rng.standard_normal((N, M)) + 1j * rng.standard_normal(
        (N, M))).astype(np.complex128)

    @sar.func
    def methods(x, w):
        return (x.transpose(), w.conjugate(), x.clip(-0.5, 0.5), x.round(),
                x.cumsum(axis=1), x.std(), x.var(), (+x) - x)

    t, cj, cl, rd, cs, sd, vr, zero = methods(a, z)
    np.testing.assert_allclose(t, a.T)
    np.testing.assert_allclose(cj, np.conj(z))
    np.testing.assert_allclose(cl, np.clip(a, -0.5, 0.5))
    # sar.round rounds half away from zero (Matlab); the inputs here are
    # generic reals, where the two conventions agree.
    np.testing.assert_allclose(rd, np.round(a))
    np.testing.assert_allclose(cs, np.cumsum(a, axis=1), rtol=1e-12)
    np.testing.assert_allclose(sd, a.std(), rtol=1e-12)
    np.testing.assert_allclose(vr, a.var(), rtol=1e-12)
    np.testing.assert_allclose(zero, np.zeros_like(a))


def test_shape_introspection_matches_numpy():
    """`.ndim`, `.size` and `len()` answer during tracing, numpy-style."""

    @sar.func
    def k(x: sar.f64[4, 3]) -> sar.f64[4, 3]:
        assert x.ndim == x.rank == 2
        assert x.size == 12
        assert len(x) == 4
        return x

    k.to_mlir()


def test_scalar_conversions_and_ufuncs_are_rejected_clearly():
    """`float(x)` and `np.conj(x)` cannot mean anything during tracing;
    both must fail with a diagnostic, not with silent object arrays."""

    @sar.func
    def to_float(x: sar.f64[4, 3]) -> sar.f64[4, 3]:
        return x * float(x)

    with pytest.raises(sar.TraceError, match="cannot convert"):
        to_float.to_mlir()

    @sar.func
    def ufunc(x: sar.c128[4, 3]) -> sar.c128[4, 3]:
        return np.conj(x)

    with pytest.raises(TypeError):
        ufunc.to_mlir()


def test_axis_is_accepted_wherever_dim_is():
    """`axis=` is the numpy spelling of `dim=`, and every op taking an axis
    accepts both -- mixing the two in one call is the error."""
    import inspect

    for name in sorted(sar.__all__):
        fn = getattr(sar, name)
        if not callable(fn):
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if "dim" in params:
            assert "axis" in params, f"sar.{name} takes dim but not axis"

    rng = np.random.default_rng(41)
    a = rng.standard_normal((4, 3))

    @sar.func
    def k(x):
        return (sar.concatenate([x, x],
                                axis=1), sar.sum(x, axis=0), sar.flip(x,
                                                                      axis=1))

    cat, total, flipped = k(a)
    np.testing.assert_allclose(cat, np.concatenate([a, a], axis=1))
    np.testing.assert_allclose(total, a.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(flipped, np.flip(a, axis=1))

    @sar.func
    def both(x):
        return sar.concatenate([x, x], dim=0, axis=1)

    with pytest.raises(sar.language.TraceError, match="not both"):
        both(a)


def test_scalar_branches_follow_working_precision():
    """`where` with two Python-scalar branches must not promote to f64.

    The branch dtype is taken from the mask (the working precision of the
    surrounding computation), not from the Python float. Getting this wrong
    silently widened f32 pipelines -- and on HLS it synthesizes
    double-precision hardware for what the user wrote as f32.
    """
    a32 = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)

    @sar.func
    def pick(x):
        return sar.where(x > 0.0, 1.0, 0.0)

    assert pick(a32).dtype == np.float32
    assert pick(a32.astype(np.float64)).dtype == np.float64

    # sign/round are `where` compositions, so they inherit the same rule.
    @sar.func
    def signs(x):
        return sar.sign(x), sar.round(x)

    got_sign, got_round = signs(a32)
    assert got_sign.dtype == np.float32
    assert got_round.dtype == np.float32
    np.testing.assert_allclose(got_sign, np.sign(a32), atol=0)
    np.testing.assert_allclose(got_round,
                               np.sign(a32) * np.floor(np.abs(a32) + 0.5),
                               atol=0)


def test_lambda_kernel_rejected_at_trace_time():
    """A kernel name has to be a legal MLIR symbol.

    Without the trace-time check the invalid `@<lambda>` symbol reached
    sar-opt and came back as a parse error about the serialized IR, which
    pointed nowhere near the offending Python.
    """
    with pytest.raises(sar.language.TraceError,
                       match="not a portable identifier"):
        sar.func(lambda x: x * 2.0)(np.ones(4, dtype=np.float32))
