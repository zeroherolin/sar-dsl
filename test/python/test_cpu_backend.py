"""CPU backend end-to-end tests: every op validated against numpy."""

import sys

import numpy as np
import pytest

import sar

from conftest import requires_cpu

pytestmark = requires_cpu

#: Reset before every test (autouse fixture below): each test draws from
#: its own deterministic stream, so adding or reordering tests never
#: shifts the data another test sees.
RNG = np.random.default_rng(2026)


@pytest.fixture(autouse=True)
def _per_test_rng():
    global RNG
    RNG = np.random.default_rng(2026)


def _c64(*shape):
    return (RNG.standard_normal(shape) +
            1j * RNG.standard_normal(shape)).astype(np.complex64)


def _c128(*shape):
    return RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)


def _f32(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def _f64(*shape):
    return RNG.standard_normal(shape)


def test_elementwise_float():
    N = 16

    @sar.func
    def k(a: sar.f32[N, N], b: sar.f32[N, N]) -> sar.f32[N, N]:
        return (a * 3.0 - b) / 2.0 + 1.0

    a, b = _f32(N, N), _f32(N, N)
    ref = ((a * np.float32(3.0) - b) / np.float32(2.0)) + np.float32(1.0)
    np.testing.assert_allclose(k(a, b), ref, rtol=1e-6)


def test_elementwise_complex():
    N = 8

    @sar.func
    def k(a: sar.c128[N], b: sar.c128[N]) -> sar.c128[N]:
        return a * b + a / b - b

    a, b = _c128(N), _c128(N)
    np.testing.assert_allclose(k(a, b), a * b + a / b - b, rtol=1e-12)


def test_complex_access_ops():
    N = 8

    @sar.func
    def k(
        z: sar.c128[N, N]
    ) -> (sar.f64[N, N], sar.f64[N, N], sar.f64[N, N], sar.c128[N, N]):
        c = sar.conj(z)
        return (sar.real(c), sar.imag(c), sar.angle(z),
                sar.make_complex(sar.imag(z), sar.real(z)))

    z = _c128(N, N)
    re, im, ang, swapped = k(z)
    np.testing.assert_allclose(re, z.real, rtol=0, atol=0)
    np.testing.assert_allclose(im, -z.imag, rtol=0, atol=0)
    np.testing.assert_allclose(ang, np.angle(z), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(swapped, z.imag + 1j * z.real, rtol=0, atol=0)


def test_exp_log_atan2():
    N = 16

    @sar.func
    def k(a: sar.f64[N], b: sar.f64[N]) -> (sar.f64[N], sar.f64[N]):
        return sar.log(sar.exp(a)), sar.atan2(a, b)

    a, b = _f64(N), _f64(N)
    got_log, got_atan2 = k(a, b)
    np.testing.assert_allclose(got_log, a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_atan2, np.arctan2(a, b), rtol=1e-12)


def test_int_casts_and_argmax_chain():
    """int<->float casts make argmax results usable in-kernel: the
    peak index participates in arithmetic instead of dead-ending."""
    n = 64
    rng = np.random.default_rng(15)
    a = rng.standard_normal(n)

    @sar.func
    def peak_frac(x: sar.f64[n]) -> sar.f64[1]:
        idx = sar.cast(sar.argmax(x), sar.f64)
        return idx * (1.0 / n)

    got = peak_frac(a)
    np.testing.assert_allclose(got, [np.argmax(a) / n], rtol=0, atol=0)

    @sar.func
    def trunc_roundtrip(x: sar.f64[n]) -> sar.f64[n]:
        return sar.cast(sar.cast(x, sar.i64), sar.f64)

    vals = rng.uniform(-100.0, 100.0, n)
    np.testing.assert_allclose(trunc_roundtrip(vals),
                               np.trunc(vals),
                               rtol=0,
                               atol=0)


def test_reductions():
    N, M = 8, 16

    @sar.func
    def k(
        a: sar.f64[N, M]
    ) -> (sar.f64[N], sar.f64[M], sar.f64[1], sar.i64[N], sar.i64[1]):
        return (sar.sum(a, dim=1), sar.max(a, dim=0), sar.min(a),
                sar.argmax(a, dim=1), sar.argmax(sar.sum(a, dim=0)))

    a = _f64(N, M)
    s, mx, mn, am, am_full = k(a)
    np.testing.assert_allclose(s, a.sum(axis=1), rtol=1e-12)
    np.testing.assert_allclose(mx, a.max(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(mn, [a.min()], rtol=0, atol=0)
    np.testing.assert_array_equal(am, a.argmax(axis=1))
    np.testing.assert_array_equal(am_full, [a.sum(axis=0).argmax()])


def test_nan_comparison_selection_and_reduction_match_numpy():

    @sar.func
    def kernel(
        x: sar.f64[4]
    ) -> (sar.f64[4], sar.f64[4], sar.f64[4], sar.f64[4], sar.f64[1],
          sar.f64[1], sar.i64[1], sar.i64[1]):
        return (x != 0.0, sar.maximum(x, 1.0), sar.minimum(x, 1.0),
                sar.clip(x, -1.0, 1.0), sar.max(x), sar.min(x), sar.argmax(x),
                sar.argmin(x))

    x = np.array([0.0, np.nan, 2.0, -2.0])
    ne, maximum, minimum, clipped, high, low, arg_high, arg_low = kernel(x)
    np.testing.assert_array_equal(ne, (x != 0.0).astype(np.float64))
    np.testing.assert_array_equal(maximum, np.maximum(x, 1.0))
    np.testing.assert_array_equal(minimum, np.minimum(x, 1.0))
    np.testing.assert_array_equal(clipped, np.clip(x, -1.0, 1.0))
    np.testing.assert_array_equal(high, [np.max(x)])
    np.testing.assert_array_equal(low, [np.min(x)])
    np.testing.assert_array_equal(arg_high, [np.argmax(x)])
    np.testing.assert_array_equal(arg_low, [np.argmin(x)])


def test_complex_sum():
    N, M = 4, 8

    @sar.func
    def k(z: sar.c128[N, M]) -> sar.c128[N]:
        return sar.sum(z, dim=1)

    z = _c128(N, M)
    np.testing.assert_allclose(k(z), z.sum(axis=1), rtol=1e-12)


def test_slice_concat_pad():
    N, M = 8, 16

    @sar.func
    def k(
        a: sar.f64[N, M], z: sar.c128[N, M]
    ) -> (sar.f64[4, 8], sar.f64[N, 2 * M], sar.c128[N + 3, M]):
        return (a[2:6, 4:12], sar.concatenate(
            (a, a), dim=1), sar.pad(z, ((1, 2), (0, 0)), value=0.5))

    a, z = _f64(N, M), _c128(N, M)
    s, c, p = k(a, z)
    np.testing.assert_allclose(s, a[2:6, 4:12], rtol=0, atol=0)
    np.testing.assert_allclose(c,
                               np.concatenate((a, a), axis=1),
                               rtol=0,
                               atol=0)
    np.testing.assert_allclose(p,
                               np.pad(z, ((1, 2), (0, 0)),
                                      constant_values=0.5),
                               rtol=0,
                               atol=0)


def test_interp1d_dim0_matches_transposed_dim1():
    N, M = 8, 16
    pos = (np.tile(np.arange(N, dtype=np.float64)[:, None],
                   (1, M)) * 0.75 + 0.5)

    @sar.func
    def by_dim(z: sar.c128[N, M], p: sar.f64[N, M]) -> sar.c128[N, M]:
        return sar.interp1d(z, p, dim=0)

    @sar.func
    def by_transpose(z: sar.c128[N, M], p: sar.f64[N, M]) -> sar.c128[N, M]:
        return sar.transpose(
            sar.interp1d(sar.transpose(z), sar.transpose(p), dim=1))

    z = _c128(N, M)
    np.testing.assert_allclose(by_dim(z, pos),
                               by_transpose(z, pos),
                               rtol=1e-12,
                               atol=1e-12)


def test_multilook():
    N, M = 8, 12

    @sar.func
    def k(a: sar.f64[N, M]) -> sar.f64[2, 4]:
        return sar.multilook(a, (4, 3))

    a = np.abs(_f64(N, M))
    ref = a.reshape(2, 4, 4, 3).mean(axis=(1, 3))
    np.testing.assert_allclose(k(a), ref, rtol=1e-12)


@pytest.mark.parametrize("n", [12, 33, 97, 100])
def test_fft_non_pow2_bluestein(n):

    @sar.func
    def k(z: sar.c128[4, n]) -> (sar.c128[4, n], sar.c128[4, n]):
        spectrum = sar.fft(z, dim=1)
        return spectrum, sar.ifft(spectrum, dim=1)

    z = _c128(4, n)
    spectrum, roundtrip = k(z)
    np.testing.assert_allclose(spectrum,
                               np.fft.fft(z, axis=1),
                               rtol=1e-9,
                               atol=1e-9)
    np.testing.assert_allclose(roundtrip, z, rtol=1e-10, atol=1e-10)


def test_neg_sqrt_abs_maximum():
    N = 64

    @sar.func
    def k(a: sar.f64[N]) -> sar.f64[N]:
        return sar.sqrt(sar.maximum(-a, 1e-3))

    a = _f64(N)
    np.testing.assert_allclose(k(a), np.sqrt(np.maximum(-a, 1e-3)), rtol=1e-12)


def test_scalar_division_preserves_ieee_and_f32_rounding():

    @sar.func
    def divide(x: sar.f32[4]) -> (sar.f32[4], sar.f32[4]):
        return x / 0.0, 3.0 / x

    x = np.array([0.0, -0.0, 3.0, 7.0], dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = (x / np.float32(0.0), np.float32(3.0) / x)
    got = divide(x)
    for actual, reference in zip(got, expected):
        np.testing.assert_array_equal(actual, reference)


def test_abs_complex_magnitude():
    N = 32

    @sar.func
    def k(a: sar.c64[N, N]) -> sar.f32[N, N]:
        return sar.absolute(a)

    a = _c64(N, N)
    np.testing.assert_allclose(k(a), np.abs(a), rtol=1e-5, atol=1e-6)


def test_expj_matches_numpy():
    N = 16

    @sar.func
    def k(p: sar.f64[N, N]) -> sar.c128[N, N]:
        return sar.expj(p)

    p = 1e6 * _f64(N, N)
    np.testing.assert_allclose(k(p), np.exp(1j * p), rtol=1e-12, atol=1e-12)


def test_cast_widening_and_to_complex():
    N = 8

    @sar.func
    def k(a: sar.f32[N], b: sar.c64[N]) -> sar.c128[N]:
        return sar.cast(a, sar.c128) + sar.cast(b, sar.c128)

    a, b = _f32(N), _c64(N)
    ref = a.astype(np.complex128) + b.astype(np.complex128)
    np.testing.assert_allclose(k(a, b), ref, rtol=1e-6)


def test_complex_constant():
    N = 4
    w = np.array([1 + 2j, 3 - 1j, 0.5j, 2 + 0j], dtype=np.complex128)

    @sar.func
    def k(x: sar.c128[N]) -> sar.c128[N]:
        return x * sar.constant(w)

    x = _c128(N)
    np.testing.assert_allclose(k(x), x * w, rtol=1e-12)


def test_large_constant_hex_encoded():
    # Above the hex threshold the dense attribute uses a raw blob; values
    # must survive the round-trip bit-exactly.
    N = 256
    w = RNG.standard_normal(N)

    @sar.func
    def k(x: sar.f64[N]) -> sar.f64[N]:
        return x + sar.constant(w)

    assert 'dense<"0x' in k.to_mlir()
    x = _f64(N)
    np.testing.assert_array_equal(k(x), x + w)


def test_transpose():

    @sar.func
    def k(a: sar.f32[4, 8]) -> sar.f32[8, 4]:
        return a.T

    a = _f32(4, 8)
    np.testing.assert_array_equal(k(a), a.T)


def test_broadcast_both_axes():
    N, M = 4, 8

    @sar.func
    def k(m: sar.f64[N, M], r: sar.f64[M], c: sar.f64[N]) -> sar.f64[N, M]:
        return (m * sar.broadcast(r, (N, M), dim=1) +
                sar.broadcast(c, (N, M), dim=0))

    m, r, c = _f64(N, M), _f64(M), _f64(N)
    np.testing.assert_allclose(k(m, r, c),
                               m * r[None, :] + c[:, None],
                               rtol=1e-12)


@pytest.mark.parametrize("n", [8, 9])  # even and odd sizes
def test_fftshift_conventions(n):

    @sar.func
    def k(a: sar.f64[n]) -> sar.f64[n]:
        return sar.fftshift(a, dim=0)

    @sar.func
    def ki(a: sar.f64[n]) -> sar.f64[n]:
        return sar.ifftshift(a, dim=0)

    a = _f64(n)
    np.testing.assert_array_equal(k(a), np.fft.fftshift(a))
    np.testing.assert_array_equal(ki(a), np.fft.ifftshift(a))


@pytest.mark.parametrize("dim", [0, 1])
def test_fft_2d_against_numpy(dim):
    N, M = 32, 64

    @sar.func
    def k(a: sar.c128[N, M]) -> sar.c128[N, M]:
        return sar.fft(a, dim=dim)

    a = _c128(N, M)
    np.testing.assert_allclose(k(a),
                               np.fft.fft(a, axis=dim),
                               rtol=1e-10,
                               atol=1e-10)


def test_ifft_roundtrip():
    N = 128

    @sar.func
    def k(a: sar.c128[N]) -> sar.c128[N]:
        return sar.ifft(sar.fft(a, dim=0), dim=0)

    a = _c128(N)
    np.testing.assert_allclose(k(a), a, rtol=1e-12, atol=1e-12)


def test_fft_c64_precision():
    N = 64

    @sar.func
    def k(a: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.fft(a, dim=1)

    a = _c64(N, N)
    np.testing.assert_allclose(k(a),
                               np.fft.fft(a, axis=1),
                               rtol=1e-4,
                               atol=1e-4)


def test_multiple_results():
    N = 8

    @sar.func
    def k(a: sar.f32[N], b: sar.f32[N]) -> (sar.f32[N], sar.f32[N]):
        return a + b, a - b

    a, b = _f32(N), _f32(N)
    s, d = k(a, b)
    np.testing.assert_allclose(s, a + b, rtol=1e-6)
    np.testing.assert_allclose(d, a - b, rtol=1e-6)


def test_duplicate_tensor_results_get_distinct_output_buffers():
    """CSE may leave two result positions naming one tensor SSA value."""
    n = 8

    @sar.func
    def duplicate(x: sar.f32[n]) -> (sar.f32[n], sar.f32[n]):
        result = x * 2.0
        return result, result

    values = _f32(n)
    first, second = duplicate(values)
    np.testing.assert_array_equal(first, values * 2.0)
    np.testing.assert_array_equal(second, values * 2.0)
    assert not np.shares_memory(first, second)


def test_launcher_validates_arguments():
    N = 8

    @sar.func
    def k(a: sar.f32[N]) -> sar.f32[N]:
        return a + 1.0

    with pytest.raises(sar.LaunchError, match="shape"):
        k(_f32(N + 1))
    with pytest.raises(sar.LaunchError, match="dtype"):
        k(_f64(N))
    with pytest.raises(sar.LaunchError, match="argument"):
        k(_f32(N), _f32(N))


@pytest.mark.skipif(sys.platform != "linux",
                    reason="RSS measurement reads /proc (Linux only)")
def test_no_memory_leak_across_launches():
    """Buffer deallocation must free every intermediate: repeated launches
    (including runtime FFT calls) may not grow the process RSS."""
    N = 64

    @sar.func
    def k(x: sar.c64[N, N]) -> sar.f32[N, N]:
        y = sar.fftshift(sar.fft(x, dim=1), dim=1)
        return sar.absolute(sar.ifft(y, dim=0))

    x = _c64(N, N)
    fn = k.compile("cpu")

    def rss_kb():
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1])

    fn(x)  # warm up (lazy pages, thread stacks)
    before = rss_kb()
    for _ in range(500):
        fn(x)
    growth_mb = (rss_kb() - before) / 1024.0
    assert growth_mb < 32, f"RSS grew by {growth_mb:.1f} MiB over 500 runs"


def test_non_contiguous_input_handled():
    N = 8

    @sar.func
    def k(a: sar.f32[N, N]) -> sar.f32[N, N]:
        return a * 2.0

    big = _f32(2 * N, 2 * N)
    view = big[::2, ::2]  # non-contiguous view
    np.testing.assert_allclose(k(view), view * np.float32(2.0), rtol=1e-6)


# ---- cumsum -----------------------------------------------------------------


def test_cumsum_dim0_f64():
    N = 8

    @sar.func
    def k(x: sar.f64[N, N]) -> sar.f64[N, N]:
        return sar.cumsum(x, dim=0)

    x = _f64(N, N)
    expected = np.cumsum(x, axis=0)
    np.testing.assert_allclose(k(x), expected, rtol=1e-10)


def test_cumsum_dim1_f32():
    N, M = 5, 12

    @sar.func
    def k(x: sar.f32[N, M]) -> sar.f32[N, M]:
        return sar.cumsum(x, dim=1)

    x = _f32(N, M)
    expected = np.cumsum(x, axis=1)
    np.testing.assert_allclose(k(x), expected, rtol=1e-5)


def test_cumsum_non_square():
    """Non-square tensor, dim=0."""
    R, C = 3, 16

    @sar.func
    def k(x: sar.f64[R, C]) -> sar.f64[R, C]:
        return sar.cumsum(x, dim=0)

    x = _f64(R, C)
    np.testing.assert_allclose(k(x), np.cumsum(x, axis=0), rtol=1e-10)


def test_cumsum_rank1():
    """A rank-1 scan runs along the only axis, complex included."""
    N = 17

    @sar.func
    def k(x: sar.f64[N]) -> sar.f64[N]:
        return sar.cumsum(x)

    @sar.func
    def kc(x: sar.c128[N]) -> sar.c128[N]:
        return sar.cumsum(x)

    x = _f64(N)
    np.testing.assert_allclose(k(x), np.cumsum(x), rtol=1e-10)
    xc = _f64(N) + 1j * _f64(N)
    np.testing.assert_allclose(kc(xc), np.cumsum(xc), rtol=1e-10)


# ---- rank_filter / median_filter -------------------------------------------


def test_rank_filter_min():
    """rank=0 is the running minimum in the window."""
    N = 16

    @sar.func
    def k(x: sar.f64[N, N]) -> sar.f64[N, N]:
        return sar.rank_filter(x, window=5, rank=0, dim=1)

    x = _f64(N, N)
    try:
        from scipy.ndimage import rank_filter as scipy_rf
        expected = scipy_rf(x, rank=0, size=(1, 5), mode="nearest")
    except ImportError:
        # Pure-numpy fallback: clamp-replicate boundary
        half = 2
        expected = np.zeros_like(x)
        for j in range(N):
            lo, hi = max(0, j - half), min(N - 1, j + half)
            # build padded window with edge replication
            w = np.pad(x[:, lo:hi + 1],
                       [(0, 0), (max(0, half - j), max(0, j + half - N + 1))],
                       mode="edge")
            expected[:, j] = np.sort(w, axis=1)[:, 0]
    np.testing.assert_allclose(k(x), expected, rtol=1e-10)


def test_median_filter_dim0():
    """median_filter along axis 0."""
    N = 16

    @sar.func
    def k(x: sar.f64[N, N]) -> sar.f64[N, N]:
        return sar.median_filter(x, window=3, dim=0)

    x = _f64(N, N)
    try:
        from scipy.ndimage import median_filter as scipy_mf
        expected = scipy_mf(x, size=(3, 1), mode="nearest")
    except ImportError:
        half = 1
        expected = np.zeros_like(x)
        for i in range(N):
            lo, hi = max(0, i - half), min(N - 1, i + half)
            w = x[lo:hi + 1, :]
            if lo == 0 and hi - lo + 1 < 3:
                w = np.pad(w, [(3 - (hi - lo + 1), 0), (0, 0)], mode="edge")
            elif hi == N - 1 and hi - lo + 1 < 3:
                w = np.pad(w, [(0, 3 - (hi - lo + 1)), (0, 0)], mode="edge")
            expected[i, :] = np.sort(w, axis=0)[1, :]
    np.testing.assert_allclose(k(x), expected, rtol=1e-10)


def test_median_filter_window3_known_values():
    """Check against a hand-computed result: median of three."""
    N = 4

    @sar.func
    def k(x: sar.f64[1, N]) -> sar.f64[1, N]:
        return sar.median_filter(x, window=3, dim=1)

    # input: [1, 3, 2, 4]
    # window with clamp:  [1,1,3], [1,3,2], [3,2,4], [2,4,4]
    # medians:             1,       2,       3,       4
    x = np.array([[1.0, 3.0, 2.0, 4.0]])
    expected = np.array([[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_allclose(k(x), expected, rtol=1e-10)


def test_rank_filter_rank1():
    """rank_filter and median_filter accept rank-1 tensors directly."""
    N = 15

    @sar.func
    def kmin(x: sar.f64[N]) -> sar.f64[N]:
        return sar.rank_filter(x, window=3, rank=0)

    @sar.func
    def kmed(x: sar.f64[N]) -> sar.f64[N]:
        return sar.median_filter(x, window=5)

    x = _f64(N)

    def reference(data, window, rank):
        half = window // 2
        out = np.empty_like(data)
        for i in range(len(data)):
            idx = np.clip(np.arange(i - half, i + half + 1), 0, len(data) - 1)
            out[i] = np.sort(data[idx])[rank]
        return out

    np.testing.assert_allclose(kmin(x), reference(x, 3, 0), rtol=1e-12)
    np.testing.assert_allclose(kmed(x), reference(x, 5, 2), rtol=1e-12)


def test_rank_filter_median_matches_numpy_sort():
    """rank = window//2 and window//2 == 1 for window=3."""
    N = 12

    @sar.func
    def k(x: sar.f32[N, N]) -> sar.f32[N, N]:
        return sar.rank_filter(x, window=3, rank=1, dim=1)

    @sar.func
    def km(x: sar.f32[N, N]) -> sar.f32[N, N]:
        return sar.median_filter(x, window=3, dim=1)

    x = _f32(N, N)
    np.testing.assert_allclose(k(x), km(x), rtol=1e-6)


# --------------------------------------------------------------------- #
# ALOS geometry at a reduced raster (examples/common/params.py)
# --------------------------------------------------------------------- #


def test_alos_params_change_only_the_window_position():
    """A reduced raster keeps every radar parameter; only `t_shift`, where
    R0 sits in the sampled window, follows the raster."""
    from dataclasses import replace

    from common.params import ALOS_PARAMS, alos_params

    assert alos_params(16384) is ALOS_PARAMS
    small = alos_params(1024)
    assert replace(small, t_shift=ALOS_PARAMS.t_shift) == ALOS_PARAMS
    # R0 has to be inside the window, or the echo misses the raster.
    assert 0 < small.t_shift * small.fs < 1024


@pytest.mark.parametrize("algorithm", ["wka", "rda", "csa"])
def test_reduced_alos_geometry_focuses_a_point_target(algorithm):
    """The reduced ALOS raster is a real acquisition geometry, not just a
    smaller array: a scene-center scatterer focuses to scene center."""
    import importlib

    from common.params import alos_params
    from common.simulate import single_target_scene

    n = 512
    params = alos_params(n)
    algo = importlib.import_module(f"{algorithm}.algorithm")
    kernel = algo.build_kernel(n, params).compile("cpu")
    image = kernel(single_target_scene(n, params),
                   *algo.make_inputs(n, params))

    peak = np.unravel_index(np.argmax(image), image.shape)
    assert peak == (n // 2, n // 2)
    # A focused response, not a smear: the peak towers over the mean.
    assert 20.0 * np.log10(image.max() / image.mean()) > 40.0
