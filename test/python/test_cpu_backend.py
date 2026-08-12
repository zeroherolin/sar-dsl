"""CPU backend end-to-end tests: every op validated against numpy."""

import numpy as np
import pytest

import sar

from conftest import requires_cpu

pytestmark = requires_cpu

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

    @sar.jit
    def k(a: sar.f32[N, N], b: sar.f32[N, N]) -> sar.f32[N, N]:
        return (a * 3.0 - b) / 2.0 + 1.0

    a, b = _f32(N, N), _f32(N, N)
    ref = ((a * np.float32(3.0) - b) / np.float32(2.0)) + np.float32(1.0)
    np.testing.assert_allclose(k(a, b), ref, rtol=1e-6)


def test_elementwise_complex():
    N = 8

    @sar.jit
    def k(a: sar.c128[N], b: sar.c128[N]) -> sar.c128[N]:
        return a * b + a / b - b

    a, b = _c128(N), _c128(N)
    np.testing.assert_allclose(k(a, b), a * b + a / b - b, rtol=1e-12)


def test_neg_sqrt_abs_maximum():
    N = 64

    @sar.jit
    def k(a: sar.f64[N]) -> sar.f64[N]:
        return sar.sqrt(sar.maximum(-a, 1e-3))

    a = _f64(N)
    np.testing.assert_allclose(k(a), np.sqrt(np.maximum(-a, 1e-3)),
                               rtol=1e-12)


def test_abs_complex_magnitude():
    N = 32

    @sar.jit
    def k(a: sar.c64[N, N]) -> sar.f32[N, N]:
        return sar.absolute(a)

    a = _c64(N, N)
    np.testing.assert_allclose(k(a), np.abs(a), rtol=1e-5, atol=1e-6)


def test_expj_matches_numpy():
    N = 16

    @sar.jit
    def k(p: sar.f64[N, N]) -> sar.c128[N, N]:
        return sar.expj(p)

    p = 1e6 * _f64(N, N)
    np.testing.assert_allclose(k(p), np.exp(1j * p), rtol=1e-12, atol=1e-12)


def test_cast_widening_and_to_complex():
    N = 8

    @sar.jit
    def k(a: sar.f32[N], b: sar.c64[N]) -> sar.c128[N]:
        return sar.cast(a, sar.c128) + sar.cast(b, sar.c128)

    a, b = _f32(N), _c64(N)
    ref = a.astype(np.complex128) + b.astype(np.complex128)
    np.testing.assert_allclose(k(a, b), ref, rtol=1e-6)


def test_complex_constant():
    N = 4
    w = np.array([1 + 2j, 3 - 1j, 0.5j, 2 + 0j], dtype=np.complex128)

    @sar.jit
    def k(x: sar.c128[N]) -> sar.c128[N]:
        return x * sar.constant(w)

    x = _c128(N)
    np.testing.assert_allclose(k(x), x * w, rtol=1e-12)


def test_large_constant_hex_encoded():
    # Above the hex threshold the dense attribute uses a raw blob; values
    # must survive the round-trip bit-exactly.
    N = 256
    w = RNG.standard_normal(N)

    @sar.jit
    def k(x: sar.f64[N]) -> sar.f64[N]:
        return x + sar.constant(w)

    assert 'dense<"0x' in k.to_mlir()
    x = _f64(N)
    np.testing.assert_array_equal(k(x), x + w)


def test_transpose():
    @sar.jit
    def k(a: sar.f32[4, 8]) -> sar.f32[8, 4]:
        return a.T

    a = _f32(4, 8)
    np.testing.assert_array_equal(k(a), a.T)


def test_broadcast_both_axes():
    N, M = 4, 8

    @sar.jit
    def k(m: sar.f64[N, M], r: sar.f64[M], c: sar.f64[N]) -> sar.f64[N, M]:
        return (m * sar.broadcast(r, (N, M), dim=1)
                + sar.broadcast(c, (N, M), dim=0))

    m, r, c = _f64(N, M), _f64(M), _f64(N)
    np.testing.assert_allclose(k(m, r, c), m * r[None, :] + c[:, None],
                               rtol=1e-12)


@pytest.mark.parametrize("n", [8, 9])  # even and odd sizes
def test_fftshift_conventions(n):
    @sar.jit
    def k(a: sar.f64[n]) -> sar.f64[n]:
        return sar.fftshift(a, dim=0)

    @sar.jit
    def ki(a: sar.f64[n]) -> sar.f64[n]:
        return sar.ifftshift(a, dim=0)

    a = _f64(n)
    np.testing.assert_array_equal(k(a), np.fft.fftshift(a))
    np.testing.assert_array_equal(ki(a), np.fft.ifftshift(a))


@pytest.mark.parametrize("dim", [0, 1])
def test_fft_2d_against_numpy(dim):
    N, M = 32, 64

    @sar.jit
    def k(a: sar.c128[N, M]) -> sar.c128[N, M]:
        return sar.fft(a, dim=dim)

    a = _c128(N, M)
    np.testing.assert_allclose(k(a), np.fft.fft(a, axis=dim), rtol=1e-10,
                               atol=1e-10)


def test_ifft_roundtrip():
    N = 128

    @sar.jit
    def k(a: sar.c128[N]) -> sar.c128[N]:
        return sar.ifft(sar.fft(a, dim=0), dim=0)

    a = _c128(N)
    np.testing.assert_allclose(k(a), a, rtol=1e-12, atol=1e-12)


def test_fft_c64_precision():
    N = 64

    @sar.jit
    def k(a: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.fft(a, dim=1)

    a = _c64(N, N)
    np.testing.assert_allclose(k(a), np.fft.fft(a, axis=1), rtol=1e-4,
                               atol=1e-4)


def test_multiple_results():
    N = 8

    @sar.jit
    def k(a: sar.f32[N], b: sar.f32[N]) -> (sar.f32[N], sar.f32[N]):
        return a + b, a - b

    a, b = _f32(N), _f32(N)
    s, d = k(a, b)
    np.testing.assert_allclose(s, a + b, rtol=1e-6)
    np.testing.assert_allclose(d, a - b, rtol=1e-6)


def test_launcher_validates_arguments():
    N = 8

    @sar.jit
    def k(a: sar.f32[N]) -> sar.f32[N]:
        return a + 1.0

    with pytest.raises(sar.LaunchError, match="shape"):
        k(_f32(N + 1))
    with pytest.raises(sar.LaunchError, match="dtype"):
        k(_f64(N))
    with pytest.raises(sar.LaunchError, match="argument"):
        k(_f32(N), _f32(N))


def test_no_memory_leak_across_launches():
    """Buffer deallocation must free every intermediate: repeated launches
    (including runtime FFT calls) may not grow the process RSS."""
    N = 64

    @sar.jit
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

    @sar.jit
    def k(a: sar.f32[N, N]) -> sar.f32[N, N]:
        return a * 2.0

    big = _f32(2 * N, 2 * N)
    view = big[::2, ::2]  # non-contiguous view
    np.testing.assert_allclose(k(view), view * np.float32(2.0), rtol=1e-6)
