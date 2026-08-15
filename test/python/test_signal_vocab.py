"""The composed signal-processing vocabulary (`sar.language.signal`):
Matlab/scipy-familiar functions that decompose into IR primitives at
trace time, each validated against its numpy semantic.
"""

import numpy as np

import sar

from conftest import requires_cpu

pytestmark = requires_cpu

N, M = 8, 16
RNG = np.random.default_rng(3001)


def _complex(n, m):
    return RNG.standard_normal((n, m)) + 1j * RNG.standard_normal((n, m))


def test_fft2_ifft2_and_full_fftshift():
    z = _complex(N, M)

    @sar.func
    def k(z):
        return sar.fftshift(sar.fft2(z)), sar.ifft2(sar.fft2(z))

    shifted, roundtrip = k(z)
    np.testing.assert_allclose(shifted,
                               np.fft.fftshift(np.fft.fft2(z)),
                               rtol=1e-10,
                               atol=1e-10)
    np.testing.assert_allclose(roundtrip, z, rtol=1e-10, atol=1e-10)


def test_flip_and_circshift():
    x = RNG.standard_normal((N, M))

    @sar.func
    def k(x):
        return (sar.flip(x, axis=1), sar.flip(x, axis=0),
                sar.circshift(x, 3, axis=1), sar.circshift(x, -2, axis=0))

    f1, f0, c1, c0 = k(x)
    np.testing.assert_allclose(f1, np.flip(x, axis=1), rtol=0, atol=0)
    np.testing.assert_allclose(f0, np.flip(x, axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(c1, np.roll(x, 3, axis=1), rtol=0, atol=0)
    np.testing.assert_allclose(c0, np.roll(x, -2, axis=0), rtol=0, atol=0)


def test_db_conversions_and_logs():
    x = np.abs(RNG.standard_normal((N, M))) + 0.1

    @sar.func
    def k(x):
        return (sar.mag2db(x), sar.pow2db(x), sar.log10(x), sar.log2(x))

    m, p, l10, l2 = k(x)
    np.testing.assert_allclose(m, 20 * np.log10(x), rtol=1e-12)
    np.testing.assert_allclose(p, 10 * np.log10(x), rtol=1e-12)
    np.testing.assert_allclose(l10, np.log10(x), rtol=1e-12)
    np.testing.assert_allclose(l2, np.log2(x), rtol=1e-12)


def test_sinc_and_hypot():
    x = np.concatenate([[0.0], RNG.standard_normal(M - 1) * 3])
    a, b = RNG.standard_normal(M), RNG.standard_normal(M)

    @sar.func
    def k(x, a, b):
        return sar.sinc(x), sar.hypot(a, b)

    s, h = k(x, a, b)
    np.testing.assert_allclose(s, np.sinc(x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h, np.hypot(a, b), rtol=1e-12)


def test_mean():
    x = RNG.standard_normal((N, M))

    @sar.func
    def k(x):
        return sar.mean(x, axis=1), sar.mean(x)

    per_row, overall = k(x)
    np.testing.assert_allclose(per_row, x.mean(axis=1), rtol=1e-12)
    np.testing.assert_allclose(overall, [x.mean()], rtol=1e-12)


def test_dechirp_and_matched_filter():
    z = _complex(N, M)
    ref = np.exp(1j * np.linspace(0.0, 4.0, M))

    @sar.func
    def k(z):
        return sar.dechirp(z, ref), sar.matched_filter(z, ref, axis=1)

    mixed, filtered = k(z)
    np.testing.assert_allclose(mixed, z * np.conj(ref), rtol=1e-12)
    want = np.fft.ifft(np.fft.fft(z, axis=1) * np.conj(np.fft.fft(ref)),
                       axis=1)
    np.testing.assert_allclose(filtered, want, rtol=1e-10, atol=1e-10)


def test_windows_are_baked_constants():
    x = RNG.standard_normal((N, M))

    @sar.func
    def k(x):
        return (x * sar.taylorwin(M, sll=-35.0), x * sar.hamming(M),
                x * sar.kaiser(M, beta=4.0))

    taylor, hamming, kaiser = k(x)
    np.testing.assert_allclose(hamming, x * np.hamming(M), rtol=1e-12)
    np.testing.assert_allclose(kaiser, x * np.kaiser(M, 4.0), rtol=1e-12)

    # Taylor: unit peak, -35 dB target keeps the window strictly inside
    # (0, 1] and symmetric.
    w = taylor[0] / x[0]
    np.testing.assert_allclose(w, w[::-1], rtol=1e-9)
    assert np.isclose(w.max(), 1.0) and w.min() > 0.0


def test_vocabulary_decomposes_to_primitives():

    @sar.func
    def k(z: sar.c64[N, M]) -> sar.f32[N, M]:
        return sar.mag2db(abs(sar.fftshift(sar.fft2(z))) + 1e-6)

    text = k.to_mlir()
    # Compositions decompose into IR primitives: no composite op names.
    assert '"sar.fft"' in text and '"sar.log"' in text
    assert "fft2" not in text and "mag2db" not in text


def test_std_var_match_numpy():
    """std / var (numpy convention: normalized by N)."""
    rng = np.random.default_rng(31)
    a = rng.uniform(0.5, 4.0, (N, M))

    @sar.func
    def k(x):
        return (sar.var(x), sar.std(x), sar.var(x, axis=1), sar.std(x, axis=0))

    v_all, s_all, v_r, s_a = k(a)
    np.testing.assert_allclose(v_all, [np.var(a)], rtol=1e-10)
    np.testing.assert_allclose(s_all, [np.std(a)], rtol=1e-10)
    np.testing.assert_allclose(v_r, np.var(a, axis=1), rtol=1e-10)
    np.testing.assert_allclose(s_a, np.std(a, axis=0), rtol=1e-10)


def test_db_conversions_roundtrip():
    """mag2db/db2mag and pow2db/db2pow are inverse pairs."""
    rng = np.random.default_rng(32)
    a = rng.uniform(0.1, 100.0, (N, M))

    @sar.func
    def k(x):
        return (sar.db2mag(sar.mag2db(x)), sar.db2pow(sar.pow2db(x)),
                sar.db2mag(x * 0.0 + 20.0))

    mag_rt, pow_rt, twenty_db = k(a)
    np.testing.assert_allclose(mag_rt, a, rtol=1e-12)
    np.testing.assert_allclose(pow_rt, a, rtol=1e-12)
    np.testing.assert_allclose(twenty_db, np.full((N, M), 10.0), rtol=1e-12)


def test_vocabulary_is_uniformly_callable_on_arrays():
    """Every name in the vocabulary works outside a kernel, on numpy
    arrays, exactly like the scipy/Matlab function it mirrors.

    The split used to be arbitrary -- `sar.sinc(arr)` worked while
    `sar.mean(arr)` raised -- because only some entries carried @sar.op.
    """
    rng = np.random.default_rng(5)
    vec = rng.standard_normal(8)
    mat = rng.standard_normal((2, 4))
    cplx = (rng.standard_normal((4, 8)) + 1j * rng.standard_normal(
        (4, 8))).astype(np.complex128)
    ref = (rng.standard_normal(8) + 1j * rng.standard_normal(8)).astype(
        np.complex128)

    np.testing.assert_allclose(sar.circshift(vec, 3, dim=0), np.roll(vec, 3))
    np.testing.assert_allclose(sar.mean(mat, axis=1),
                               mat.mean(axis=1),
                               rtol=1e-12)
    np.testing.assert_allclose(sar.var(mat, axis=1),
                               mat.var(axis=1),
                               rtol=1e-12)
    np.testing.assert_allclose(sar.std(mat, axis=1),
                               mat.std(axis=1),
                               rtol=1e-12)
    np.testing.assert_allclose(sar.hypot(vec, vec),
                               np.hypot(vec, vec),
                               rtol=1e-12)
    np.testing.assert_allclose(sar.fft2(cplx), np.fft.fft2(cplx), rtol=1e-10)

    # Shape-only checks for the radar staples.
    assert sar.dechirp(cplx, ref).shape == cplx.shape
    assert sar.matched_filter(cplx, ref, dim=1).shape == cplx.shape
    assert sar.multilook(mat, (2, 2)).shape == (1, 2)


def test_windows_are_constructors_not_operators():
    """Windows carry @sar.op like the rest of the vocabulary, but they take
    a length rather than a tensor, so there is nothing to specialize on:
    the decorator passes them straight through. Outside a kernel that
    hands back the numpy samples (usable in host code); inside one they
    bake in as a constant."""
    w = sar.hanning(8)
    assert isinstance(w, np.ndarray)
    np.testing.assert_allclose(w, np.hanning(8), rtol=1e-12)
    np.testing.assert_allclose(sar.kaiser(8, 4.0),
                               np.kaiser(8, 4.0),
                               rtol=1e-12)
    assert isinstance(sar.window("taylor", 8, sll=-35.0), np.ndarray)

    @sar.func
    def tapered(x: sar.f64[8]) -> sar.f64[8]:
        return x * sar.hanning(8)

    np.testing.assert_allclose(tapered(np.ones(8)), np.hanning(8), rtol=1e-12)


def test_operator_keywords_specialize_independently():
    """Keyword arguments are compile-time constants, so each combination
    gets its own specialization rather than leaking the first one."""
    rng = np.random.default_rng(6)
    mat = rng.standard_normal((4, 4))

    np.testing.assert_allclose(sar.mean(mat, axis=0),
                               mat.mean(axis=0),
                               rtol=1e-12)
    np.testing.assert_allclose(sar.mean(mat, axis=1),
                               mat.mean(axis=1),
                               rtol=1e-12)
    np.testing.assert_allclose(sar.circshift(mat, 1, dim=0),
                               np.roll(mat, 1, axis=0))
    np.testing.assert_allclose(sar.circshift(mat, 1, dim=1),
                               np.roll(mat, 1, axis=1))


def test_whole_vocabulary_carries_the_op_decorator():
    """Every public name in the vocabulary is spelled `@sar.op`.

    The uniformity is the point: users should not have to learn which
    entries are "real" operators and which are helpers. Constructors
    (windows) carry it too -- the decorator detects that there is no
    tensor argument and passes them through.
    """
    import inspect
    import re

    from sar.language import signal

    source = inspect.getsource(signal).splitlines()
    undecorated = []
    for i, line in enumerate(source):
        match = re.match(r"def (\w+)\(", line)
        if not match or match.group(1).startswith("_"):
            continue
        window = source[max(0, i - 3):i]
        if not any(w.lstrip().startswith("@sar.op") for w in window):
            undecorated.append(match.group(1))
    assert not undecorated, f"missing @sar.op: {undecorated}"
