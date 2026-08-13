"""sar.interp1d: numerical validation and orthogonality.

The key property: `sar.stolt_interp` must be exactly expressible as
element-wise position computation + `sar.interp1d` + phase multiplies,
proving interp1d is the right primitive for SAR resampling stages.
"""

import math

import numpy as np

import sar

from conftest import requires_cpu

pytestmark = requires_cpu


def _sinc_interp_reference(data, positions):
    n, m = data.shape
    out = np.zeros_like(data)
    for i in range(n):
        idx0 = np.floor(positions[i]).astype(int)
        acc = np.zeros(m, dtype=complex)
        for k in range(-3, 5):
            idx = idx0 + k
            valid = (idx >= 0) & (idx < m)
            idxs = np.clip(idx, 0, m - 1)
            d = positions[i] - idx
            w = np.sinc(d) * (0.5 + 0.5 * np.cos(np.pi * d / 4.0))
            acc += np.where(valid, data[i, idxs] * w, 0)
        out[i] = acc
    return out


def test_interp1d_matches_reference():
    n, m = 16, 48
    rng = np.random.default_rng(3)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    positions = rng.uniform(-2.0, m + 1.0, size=(n, m))

    @sar.jit
    def k(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
        return sar.interp1d(d, p)

    np.testing.assert_allclose(k(data, positions),
                               _sinc_interp_reference(data, positions),
                               rtol=1e-12, atol=1e-12)


def test_interp1d_identity_positions():
    """Integer positions on the grid reproduce the input exactly
    (sinc(0)=1, all other taps hit sinc(k)=0)."""
    n, m = 8, 32
    rng = np.random.default_rng(4)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    positions = np.tile(np.arange(m, dtype=np.float64), (n, 1))

    @sar.jit
    def k(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
        return sar.interp1d(d, p)

    np.testing.assert_allclose(k(data, positions), data, rtol=1e-12,
                               atol=1e-12)


def test_stolt_expressible_via_interp1d():
    """stolt_interp == smoothing multiply + position computation + interp1d
    + de-smoothing multiply, all built from existing DSL ops."""
    n, m = 32, 64
    c, fc, vr, t_shift = 3.0e8, 1.27e9, 7100.0, 1.5e-4

    rng = np.random.default_rng(5)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    fa = np.fft.fftshift(np.fft.fftfreq(n, d=1 / 2000.0))
    fr = np.fft.fftshift(np.fft.fftfreq(m, d=1 / 3.2e7))
    f_start, df = fr[0], fr[1] - fr[0]

    @sar.jit
    def dedicated(d: sar.c128[n, m], fa_: sar.f64[n],
                  fr_: sar.f64[m]) -> sar.c128[n, m]:
        return sar.stolt_interp(d, fa_, fr_, c=c, fc=fc, vr=vr,
                                t_shift=t_shift)

    @sar.jit
    def composed(d: sar.c128[n, m], fa_: sar.f64[n],
                 fr_: sar.f64[m]) -> sar.c128[n, m]:
        fr2 = sar.broadcast(fr_, (n, m), dim=1)
        fa2 = sar.broadcast(fa_, (n, m), dim=0)
        # Smoothing phase ramp exp(+2 pi j fr t_shift).
        smoothed = d * sar.expj(fr2 * (2.0 * math.pi * t_shift))
        # Stolt frequency mapping and fractional positions.
        shifted = fr2 + fc
        fa_term = fa2 * (c / (2.0 * vr))
        f_query = sar.sqrt(sar.maximum(shifted * shifted + fa_term * fa_term,
                                       1e-10)) - fc
        positions = (f_query - f_start) / df
        remapped = sar.interp1d(smoothed, positions)
        # De-smoothing ramp on the output-grid frequency.
        return remapped * sar.expj(fr2 * (-2.0 * math.pi * t_shift))

    a = dedicated(data, fa, fr)
    b = composed(data, fa, fr)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
