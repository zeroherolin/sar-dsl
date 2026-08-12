"""Numerical validation of sar.stolt_interp against a direct numpy port."""

import numpy as np

import sar

from conftest import requires_cpu

pytestmark = requires_cpu


def _stolt_reference(data, fa, fr, c, fc, vr, t_shift):
    n, m = data.shape
    smooth = data * np.exp(1j * 2 * np.pi * fr[None, :] * t_shift)
    out = np.zeros_like(data)
    f_start, df = fr[0], fr[1] - fr[0]
    for i in range(n):
        term = (fr + fc) ** 2 + (c * fa[i] / (2 * vr)) ** 2
        frq = np.sqrt(np.maximum(term, 1e-10)) - fc
        idxf = (frq - f_start) / df
        idx0 = np.floor(idxf).astype(int)
        acc = np.zeros(m, dtype=complex)
        for k in range(-3, 5):
            idx = idx0 + k
            valid = (idx >= 0) & (idx < m)
            idxs = np.clip(idx, 0, m - 1)
            d = idxf - idx
            w = np.sinc(d) * (0.5 + 0.5 * np.cos(np.pi * d / 4.0))
            acc += np.where(valid, smooth[i, idxs] * w, 0)
        out[i] = acc * np.exp(-1j * 2 * np.pi * frq * t_shift)
    return out


def test_stolt_matches_reference():
    n, m = 32, 48
    c, fc, vr, t_shift = 3.0e8, 1.27e9, 7100.0, 1.5e-4

    @sar.jit
    def k(d: sar.c128[n, m], fa: sar.f64[n], fr: sar.f64[m]) -> sar.c128[n, m]:
        return sar.stolt_interp(d, fa, fr, c=c, fc=fc, vr=vr,
                                t_shift=t_shift)

    rng = np.random.default_rng(7)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    fa = np.fft.fftshift(np.fft.fftfreq(n, d=1 / 2000.0))
    fr = np.fft.fftshift(np.fft.fftfreq(m, d=1 / 3.2e7))

    out = k(data, fa, fr)
    ref = _stolt_reference(data, fa, fr, c, fc, vr, t_shift)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
