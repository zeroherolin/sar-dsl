"""Numerical validation of sar.stolt_interp -- a composition of the
in-kernel position computation, `sar.interp1d` and the re-referencing
ramps -- against a direct numpy port of the fused formula."""

import numpy as np
import pytest

import sar

from conftest import requires_cpu

pytestmark = requires_cpu


def _stolt_reference(data, fa, fr, c, fc, vr, t_shift):
    n, m = data.shape
    smooth = data * np.exp(1j * 2 * np.pi * fr[None, :] * t_shift)
    out = np.zeros_like(data)
    f_start, df = fr[0], fr[1] - fr[0]
    for i in range(n):
        term = (fr + fc)**2 + (c * fa[i] / (2 * vr))**2
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
        out[i] = acc * np.exp(-1j * 2 * np.pi * fr * t_shift)
    return out


def test_stolt_matches_reference():
    n, m = 32, 48
    c, fc, vr, t_shift = 3.0e8, 1.27e9, 7100.0, 1.5e-4
    fa = np.fft.fftshift(np.fft.fftfreq(n, d=1 / 2000.0))
    fr = np.fft.fftshift(np.fft.fftfreq(m, d=1 / 3.2e7))

    @sar.func
    def k(d: sar.c128[n, m]) -> sar.c128[n, m]:
        # The axes bake into the kernel as constants.
        return sar.stolt_interp(d, fa, fr, c=c, fc=fc, vr=vr, t_shift=t_shift)

    rng = np.random.default_rng(7)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))

    out = k(data)
    ref = _stolt_reference(data, fa, fr, c, fc, vr, t_shift)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case,match", [
    ("nonuniform", "uniformly spaced"),
    ("nonfinite", "must be finite"),
    ("zero_vr", "vr must be nonzero"),
    ("bad_c", "c must be positive"),
])
def test_stolt_rejects_invalid_geometry(case, match):
    n, m = 4, 8
    fa = np.linspace(-1.0, 1.0, n)
    fr = np.linspace(-2.0, 2.0, m)
    kwargs = dict(c=3.0e8, fc=1.27e9, vr=7100.0, t_shift=0.0)
    if case == "nonuniform":
        fr[3] += 0.25
    elif case == "nonfinite":
        fa[0] = np.nan
    elif case == "zero_vr":
        kwargs["vr"] = 0.0
    else:
        kwargs["c"] = -1.0

    @sar.func
    def kernel(data: sar.c128[n, m]) -> sar.c128[n, m]:
        return sar.stolt_interp(data, fa, fr, **kwargs)

    with pytest.raises(sar.TraceError, match=match):
        kernel.to_mlir()
