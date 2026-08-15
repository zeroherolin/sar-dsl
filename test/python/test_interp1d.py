"""sar.interp1d: numerical validation of every interpolation kernel
against direct numpy ports, plus the identity/edge properties SAR
resampling stages rely on."""

import numpy as np
import pytest

import sar

from conftest import requires_cpu

pytestmark = requires_cpu


def _window(window, t, beta):
    if window == "rect":
        return np.ones_like(t)
    if window == "hann":
        return 0.5 + 0.5 * np.cos(np.pi * t)
    if window == "hamming":
        return 0.54 + 0.46 * np.cos(np.pi * t)
    return np.i0(beta * np.sqrt(np.maximum(1.0 - t * t, 0.0))) / np.i0(beta)


def _interp_reference(data,
                      positions,
                      kernel="sinc",
                      taps=8,
                      window="hann",
                      beta=2.5):
    n, m = data.shape
    out = np.zeros_like(data)
    for i in range(n):
        pos = positions[i]
        if kernel == "nearest":
            idx = np.floor(pos + 0.5).astype(int)
            valid = (idx >= 0) & (idx < m)
            out[i] = np.where(valid, data[i, np.clip(idx, 0, m - 1)], 0)
            continue
        idx0 = np.floor(pos).astype(int)
        frac = pos - idx0
        if kernel == "linear":
            offsets, weights = [0, 1], [1.0 - frac, frac]
        elif kernel == "cubic":
            s, s2, s3 = frac, frac**2, frac**3
            offsets = [-1, 0, 1, 2]
            weights = [
                -0.5 * s3 + s2 - 0.5 * s, 1.5 * s3 - 2.5 * s2 + 1.0,
                -1.5 * s3 + 2.0 * s2 + 0.5 * s, 0.5 * s3 - 0.5 * s2
            ]
        else:
            half = taps // 2
            offsets = list(range(1 - half, half + 1))
            weights = [
                np.sinc(pos - (idx0 + k)) *
                _window(window, (pos - (idx0 + k)) / half, beta)
                for k in offsets
            ]
        acc = np.zeros(m, dtype=complex)
        for k, w in zip(offsets, weights):
            idx = idx0 + k
            valid = (idx >= 0) & (idx < m)
            acc += np.where(valid, data[i, np.clip(idx, 0, m - 1)] * w, 0)
        out[i] = acc
    return out


def _sinc_interp_reference(data, positions):
    return _interp_reference(data, positions)


def test_interp1d_matches_reference():
    n, m = 16, 48
    rng = np.random.default_rng(3)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    positions = rng.uniform(-2.0, m + 1.0, size=(n, m))

    @sar.func
    def k(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
        return sar.interp1d(d, p)

    np.testing.assert_allclose(k(data, positions),
                               _sinc_interp_reference(data, positions),
                               rtol=1e-12,
                               atol=1e-12)


@pytest.mark.parametrize("kwargs", [
    dict(kernel="nearest"),
    dict(kernel="linear"),
    dict(kernel="cubic"),
    dict(kernel="sinc", taps=16, window="hamming"),
    dict(kernel="sinc", taps=8, window="rect"),
    dict(kernel="sinc", taps=16, window="kaiser", beta=4.0),
])
def test_interp_kernel_selection(kwargs):
    """Each selectable kernel matches its direct numpy port, including
    out-of-range positions (zero contribution)."""
    n, m = 12, 40
    rng = np.random.default_rng(11)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    positions = rng.uniform(-2.0, m + 1.0, size=(n, m))

    @sar.func
    def k(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
        return sar.interp1d(d, p, **kwargs)

    np.testing.assert_allclose(k(data, positions),
                               _interp_reference(data, positions, **kwargs),
                               rtol=1e-12,
                               atol=1e-12)


def test_interp_kernel_validation():
    """Bad kernel parameters are rejected at trace time."""
    n, m = 4, 8

    def build(**kwargs):

        @sar.func
        def k(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
            return sar.interp1d(d, p, **kwargs)

        return k.to_mlir()

    from sar.language import TraceError
    with pytest.raises(TraceError, match="kernel must be"):
        build(kernel="spline")
    with pytest.raises(TraceError, match="taps must be even"):
        build(taps=7)
    with pytest.raises(TraceError, match="window must be"):
        build(window="blackman")
    with pytest.raises(TraceError, match="beta must be"):
        build(window="kaiser", beta=0.0)


def test_interp1d_identity_positions():
    """Integer positions on the grid reproduce the input exactly
    (sinc(0)=1, all other taps hit sinc(k)=0)."""
    n, m = 8, 32
    rng = np.random.default_rng(4)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    positions = np.tile(np.arange(m, dtype=np.float64), (n, 1))

    @sar.func
    def k(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
        return sar.interp1d(d, p)

    np.testing.assert_allclose(k(data, positions),
                               data,
                               rtol=1e-12,
                               atol=1e-12)
