"""NumPy reference implementation of the Polar Format Algorithm.

Mirrors the DSL chain stage by stage, including the 8-tap windowed-sinc
resampling kernel of `sar.interp1d`, so the compiled kernel can be
checked for numerical equivalence.
"""

from __future__ import annotations

import numpy as np

from .geometry import Geometry

__all__ = ["PFAProcessor", "sva_range", "sva2d"]


def _interp_rows(data: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Windowed-sinc resampling of each row (the sar.interp1d kernel)."""
    n = data.shape[1]
    out = np.zeros_like(data)
    for i in range(data.shape[0]):
        idx0 = np.floor(positions[i]).astype(int)
        acc = np.zeros(n, dtype=complex)
        for tap in range(-3, 5):
            idx = idx0 + tap
            valid = (idx >= 0) & (idx < n)
            idxs = np.clip(idx, 0, n - 1)
            d = positions[i] - idx
            w = np.sinc(d) * (0.5 + 0.5 * np.cos(np.pi * d / 4.0))
            acc += np.where(valid, data[i, idxs] * w, 0)
        out[i] = acc
    return out


def _sva_component(center: np.ndarray, q: np.ndarray) -> np.ndarray:
    """SVA weight selection on one I/Q component (mirrors the DSL op)."""
    a = -center / np.where(q == 0.0, 1.0, q)
    inside = np.where(a <= 0.0, center, 0.0)
    picked = np.where(a >= 0.5, center + 0.5 * q, inside)
    return np.where(q == 0.0, center, picked)


def sva_range(img: np.ndarray) -> np.ndarray:
    """Spatially variant apodization along range (2x-oversampled input:
    neighbors one resolution cell = two samples away)."""
    out = img.copy()
    center, q = img[:, 2:-2], img[:, :-4] + img[:, 4:]
    out[:, 2:-2] = (_sva_component(center.real, q.real) +
                    1j * _sva_component(center.imag, q.imag))
    return out


def sva2d(img: np.ndarray) -> np.ndarray:
    """Separable 2-D SVA: the range pass, then the same pass across
    azimuth (Stankwitz's sequential form)."""
    return sva_range(sva_range(img).T).T


class PFAProcessor:
    """Reference PFA processor for square `n x n` collections;
    `process` returns the (uniform, SVA-filtered) magnitude pair the DSL
    kernel produces."""

    def __init__(self, n: int, geom: Geometry):
        self.n = n
        self.geom = geom

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def positions(self):
        """Fractional positions of both regrid passes (the formulas the
        DSL kernel evaluates in-kernel)."""
        g = self.geom
        dk = g.k[1] - g.k[0]
        dtheta = g.theta[1] - g.theta[0]
        k_wanted = g.v[np.newaxis, :] / np.cos(g.theta)[:, np.newaxis]
        pos_range = (k_wanted - g.k[0]) / dk
        theta_wanted = np.arctan2(g.u[:, np.newaxis], g.v[np.newaxis, :])
        pos_cross = (theta_wanted - g.theta[0]) / dtheta
        return pos_range, pos_cross

    def polar_to_rect(self, data: np.ndarray) -> np.ndarray:
        pos_range, pos_cross = self.positions()
        keystone = _interp_rows(data, pos_range)
        return _interp_rows(keystone.T, pos_cross.T).T

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #

    def process_complex(self, raw: np.ndarray) -> np.ndarray:
        assert raw.shape == (self.n, self.n)
        grid = self.polar_to_rect(raw.astype(np.complex128))
        half = self.n // 2
        padded = np.pad(grid, ((half, half), (half, half)))
        image = np.fft.ifft2(np.fft.ifftshift(padded))
        return np.fft.fftshift(image)

    def process(self, raw: np.ndarray):
        image = self.process_complex(raw)
        return np.abs(image), np.abs(sva2d(image))
