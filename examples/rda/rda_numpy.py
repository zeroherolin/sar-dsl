"""NumPy reference implementation of the Range-Doppler Algorithm (RDA).

Zero-Doppler-centroid stripmap RDA:

    1. range compression (frequency-domain matched filter)
    2. azimuth FFT (into the range-Doppler domain)
    3. range cell migration correction (RCMC, windowed-sinc resampling)
    4. azimuth compression (Doppler-domain matched filter)
    5. azimuth IFFT

Shares the acquisition-parameter dataclass with the omega-K example.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wka"))

from wka_numpy import WKAParams  # noqa: E402

__all__ = ["RDAProcessor", "make_range_reference"]


def make_range_reference(n: int, p: WKAParams) -> np.ndarray:
    """Frequency-domain range matched filter (unshifted FFT layout).

    The chirp replica matches the synthetic echo model: duration half the
    sampled window, centered in fast time; it is rolled to start at sample
    zero so that compressed peaks appear at the echo delay position.
    """
    t = (np.arange(n) - n / 2) / p.fs
    pulse_len = 0.5 * n / p.fs
    replica = np.where(np.abs(t) <= pulse_len / 2.0,
                       np.exp(1j * np.pi * p.kr * t ** 2), 0.0)
    replica = np.roll(replica, -(n // 2))
    return np.conj(np.fft.fft(replica))


class RDAProcessor:
    """Reference RDA processor for square `n x n` rasters
    (azimuth x range)."""

    def __init__(self, n: int, params: WKAParams):
        self.n = n
        self.p = params
        self.wavelength = params.c / params.fc
        self.fa = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / params.prf))
        self.range_ref = make_range_reference(n, params)

    # ------------------------------------------------------------------ #

    def range_compress(self, data: np.ndarray) -> np.ndarray:
        spectrum = np.fft.fft(data, axis=1) * self.range_ref[np.newaxis, :]
        return np.fft.ifft(spectrum, axis=1)

    def rcmc_positions(self) -> np.ndarray:
        """Fractional range positions per (Doppler bin, range bin)."""
        p = self.p
        delta_r = (self.wavelength ** 2 * p.r0 * self.fa ** 2
                   / (8.0 * p.vr ** 2))
        delta_bins = delta_r * 2.0 * p.fs / p.c
        cols = np.arange(self.n, dtype=np.float64)
        return cols[np.newaxis, :] + delta_bins[:, np.newaxis]

    def rcmc(self, data: np.ndarray) -> np.ndarray:
        positions = self.rcmc_positions()
        out = np.zeros_like(data)
        for i in range(self.n):
            idx0 = np.floor(positions[i]).astype(int)
            acc = np.zeros(self.n, dtype=complex)
            for k in range(-3, 5):
                idx = idx0 + k
                valid = (idx >= 0) & (idx < self.n)
                idxs = np.clip(idx, 0, self.n - 1)
                d = positions[i] - idx
                w = np.sinc(d) * (0.5 + 0.5 * np.cos(np.pi * d / 4.0))
                acc += np.where(valid, data[i, idxs] * w, 0)
            out[i] = acc
        return out

    def azimuth_filter(self) -> np.ndarray:
        """Doppler-domain azimuth matched filter (zero Doppler centroid).

        The echo azimuth phase is -4 pi R(eta) / lambda with the hyperbolic
        range history, i.e. an FM rate Ka = -2 Vr^2 / (lambda R0). By the
        stationary-phase approximation the chirp spectrum carries the phase
        exp(-j pi fa^2 / Ka), so the matched filter is its conjugate,
        exp(+j pi fa^2 / Ka)."""
        p = self.p
        ka = -2.0 * p.vr ** 2 / (self.wavelength * p.r0)
        return np.exp(1j * np.pi * self.fa ** 2 / ka)

    # ------------------------------------------------------------------ #

    def process_complex(self, raw: np.ndarray) -> np.ndarray:
        assert raw.shape == (self.n, self.n)
        data = raw.astype(np.complex128)
        data = self.range_compress(data)
        data = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
        data = self.rcmc(data)
        data = data * self.azimuth_filter()[:, np.newaxis]
        data = np.fft.ifft(np.fft.ifftshift(data, axes=0), axis=0)
        return data

    def process(self, raw: np.ndarray) -> np.ndarray:
        return np.abs(self.process_complex(raw)).astype(np.float32)
