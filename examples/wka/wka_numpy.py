"""NumPy reference implementation of the omega-K (WKA) imaging algorithm.

This is the numerical ground truth the SAR-DSL kernel is validated against.
The processing chain follows the classic wavenumber-domain algorithm:

    1. range FFT (+ spectrum shift)
    2. azimuth FFT (+ spectrum shift), via corner turns
    3. bulk compression (reference function multiply)
    4. Stolt interpolation (frequency remapping)
    5. range / azimuth windowing
    6. 2-D inverse FFT back to the image domain

All spectra are kept in the numpy `fftshift` layout between stages, matching
the `sar.fftshift` conventions of the DSL kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WKAParams", "WKAProcessor", "ALOS_PARAMS"]


@dataclass(frozen=True)
class WKAParams:
    """Radar/platform parameters of a stripmap acquisition."""

    c: float    # propagation speed (m/s)
    fc: float   # carrier frequency (Hz)
    fs: float   # range sampling rate (Hz)
    prf: float  # pulse repetition frequency (Hz)
    vr: float   # effective radar velocity (m/s)
    r0: float   # reference slant range (m)
    kr: float   # range chirp rate (Hz/s)
    t_shift: float  # reference fast-time shift used by Stolt interpolation (s)


#: ALOS-1 PALSAR parameters used by the San Francisco dataset.
ALOS_PARAMS = WKAParams(
    c=299792458.0,
    fc=1269999750.06,
    fs=32000000.00,
    prf=2155.172,
    vr=7155.0,
    r0=843013.994,
    kr=-1.037e12,
    t_shift=4800.0 / 32000000.00,
)


class WKAProcessor:
    """Reference omega-K processor for square `n x n` rasters."""

    def __init__(self, n: int, params: WKAParams):
        self.n = n
        self.p = params
        # Azimuth / range frequency axes in fftshift layout.
        self.fa = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / params.prf))
        self.fr = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / params.fs))
        self.win_range = np.hanning(n)
        self.win_azimuth = np.hanning(n)

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fft_axis1(data: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)

    @staticmethod
    def _ifft_axis1(data: np.ndarray) -> np.ndarray:
        return np.fft.ifft(np.fft.ifftshift(data, axes=1), axis=1)

    def bulk_compression(self, data: np.ndarray) -> np.ndarray:
        p = self.p
        fa2 = self.fa[:, np.newaxis]
        fr2 = self.fr[np.newaxis, :]
        term1 = (p.fc + fr2) ** 2
        term2 = (p.c * fa2 / (2.0 * p.vr)) ** 2
        root = np.sqrt(np.maximum(term1 - term2, 1e-10))
        phase = ((4.0 * np.pi * p.r0 / p.c) * (root - (p.fc + fr2))
                 + np.pi * fr2 ** 2 / p.kr)
        return data * np.exp(1j * phase)

    def stolt_interpolation(self, data: np.ndarray) -> np.ndarray:
        p = self.p
        n = self.n
        fr = self.fr
        smooth = data * np.exp(1j * 2.0 * np.pi * fr[np.newaxis, :]
                               * p.t_shift)

        out = np.zeros_like(data)
        f_start = fr[0]
        df = fr[1] - fr[0]
        for i in range(n):
            term = ((fr + p.fc) ** 2
                    + (p.c * self.fa[i] / (2.0 * p.vr)) ** 2)
            fr_query = np.sqrt(np.maximum(term, 1e-10)) - p.fc
            idx_float = (fr_query - f_start) / df
            idx_int = np.floor(idx_float).astype(np.int64)

            acc = np.zeros(n, dtype=np.complex128)
            for k in range(-3, 5):
                idx_k = idx_int + k
                valid = (idx_k >= 0) & (idx_k < n)
                idx_safe = np.clip(idx_k, 0, n - 1)
                dist = idx_float - idx_k
                weight = np.sinc(dist) * (0.5 + 0.5 * np.cos(np.pi * dist
                                                             / 4.0))
                acc += np.where(valid, smooth[i, idx_safe] * weight, 0.0)
            out[i] = acc * np.exp(-1j * 2.0 * np.pi * fr_query * p.t_shift)
        return out

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #

    def process_complex(self, raw: np.ndarray) -> np.ndarray:
        """Runs the full chain, returning the complex focused image."""
        assert raw.shape == (self.n, self.n)
        data = raw.astype(np.complex128)

        data = self._fft_axis1(data)              # range FFT
        data = data.T                             # corner turn
        data = self._fft_axis1(data)              # azimuth FFT
        data = data.T                             # corner turn
        data = self.bulk_compression(data)
        data = self.stolt_interpolation(data)
        data = data * self.win_range[np.newaxis, :]
        data = data.T
        data = data * self.win_azimuth[np.newaxis, :]
        data = data.T
        data = self._ifft_axis1(data)             # range IFFT
        data = data.T
        data = self._ifft_axis1(data)             # azimuth IFFT
        data = data.T
        return data

    def process(self, raw: np.ndarray) -> np.ndarray:
        """Runs the full chain, returning the magnitude image (float32)."""
        return np.abs(self.process_complex(raw)).astype(np.float32)
