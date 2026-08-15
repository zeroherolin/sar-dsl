"""NumPy reference implementation of the Chirp Scaling Algorithm (CSA).

CSA (Raney et al. 1994; Cumming & Wong ch. 7) focuses raw stripmap data
without any interpolation: range cell migration is equalized by a
quadratic phase multiply (chirp scaling) in the range-Doppler domain, and
all remaining corrections are phase multiplies in the 2-D frequency and
range-Doppler domains. Zero Doppler centroid is assumed.

    1. azimuth FFT
    2. chirp scaling multiply           (range-Doppler domain)
    3. range FFT
    4. range compression + SRC + bulk RCMC multiply   (2-D frequency)
    5. range IFFT
    6. azimuth compression multiply     (range-Doppler domain)
    7. azimuth IFFT

Sign conventions follow the echo model of the point-target simulator
(transmitted chirp `exp(+j pi Kr t^2)`, azimuth phase `-4 pi fc R / c`);
the bulk-RCMC sign is derived in `rc_src_rcmc_phase`.
"""

from __future__ import annotations

import numpy as np

from common.params import RadarParams, band_windows

__all__ = ["CSAProcessor"]


class CSAProcessor:
    """Reference CSA processor for square `n x n` rasters
    (azimuth x range)."""

    def __init__(self, n: int, params: RadarParams):
        self.n = n
        self.p = params
        self.wavelength = params.c / params.fc
        self.fa = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / params.prf))
        self.fr = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / params.fs))
        # Absolute fast time: R0 sits at offset t_shift into the window.
        self.tau = (2.0 * params.r0 / params.c - params.t_shift +
                    np.arange(n) / params.fs)
        self.win_range, self.win_azimuth = band_windows(n, params)

    # ------------------------------------------------------------------ #
    # Doppler-dependent factors
    # ------------------------------------------------------------------ #

    def migration_factor(self) -> np.ndarray:
        """D(fa) = sqrt(1 - (lambda fa / 2 Vr)^2)."""
        sin_theta = self.wavelength * self.fa / (2.0 * self.p.vr)
        return np.sqrt(np.maximum(1.0 - sin_theta**2, 1e-10))

    def modified_chirp_rate(self, d: np.ndarray) -> np.ndarray:
        """Km(fa): range FM rate modified by range-azimuth coupling."""
        p = self.p
        coupling = (p.kr * p.c * p.r0 * self.fa**2 /
                    (2.0 * p.vr**2 * p.fc**3 * d**3))
        return p.kr / (1.0 - coupling)

    # ------------------------------------------------------------------ #
    # Phase terms (each returns the real phase; the data is multiplied by
    # exp(+j phase))
    # ------------------------------------------------------------------ #

    def chirp_scaling_phase(self, d, km) -> np.ndarray:
        """Equalizes all migration trajectories to the reference one."""
        tau_ref = 2.0 * self.p.r0 / (self.p.c * d[:, None])
        return (np.pi * km[:, None] * (1.0 / d[:, None] - 1.0) *
                (self.tau[None, :] - tau_ref)**2)

    def rc_src_rcmc_phase(self, d, km) -> np.ndarray:
        """Range compression + secondary range compression + bulk RCMC.

        The quadratic term conjugates the (Km-modified) chirp spectrum.
        The linear term shifts every target *earlier* in fast time by
        dt = 2 R0 (1/D - 1) / c -- targets appear at range R0/D, and
        x(t - dt) <-> X(f) exp(-j 2 pi f dt) gives the +j sign here.
        """
        d2 = d[:, None]
        fr2 = self.fr[None, :]
        quadratic = np.pi * fr2**2 * d2 / km[:, None]
        linear = (4.0 * np.pi * self.p.r0 / self.p.c) * (1.0 / d2 - 1.0) * fr2
        return quadratic + linear

    def azimuth_compression_phase(self, d) -> np.ndarray:
        """Conjugates the hyperbolic azimuth phase per range gate
        (the fa-independent carrier is dropped: magnitude imaging)."""
        r_gate = self.p.c * self.tau[None, :] / 2.0
        return (4.0 * np.pi * self.p.fc / self.p.c) * r_gate * (d[:, None] -
                                                                1.0)

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #

    def process_complex(self, raw: np.ndarray) -> np.ndarray:
        assert raw.shape == (self.n, self.n)
        d = self.migration_factor()
        km = self.modified_chirp_rate(d)

        data = np.fft.fftshift(np.fft.fft(raw.astype(np.complex128), axis=0),
                               axes=0)
        data = data * np.exp(1j * self.chirp_scaling_phase(d, km))
        data = np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)
        data = data * np.exp(1j * self.rc_src_rcmc_phase(d, km))
        data = data * self.win_range[np.newaxis, :]
        data = np.fft.ifft(np.fft.ifftshift(data, axes=1), axis=1)
        data = data * np.exp(1j * self.azimuth_compression_phase(d))
        data = data * self.win_azimuth[:, np.newaxis]
        return np.fft.ifft(np.fft.ifftshift(data, axes=0), axis=0)

    def process(self, raw: np.ndarray) -> np.ndarray:
        return np.abs(self.process_complex(raw)).astype(np.float32)
