"""The Range-Doppler Algorithm (RDA) expressed in SAR-DSL.

Demonstrates that the dialect generalizes beyond omega-K: RCMC is built
from the orthogonal `sar.interp1d` primitive plus element-wise position
computation -- no RDA-specific operation exists in the compiler.
"""

from __future__ import annotations

import math

import numpy as np

import sar

from common.params import RadarParams

__all__ = ["build_kernel", "make_inputs"]


def build_kernel(n: int, p: RadarParams) -> sar.Kernel:
    """Builds an `n x n` RDA imaging kernel (azimuth x range).

    Both corrections are range-dependent, as in the textbook algorithm:
    with R = c tau / 2 per range gate,

        RCMC shift (bins)  = lambda^2 R fa^2 / (8 Vr^2) * 2 Fs / c
        azimuth filter     = exp(+j pi fa^2 / Ka(R)),
                             Ka(R) = -2 Vr^2 / (lambda R)
    """

    N = int(n)
    wavelength = p.c / p.fc
    # Per unit (fa^2 * tau): migration in bins and matched-filter phase.
    rcmc_scale = wavelength ** 2 * p.fs / (8.0 * p.vr ** 2)
    az_phase_scale = -math.pi * wavelength * p.c / (4.0 * p.vr ** 2)

    grid = np.arange(N, dtype=np.float64)

    @sar.jit
    def rda(raw: sar.c64[N, N], range_ref: sar.c128[N], fa: sar.f64[N],
            tau: sar.f64[N], win_a: sar.f64[N]) -> sar.f32[N, N]:
        data = sar.cast(raw, sar.c128)

        # Range compression (the window is folded into range_ref on the
        # host).
        spectrum = sar.fft(data, dim=1)
        spectrum = spectrum * sar.broadcast(range_ref, (N, N), dim=1)
        data = sar.ifft(spectrum, dim=1)

        # Into the range-Doppler domain.
        data = sar.fftshift(sar.fft(data, dim=0), dim=0)

        # Range-dependent factors: fa^2 (rows) x tau (columns).
        fa2_tau = (sar.broadcast(fa * fa, (N, N), dim=0)
                   * sar.broadcast(tau, (N, N), dim=1))

        # RCMC: positions = column index + migration shift(fa, R).
        positions = (sar.broadcast(sar.constant(grid), (N, N), dim=1)
                     + fa2_tau * rcmc_scale)
        data = sar.interp1d(data, positions)

        # Azimuth compression + window.
        data = data * sar.expj(fa2_tau * az_phase_scale)
        data = data * sar.cast(sar.broadcast(win_a, (N, N), dim=0), sar.c128)

        # Back to the image domain.
        data = sar.ifft(sar.ifftshift(data, dim=0), dim=0)
        return sar.cast(sar.absolute(data), sar.f32)

    return rda


def make_inputs(n: int, p: RadarParams):
    """Host-precomputed inputs: (range_ref, fa, tau, win_a)."""
    from rda.reference import make_range_reference
    fa = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / p.prf))
    tau = 2.0 * p.r0 / p.c - p.t_shift + np.arange(n) / p.fs
    return make_range_reference(n, p), fa, tau, np.hanning(n)
