"""The Range-Doppler Algorithm (RDA) expressed in SAR-DSL.

Range cell migration correction is built from the `sar.interp1d`
primitive fed by an element-wise position computation, so the chain uses
the same orthogonal operations as the other algorithms.
"""

import math

import numpy as np

import sar

from common.params import RadarParams, band_windows

__all__ = ["build_kernel", "make_inputs"]


def build_kernel(n: int,
                 p: RadarParams,
                 name: str = "rda",
                 dtype=sar.c128) -> sar.Kernel:
    """Builds an `n x n` RDA imaging kernel (azimuth x range).

    Both corrections are range-dependent, as in the textbook algorithm:
    with R = c tau / 2 per range gate,

        RCMC shift (bins)  = lambda^2 R fa^2 / (8 Vr^2) * 2 Fs / c
        azimuth filter     = exp(+j pi fa^2 / Ka(R)),
                             Ka(R) = -2 Vr^2 / (lambda R)

    `name` becomes the IR symbol and the HLS top function, so a second
    kernel can be emitted beside the first without a symbol clash.
    `dtype` selects the spectral working precision (`sar.c128` default,
    `sar.c64` for a single-precision data path).
    """
    if dtype not in (sar.c128, sar.c64):
        raise ValueError("dtype must be sar.c128 or sar.c64")
    fd = sar.f64 if dtype is sar.c128 else sar.f32

    N = int(n)
    wavelength = p.c / p.fc
    # Per unit (fa^2 * tau): migration in bins and matched-filter phase.
    rcmc_scale = wavelength**2 * p.fs / (8.0 * p.vr**2)
    az_phase_scale = -math.pi * wavelength * p.c / (4.0 * p.vr**2)

    grid = np.arange(N, dtype=np.float64)

    def rda(raw: sar.c64[N, N], range_ref: dtype[N], fa: fd[N], tau: fd[N],
            win_a: fd[N]) -> sar.f32[N, N]:
        data = sar.cast(raw, dtype)

        # Range compression (the window is folded into range_ref on the
        # host).
        spectrum = sar.fft(data, axis=1)
        spectrum = spectrum * sar.broadcast(range_ref, (N, N), dim=1)
        data = sar.ifft(spectrum, axis=1)

        # Into the range-Doppler domain.
        data = sar.fftshift(sar.fft(data, axis=0), axis=0)

        # Range-dependent factors: fa^2 (rows) x tau (columns).
        fa2_tau = (sar.broadcast(fa * fa, (N, N), dim=0) *
                   sar.broadcast(tau, (N, N), dim=1))

        # RCMC: positions = column index + migration shift(fa, R).
        migration = sar.cast(fa2_tau * rcmc_scale, sar.f64)
        positions = (sar.broadcast(sar.constant(grid),
                                   (N, N), dim=1) + migration)
        data = sar.interp1d(data, positions)

        # Azimuth compression + window.
        data = data * sar.expj(fa2_tau * az_phase_scale)
        data = data * sar.broadcast(win_a, (N, N), dim=0)

        # Back to the image domain.
        data = sar.ifft(sar.ifftshift(data, axis=0), axis=0)
        return sar.cast(sar.absolute(data), sar.f32)

    rda.__name__ = name
    return sar.func(rda)


def make_inputs(n: int, p: RadarParams):
    """Host-precomputed inputs: (range_ref, fa, tau, win_a)."""
    from rda.reference import make_range_reference
    fa = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / p.prf))
    tau = 2.0 * p.r0 / p.c - p.t_shift + np.arange(n) / p.fs
    win_a = band_windows(n, p)[1]
    return make_range_reference(n, p), fa, tau, win_a
