"""The Chirp Scaling Algorithm (CSA) expressed in SAR-DSL.

CSA is interpolation-free: every correction is an element-wise phase
multiply between FFTs, so the whole chain lowers to both backends without
any gather operation. The Doppler-dependent factors D(fa) and Km(fa) are
computed inside the kernel from the `fa` axis with element-wise ops.
"""

from __future__ import annotations

import math

import numpy as np

import sar

from common.params import RadarParams

__all__ = ["build_kernel", "make_inputs"]


def build_kernel(n: int, p: RadarParams) -> sar.Kernel:
    """Builds an `n x n` CSA imaging kernel (azimuth x range)."""

    N = int(n)
    wavelength = p.c / p.fc
    sin_scale = wavelength / (2.0 * p.vr)
    coupling_scale = p.kr * p.c * p.r0 / (2.0 * p.vr ** 2 * p.fc ** 3)
    tau_ref_scale = 2.0 * p.r0 / p.c
    rcmc_scale = 4.0 * math.pi * p.r0 / p.c
    az_scale = 4.0 * math.pi * p.fc / p.c

    @sar.jit
    def csa(raw: sar.c64[N, N], fa: sar.f64[N], fr: sar.f64[N],
            tau: sar.f64[N], win_r: sar.f64[N],
            win_a: sar.f64[N]) -> sar.f32[N, N]:
        ones = sar.constant(1.0, dtype=sar.f64, shape=(N, N))

        # Doppler-dependent factors, broadcast along range.
        fa2 = sar.broadcast(fa, (N, N), dim=0)
        sin_theta = fa2 * sin_scale
        d = sar.sqrt(sar.maximum(ones - sin_theta * sin_theta, 1e-10))
        inv_d = ones / d
        coupling = (fa2 * fa2) * coupling_scale / (d * d * d)
        km = (ones * p.kr) / (ones - coupling)

        tau2 = sar.broadcast(tau, (N, N), dim=1)
        fr2 = sar.broadcast(fr, (N, N), dim=1)

        data = sar.cast(raw, sar.c128)

        # 1. Azimuth FFT.
        data = sar.fftshift(sar.fft(data, dim=0), dim=0)

        # 2. Chirp scaling.
        tau_diff = tau2 - inv_d * tau_ref_scale
        phi1 = km * (inv_d - ones) * (tau_diff * tau_diff) * math.pi
        data = data * sar.expj(phi1)

        # 3. Range FFT.
        data = sar.fftshift(sar.fft(data, dim=1), dim=1)

        # 4. Range compression + SRC + bulk RCMC (+ range window).
        phi2 = ((fr2 * fr2) * d / km * math.pi
                + (inv_d - ones) * fr2 * rcmc_scale)
        data = data * sar.expj(phi2)
        data = data * sar.cast(sar.broadcast(win_r, (N, N), dim=1), sar.c128)

        # 5. Range IFFT.
        data = sar.ifft(sar.ifftshift(data, dim=1), dim=1)

        # 6. Azimuth compression (+ azimuth window).
        phi3 = tau2 * (d - ones) * (az_scale * p.c / 2.0)
        data = data * sar.expj(phi3)
        data = data * sar.cast(sar.broadcast(win_a, (N, N), dim=0), sar.c128)

        # 7. Azimuth IFFT.
        data = sar.ifft(sar.ifftshift(data, dim=0), dim=0)
        return sar.cast(sar.absolute(data), sar.f32)

    return csa


def make_inputs(n: int, p: RadarParams):
    """Host-precomputed inputs: (fa, fr, tau, win_r, win_a)."""
    fa = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / p.prf))
    fr = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / p.fs))
    # Absolute fast time: R0 sits at offset t_shift into the window.
    tau = 2.0 * p.r0 / p.c - p.t_shift + np.arange(n) / p.fs
    win = np.hanning(n)
    return fa, fr, tau, win.copy(), win.copy()
