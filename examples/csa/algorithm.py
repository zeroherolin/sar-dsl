"""The Chirp Scaling Algorithm (CSA) expressed in SAR-DSL.

CSA is interpolation-free: every correction is an element-wise phase
multiply between FFTs, so the whole chain lowers to both backends without
any gather operation. The Doppler-dependent factors D(fa) and Km(fa) are
computed inside the kernel from the `fa` axis with element-wise ops.
"""

import math

import numpy as np

import sar

from common.params import RadarParams, band_windows

__all__ = ["build_kernel", "make_inputs"]


def build_kernel(n: int,
                 p: RadarParams,
                 name: str = "csa",
                 dtype=sar.c128) -> sar.Kernel:
    """Builds an `n x n` CSA imaging kernel (azimuth x range).

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
    sin_scale = wavelength / (2.0 * p.vr)
    coupling_scale = p.kr * p.c * p.r0 / (2.0 * p.vr**2 * p.fc**3)
    tau_ref_scale = 2.0 * p.r0 / p.c
    rcmc_scale = 4.0 * math.pi * p.r0 / p.c
    az_scale = 4.0 * math.pi * p.fc / p.c

    @sar.func
    def csa(raw: sar.c64[N, N], fa: fd[N], fr: fd[N], tau: fd[N], win_r: fd[N],
            win_a: fd[N]) -> sar.f32[N, N]:
        ones = sar.constant(1.0, dtype=sar.f64, shape=(N, N))
        fa_geometry = sar.cast(fa, sar.f64)
        fr_geometry = sar.cast(fr, sar.f64)
        tau_geometry = sar.cast(tau, sar.f64)

        # Doppler-dependent factors, broadcast along range.
        fa2 = sar.broadcast(fa_geometry, (N, N), dim=0)
        sin_theta = fa2 * sin_scale
        d = sar.sqrt(sar.maximum(ones - sin_theta * sin_theta, 1e-10))
        inv_d = ones / d
        coupling = (fa2 * fa2) * coupling_scale / (d * d * d)
        km = (ones * p.kr) / (ones - coupling)

        tau2 = sar.broadcast(tau_geometry, (N, N), dim=1)
        fr2 = sar.broadcast(fr_geometry, (N, N), dim=1)

        data = sar.cast(raw, dtype)

        # 1. Azimuth FFT.
        data = sar.fftshift(sar.fft(data, axis=0), axis=0)

        # 2. Chirp scaling.
        tau_diff = tau2 - inv_d * tau_ref_scale
        phi1 = km * (inv_d - ones) * (tau_diff * tau_diff) * math.pi
        data = data * sar.cast(sar.expj(phi1), dtype)

        # 3. Range FFT.
        data = sar.fftshift(sar.fft(data, axis=1), axis=1)

        # 4. Range compression + SRC + bulk RCMC (+ range window).
        phi2 = ((fr2 * fr2) * d / km * math.pi +
                (inv_d - ones) * fr2 * rcmc_scale)
        data = data * sar.cast(sar.expj(phi2), dtype) * win_r

        # 5. Range IFFT.
        data = sar.ifft(sar.ifftshift(data, axis=1), axis=1)

        # 6. Azimuth compression (+ azimuth window).
        phi3 = tau2 * (d - ones) * (az_scale * p.c / 2.0)
        data = data * sar.cast(sar.expj(phi3), dtype)
        data = data * sar.broadcast(win_a, (N, N), dim=0)

        # 7. Azimuth IFFT.
        data = sar.ifft(sar.ifftshift(data, axis=0), axis=0)
        return sar.cast(sar.absolute(data), sar.f32)

    csa.name = name
    return csa


def make_inputs(n: int, p: RadarParams):
    """Host-precomputed inputs: (fa, fr, tau, win_r, win_a)."""
    fa = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / p.prf))
    fr = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / p.fs))
    # Absolute fast time: R0 sits at offset t_shift into the window.
    tau = 2.0 * p.r0 / p.c - p.t_shift + np.arange(n) / p.fs
    win_r, win_a = band_windows(n, p)
    return fa, fr, tau, win_r, win_a
