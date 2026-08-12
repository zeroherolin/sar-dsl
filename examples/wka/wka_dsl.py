"""The omega-K (WKA) imaging algorithm expressed in SAR-DSL.

`build_wka_kernel` traces the *entire* imaging chain -- range/azimuth FFTs,
bulk compression, Stolt interpolation, windowing and the inverse transforms
-- into a single `sar` dialect kernel that compiles to native code on the
CPU backend.

Host responsibilities are limited to acquisition metadata: frequency axes
(`fftfreq`) and window vectors are precomputed with numpy and passed as
kernel inputs, while scalar radar parameters are baked into the IR as
attributes at trace time.
"""

from __future__ import annotations

import math

import numpy as np

import sar

from wka_numpy import WKAParams

__all__ = ["build_wka_kernel", "make_kernel_inputs"]


def build_wka_kernel(n: int, p: WKAParams) -> sar.Kernel:
    """Builds an `n x n` WKA imaging kernel for the given parameters."""

    N = int(n)

    @sar.jit
    def wka(raw: sar.c64[N, N], fa: sar.f64[N], fr: sar.f64[N],
            win_r: sar.f64[N], win_a: sar.f64[N]) -> sar.f32[N, N]:
        # All spectral processing runs in double precision, mirroring
        # numpy.fft's promotion behaviour in the reference implementation.
        data = sar.cast(raw, sar.c128)

        # -- 2-D forward spectrum (range FFT, corner turn, azimuth FFT) ----
        data = sar.fftshift(sar.fft(data, dim=1), dim=1)
        data = sar.transpose(data)
        data = sar.fftshift(sar.fft(data, dim=1), dim=1)
        data = sar.transpose(data)

        # -- bulk compression ----------------------------------------------
        # phase = (4 pi R0 / c) * (sqrt((fc + fr)^2 - (c fa / 2 vr)^2)
        #          - (fc + fr)) + pi fr^2 / Kr
        fa2 = sar.broadcast(fa, (N, N), dim=0)   # varies along azimuth rows
        fr2 = sar.broadcast(fr, (N, N), dim=1)   # varies along range cols
        fr_shifted = fr2 + p.fc
        term1 = fr_shifted * fr_shifted
        fa_scaled = fa2 * (p.c / (2.0 * p.vr))
        term2 = fa_scaled * fa_scaled
        root = sar.sqrt(sar.maximum(term1 - term2, 1e-10))
        phase = ((root - fr_shifted) * (4.0 * math.pi * p.r0 / p.c)
                 + (fr2 * fr2) * (math.pi / p.kr))
        data = data * sar.expj(phase)

        # -- Stolt interpolation ---------------------------------------------
        data = sar.stolt_interp(data, fa, fr, c=p.c, fc=p.fc, vr=p.vr,
                                t_shift=p.t_shift)

        # -- windowing (range, then azimuth via corner turn) -----------------
        data = data * sar.cast(sar.broadcast(win_r, (N, N), dim=1), sar.c128)
        data = sar.transpose(data)
        data = data * sar.cast(sar.broadcast(win_a, (N, N), dim=1), sar.c128)
        data = sar.transpose(data)

        # -- back to the image domain ----------------------------------------
        data = sar.ifft(sar.ifftshift(data, dim=1), dim=1)
        data = sar.transpose(data)
        data = sar.ifft(sar.ifftshift(data, dim=1), dim=1)
        data = sar.transpose(data)

        return sar.cast(sar.absolute(data), sar.f32)

    return wka


def make_kernel_inputs(n: int, p: WKAParams):
    """Returns the host-precomputed inputs (fa, fr, win_r, win_a)."""
    fa = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / p.prf))
    fr = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / p.fs))
    win = np.hanning(n)
    return fa, fr, win.copy(), win.copy()


if __name__ == "__main__":
    from wka_numpy import ALOS_PARAMS

    n = 256
    kernel = build_wka_kernel(n, ALOS_PARAMS)
    print(kernel.to_mlir())
