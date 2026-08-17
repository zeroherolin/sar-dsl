"""The omega-K (WKA) imaging algorithm expressed in SAR-DSL.

`build_kernel` traces the whole imaging chain -- range/azimuth FFTs,
bulk compression, Stolt interpolation, windowing and the inverse
transforms -- into a single kernel.

Acquisition metadata (scalar radar parameters and the frequency axes)
bakes into the IR at trace time; only the data and the window vectors
remain kernel inputs.
"""

import math

import numpy as np

import sar

from common.params import RadarParams, band_windows

__all__ = ["build_kernel", "make_inputs"]


def build_kernel(n: int,
                 p: RadarParams,
                 name: str = "wka",
                 dtype=sar.c128) -> sar.Kernel:
    """Builds an `n x n` WKA imaging kernel for the given parameters.

    `name` becomes the IR symbol and the HLS top function, so a second
    kernel can be emitted beside the first without a symbol clash.
    `dtype` is the spectral working precision: `sar.c128` (default)
    mirrors numpy.fft's promotion in the reference implementation,
    `sar.c64` keeps the whole data path single-precision -- see
    `benchmarks/run_precision.py` for what that trades.
    """
    if dtype not in (sar.c128, sar.c64):
        raise ValueError("dtype must be sar.c128 or sar.c64")
    fd = sar.f64 if dtype is sar.c128 else sar.f32

    N = int(n)
    fa = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / p.prf))
    fr = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / p.fs))

    def wka(raw: sar.c64[N, N], win_r: fd[N], win_a: fd[N]) -> sar.f32[N, N]:
        data = sar.cast(raw, dtype)

        # 2-D forward spectrum: range FFT, corner turn, azimuth FFT.
        data = sar.fftshift(sar.fft(data, axis=1), axis=1)
        data = sar.transpose(data)
        data = sar.fftshift(sar.fft(data, axis=1), axis=1)
        data = sar.transpose(data)

        # Bulk compression:
        #   phase = (4 pi R0 / c) * (sqrt((fc + fr)^2 - (c fa / 2 vr)^2)
        #           - (fc + fr)) + pi fr^2 / Kr
        fa2 = sar.broadcast(fa, (N, N), dim=0)  # varies along azimuth rows
        fr2 = sar.broadcast(fr, (N, N), dim=1)  # varies along range cols
        fr_shifted = fr2 + p.fc
        term1 = fr_shifted * fr_shifted
        fa_scaled = fa2 * (p.c / (2.0 * p.vr))
        term2 = fa_scaled * fa_scaled
        root = sar.sqrt(sar.maximum(term1 - term2, 1e-10))
        phase = ((root - fr_shifted) * (4.0 * math.pi * p.r0 / p.c) +
                 (fr2 * fr2) * (math.pi / p.kr))
        data = data * sar.expj(phase)

        data = sar.stolt_interp(data,
                                fa,
                                fr,
                                c=p.c,
                                fc=p.fc,
                                vr=p.vr,
                                t_shift=p.t_shift)

        # Windowing: range, then azimuth via a corner turn (rank-1
        # operands broadcast along the last axis, floats promote).
        data = data * win_r
        data = sar.transpose(data) * win_a
        data = sar.transpose(data)

        # Back to the image domain.
        data = sar.ifft(sar.ifftshift(data, axis=1), axis=1)
        data = sar.transpose(data)
        data = sar.ifft(sar.ifftshift(data, axis=1), axis=1)
        data = sar.transpose(data)

        return sar.cast(sar.absolute(data), sar.f32)

    wka.__name__ = name
    return sar.func(wka)


def make_inputs(n: int, p: RadarParams):
    """Host-precomputed inputs: (win_r, win_a), band-matched Hann tapers."""
    return band_windows(n, p)
