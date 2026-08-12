"""The Range-Doppler Algorithm (RDA) expressed in SAR-DSL.

Demonstrates that the dialect generalizes beyond omega-K: RCMC is built
from the orthogonal `sar.interp1d` primitive plus element-wise position
computation -- no RDA-specific operation exists in the compiler.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

import sar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wka"))

from wka_numpy import WKAParams  # noqa: E402

__all__ = ["build_rda_kernel", "make_kernel_inputs"]


def build_rda_kernel(n: int, p: WKAParams) -> sar.Kernel:
    """Builds an `n x n` RDA imaging kernel (azimuth x range)."""

    N = int(n)
    wavelength = p.c / p.fc
    # RCMC shift in range bins per unit fa^2.
    rcmc_scale = (wavelength ** 2 * p.r0 / (8.0 * p.vr ** 2)) * 2.0 * p.fs / p.c
    # Azimuth matched filter: exp(+j pi fa^2 / Ka), Ka = -2 Vr^2/(lambda R0)
    # (conjugate of the stationary-phase chirp spectrum).
    ka = -2.0 * p.vr ** 2 / (wavelength * p.r0)
    az_phase_scale = math.pi / ka

    grid = np.arange(N, dtype=np.float64)

    @sar.jit
    def rda(raw: sar.c64[N, N], range_ref: sar.c128[N],
            fa: sar.f64[N]) -> sar.f32[N, N]:
        data = sar.cast(raw, sar.c128)

        # -- range compression (frequency-domain matched filter) ----------
        spectrum = sar.fft(data, dim=1)
        spectrum = spectrum * sar.broadcast(range_ref, (N, N), dim=1)
        data = sar.ifft(spectrum, dim=1)

        # -- into the range-Doppler domain ---------------------------------
        data = sar.fftshift(sar.fft(data, dim=0), dim=0)

        # -- RCMC: positions = column index + migration shift(fa) ----------
        delta_bins = (fa * fa) * rcmc_scale
        positions = (sar.broadcast(sar.constant(grid), (N, N), dim=1)
                     + sar.broadcast(delta_bins, (N, N), dim=0))
        data = sar.interp1d(data, positions)

        # -- azimuth compression -------------------------------------------
        az_filter = sar.expj((fa * fa) * az_phase_scale)
        data = data * sar.broadcast(az_filter, (N, N), dim=0)

        # -- back to the image domain ---------------------------------------
        data = sar.ifft(sar.ifftshift(data, dim=0), dim=0)
        return sar.cast(sar.absolute(data), sar.f32)

    return rda


def make_kernel_inputs(n: int, p: WKAParams):
    """Host-precomputed inputs: (range_ref, fa)."""
    from rda_numpy import make_range_reference
    fa = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / p.prf))
    return make_range_reference(n, p), fa


if __name__ == "__main__":
    from wka_numpy import ALOS_PARAMS

    kernel = build_rda_kernel(256, ALOS_PARAMS)
    print(kernel.to_mlir())
