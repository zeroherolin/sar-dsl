"""Synthetic stripmap SAR echo simulation for point targets.

Generates raw echoes compatible with the WKA processing chain: a linear-FM
chirp in fast time and the hyperbolic range migration in slow time. Useful
for demos and focusing-quality tests without the multi-hundred-MB ALOS
dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from wka_numpy import WKAParams

__all__ = ["PointTarget", "simulate_point_targets", "synthetic_params",
           "demo_scene"]


def synthetic_params(n: int) -> WKAParams:
    """Derives a self-consistent airborne C-band geometry for an `n x n`
    raster, so that point targets focus to roughly one resolution cell.

    - The chirp occupies half the sampled range window with ~70% of the
      sampling bandwidth (range resolution ~1.4 bins).
    - The platform velocity is chosen so the Doppler history of a full
      synthetic aperture spans ~70% of the PRF (azimuth resolution ~1.4
      bins, no Doppler aliasing).
    """
    c = 299792458.0
    fc = 5.3e9
    fs = 32.0e6
    prf = 400.0
    r0 = 20000.0
    wavelength = c / fc

    pulse_len = 0.5 * n / fs
    kr = -0.7 * fs / pulse_len
    vr = prf * np.sqrt(0.7 * wavelength * r0 / (2.0 * n))

    return WKAParams(c=c, fc=fc, fs=fs, prf=prf, vr=vr, r0=r0, kr=kr,
                     t_shift=0.0)


@dataclass(frozen=True)
class PointTarget:
    """A scatterer at `range_offset` m from the reference slant range and
    `azimuth_offset` m along track from scene center, with amplitude `rcs`."""

    range_offset: float
    azimuth_offset: float
    rcs: float = 1.0


def simulate_point_targets(n: int, p: WKAParams,
                           targets: Sequence[PointTarget]) -> np.ndarray:
    """Simulates an `n x n` raw echo raster (azimuth x range, complex64)."""
    # Fast time: window centered on the reference range.
    tau = 2.0 * p.r0 / p.c + (np.arange(n) - n / 2) / p.fs
    # Slow time: aperture centered on scene center.
    eta = (np.arange(n) - n / 2) / p.prf

    # Chirp duration: half the sampled window keeps the pulse inside it.
    pulse_len = 0.5 * n / p.fs

    raw = np.zeros((n, n), dtype=np.complex128)
    for target in targets:
        r0_t = p.r0 + target.range_offset
        eta_c = target.azimuth_offset / p.vr
        # Instantaneous slant range (hyperbolic model).
        r_eta = np.sqrt(r0_t ** 2 + (p.vr * (eta - eta_c)) ** 2)

        delay = 2.0 * r_eta / p.c
        t = tau[np.newaxis, :] - delay[:, np.newaxis]
        envelope = (np.abs(t) <= pulse_len / 2.0)

        phase = (-4.0 * np.pi * p.fc * r_eta[:, np.newaxis] / p.c
                 + np.pi * p.kr * t ** 2)
        raw += target.rcs * envelope * np.exp(1j * phase)

    return raw.astype(np.complex64)


def demo_scene(n: int, p: WKAParams) -> Tuple[np.ndarray, list]:
    """A small constellation of point targets plus the raw echoes."""
    swath = n / p.fs * p.c / 2.0          # slant-range extent (m)
    strip = n / p.prf * p.vr              # along-track extent (m)
    targets = [
        PointTarget(0.0, 0.0, 1.0),
        PointTarget(-0.15 * swath, -0.2 * strip, 0.8),
        PointTarget(0.2 * swath, 0.15 * strip, 1.2),
    ]
    return simulate_point_targets(n, p, targets), targets


def single_target_scene(n: int, p: WKAParams) -> np.ndarray:
    """Raw echoes of a lone scene-center scatterer (focus-quality tests)."""
    return simulate_point_targets(n, p, [PointTarget(0.0, 0.0, 1.0)])
