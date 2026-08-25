"""Synthetic stripmap SAR echo simulation for point targets.

Generates raw echoes compatible with all imaging chains: a linear-FM chirp
in fast time and the hyperbolic range migration in slow time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .params import RadarParams

__all__ = [
    "PointTarget", "simulate_point_targets", "demo_scene",
    "single_target_scene", "target_pixel"
]


@dataclass(frozen=True)
class PointTarget:
    """A scatterer at `range_offset` m from the reference slant range and
    `azimuth_offset` m along track from scene center, with amplitude `rcs`."""

    range_offset: float
    azimuth_offset: float
    rcs: float = 1.0


def simulate_point_targets(n: int, p: RadarParams,
                           targets: Sequence[PointTarget]) -> np.ndarray:
    """Simulates an `n x n` raw echo raster (azimuth x range, complex64)."""
    # Fast time: window positioned so R0 sits at offset t_shift.
    tau = 2.0 * p.r0 / p.c - p.t_shift + np.arange(n) / p.fs
    # Slow time: aperture centered on scene center.
    eta = (np.arange(n) - n / 2) / p.prf
    pulse_len = p.pulse_len

    raw = np.zeros((n, n), dtype=np.complex128)
    for target in targets:
        r0_t = p.r0 + target.range_offset
        eta_c = target.azimuth_offset / p.vr
        # Instantaneous slant range (hyperbolic model).
        r_eta = np.sqrt(r0_t**2 + (p.vr * (eta - eta_c))**2)

        delay = 2.0 * r_eta / p.c
        t = tau[np.newaxis, :] - delay[:, np.newaxis]
        envelope = (np.abs(t) <= pulse_len / 2.0)

        phase = (-4.0 * np.pi * p.fc * r_eta[:, np.newaxis] / p.c +
                 np.pi * p.kr * t**2)
        raw += target.rcs * envelope * np.exp(1j * phase)

    return raw.astype(np.complex64)


def demo_scene(n: int, p: RadarParams) -> Tuple[np.ndarray, List[PointTarget]]:
    """A fixed constellation of point targets plus the raw echoes."""
    swath = n / p.fs * p.c / 2.0  # slant-range extent (m)
    strip = n / p.prf * p.vr  # along-track extent (m)
    targets = [
        PointTarget(0.0, 0.0, 1.0),
        PointTarget(-0.15 * swath, -0.2 * strip, 0.8),
        PointTarget(0.2 * swath, 0.15 * strip, 1.2),
    ]
    return simulate_point_targets(n, p, targets), targets


def single_target_scene(n: int, p: RadarParams) -> np.ndarray:
    """Raw echoes of a lone scene-center scatterer (focus-quality tests)."""
    return simulate_point_targets(n, p, [PointTarget(0.0, 0.0, 1.0)])


def target_pixel(target: PointTarget, n: int, p: RadarParams):
    """(row, col) the focused image peaks at for `target`.

    Range: R0 lands `t_shift` into the sampled window, and a slant-range
    offset moves the echo by one sample per `c / 2 Fs` metres. Azimuth:
    the aperture is centered on scene center, so an along-track offset
    moves the target by one pulse per `Vr / PRF` metres.
    """
    col = p.t_shift * p.fs + target.range_offset * 2.0 * p.fs / p.c
    row = n / 2.0 + target.azimuth_offset * p.prf / p.vr
    return int(round(row)), int(round(col))
