"""Spotlight collection geometry shared by the DSL and reference paths.

Wavenumbers are scaled so that one image pixel equals one scene unit:
the rectangular target grid has spacing `2 pi / n` along both axes, and
the polar collection grid encloses it with a narrow margin.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Geometry"]

#: Aperture half-angle tangent; sets the polar wedge and thus the
#: cross-range coverage of the inscribed rectangle.
_TAN_HALF_APERTURE = 0.115


class Geometry:
    """Polar collection grid and the inscribed rectangular target grid."""

    def __init__(self, n: int):
        self.n = n
        du = dv = 2.0 * np.pi / n

        # Rectangle: v (range wavenumber) starts high enough that the
        # +/- u extent stays inside the polar wedge.
        self.u = (np.arange(n) - n / 2) * du
        v_min = (n / 2) * du / _TAN_HALF_APERTURE
        self.v = v_min + np.arange(n) * dv

        # Polar grid enclosing the rectangle (2% angular margin).
        self.theta_max = np.arctan(_TAN_HALF_APERTURE) * 1.02
        self.theta = np.linspace(-self.theta_max, self.theta_max, n)
        k_lo = self.v[0] * 0.999
        k_hi = self.v[-1] / np.cos(self.theta_max) * 1.001
        self.k = np.linspace(k_lo, k_hi, n)

    def simulate(self, targets) -> np.ndarray:
        """Phase history of unit scatterers at (x, y) scene positions
        (pixel units; the image peak lands at `center + (y, x)`)."""
        k = self.k[np.newaxis, :]
        theta = self.theta[:, np.newaxis]
        data = np.zeros((self.n, self.n), dtype=np.complex128)
        for x, y in targets:
            phase = -(k * np.cos(theta) * x + k * np.sin(theta) * y)
            data += np.exp(1j * phase)
        return data

    def demo_targets(self):
        n = self.n
        return [(0.0, 0.0), (-n * 0.2, n * 0.15), (n * 0.25, -n * 0.1)]
