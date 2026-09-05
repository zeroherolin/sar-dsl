"""Radar acquisition parameters shared by all imaging algorithms."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "RadarParams", "ALOS_PARAMS", "alos_params", "synthetic_params",
    "band_windows"
]


@dataclass(frozen=True)
class RadarParams:
    """Radar/platform parameters of a stripmap acquisition."""

    c: float  # propagation speed (m/s)
    fc: float  # carrier frequency (Hz)
    fs: float  # range sampling rate (Hz)
    prf: float  # pulse repetition frequency (Hz)
    vr: float  # effective radar velocity (m/s)
    r0: float  # reference slant range (m)
    kr: float  # range chirp rate (Hz/s)
    pulse_len: float  # transmitted chirp duration (s)
    t_shift: float  # fast-time offset of R0 from the window start (s)


#: ALOS-1 PALSAR parameters used by the San Francisco dataset.
#:
#: `vr` is autofocus-calibrated (image-contrast maximization over the urban
#: area, 1 m/s grid): the platform's nominal ~7155 m/s does not equal the
#: effective radar velocity that focuses the hyperbolic phase history.
ALOS_PARAMS = RadarParams(
    c=299792458.0,
    fc=1269999750.06,
    fs=32000000.00,
    prf=2155.172,
    vr=7072.0,
    r0=843013.994,
    kr=-1.037e12,
    pulse_len=27.0e-6,
    t_shift=4800.0 / 32000000.00,
)


def alos_params(n: int) -> RadarParams:
    """ALOS-1 parameters for an `n`-sample range window.

    Only `t_shift` depends on `n`: it says where the reference range sits
    in the sampled window, which is a property of the recording, not of
    the radar. The product puts R0 at sample 4800, so any window long
    enough to hold it (and the trailing chirp) keeps the recorded value
    and is a true sub-window of the acquisition. A shorter window would
    place R0 past its own end -- the echo would miss the raster entirely
    -- so it centres R0 instead. Everything the radar itself fixes (`fc`,
    `fs`, PRF, `Vr`, `R0`, `Kr`, `Tp`) keeps the acquisition's values at
    every size, which is what makes every selected raster exercise the
    real geometry.
    """
    r0_sample = ALOS_PARAMS.t_shift * ALOS_PARAMS.fs
    if r0_sample + 0.5 * ALOS_PARAMS.pulse_len * ALOS_PARAMS.fs < n:
        return ALOS_PARAMS
    return replace(ALOS_PARAMS, t_shift=n / (2.0 * ALOS_PARAMS.fs))


def _hann_band(n: int, frac: float) -> np.ndarray:
    """Hann taper over the central `frac` of an fftshift-centered
    frequency axis, zero outside."""
    m = max(4, min(n, int(round(n * frac))))
    win = np.zeros(n)
    start = (n - m) // 2
    win[start:start + m] = np.hanning(m)
    return win


def band_windows(n: int, p: RadarParams):
    """(win_r, win_a): Hann tapers matched to the occupied signal bands.

    A taper wider than the signal support degrades sidelobe control --
    the band edges then see a *truncated* window -- so each window covers
    exactly the occupied bandwidth: the chirp bandwidth `|kr| T_p` in
    range, and the full-aperture Doppler span `2 vr^2 T / (lambda r0)`
    in azimuth, clipped to the PRF for long scenes where the dwell is
    antenna-limited. Out-of-band bins are zeroed (they carry only
    spectral leakage and noise).
    """
    frac_r = min(1.0, abs(p.kr) * p.pulse_len / p.fs)
    doppler_span = 2.0 * p.vr**2 * (n / p.prf) / ((p.c / p.fc) * p.r0)
    frac_a = min(1.0, doppler_span / p.prf)
    return _hann_band(n, frac_r), _hann_band(n, frac_a)


def synthetic_params(n: int) -> RadarParams:
    """Derives a self-consistent airborne C-band geometry for an `n x n`
    raster, so that point targets focus to roughly one resolution cell.

    - The chirp occupies half the sampled range window with ~70% of the
      sampling bandwidth (range resolution ~1.4 bins).
    - The platform velocity is chosen so the Doppler history of a full
      synthetic aperture spans ~70% of the PRF (azimuth resolution ~1.4
      bins, no Doppler aliasing).
    - `t_shift` is the fast-time offset of the reference range R0 from the
      window start (here: the window center). The Stolt interpolation
      requires spectra referenced to the scene center; feeding a
      window-relative spectrum into the nonlinear Stolt mapping would turn
      the reference ramp into an azimuth-dependent defocus.
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
    t_shift = n / (2.0 * fs)

    return RadarParams(c=c,
                       fc=fc,
                       fs=fs,
                       prf=prf,
                       vr=vr,
                       r0=r0,
                       kr=kr,
                       pulse_len=pulse_len,
                       t_shift=t_shift)
