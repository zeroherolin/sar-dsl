"""Point-target image-quality metrics.

    IRW   impulse response width (-3 dB), in samples, measured on a
          band-limited upsampled cut through the peak
    PSLR  peak sidelobe ratio (dB), highest sidelobe outside the mainlobe
    ISLR  integrated sidelobe ratio (dB), sidelobe over mainlobe energy

Pure measurement code: no algorithm imports, no CLI. Used by the
`run_cpu_quality` runner, by `plot_cpu_impulse_response`, and by
`test/python/test_quality.py`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "UPSAMPLE", "upsample", "measure_cut", "measure_image", "urban_contrast"
]

#: Upsampling factor for impulse-response cuts (sub-sample IRW resolution).
UPSAMPLE = 32


def upsample(cut: np.ndarray, factor: int = UPSAMPLE) -> np.ndarray:
    """Band-limited upsampling by zero-padding the spectrum."""
    cut = np.asarray(cut)
    if cut.ndim != 1 or cut.size == 0:
        raise ValueError("cut must be a non-empty 1-D array")
    if (isinstance(factor, (bool, np.bool_))
            or not isinstance(factor, (int, np.integer)) or factor < 1):
        raise ValueError("factor must be a positive integer")
    n = len(cut)
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(cut)))
    padded = np.zeros(n * factor, dtype=complex)
    padded[(n * factor - n) // 2:(n * factor + n) // 2] = spectrum
    return np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(padded)))) \
        * factor


def measure_cut(cut: np.ndarray, factor: int = UPSAMPLE) -> dict:
    """IRW / PSLR / ISLR of a 1-D impulse-response cut through the peak."""
    fine = upsample(np.asarray(cut, dtype=complex), factor)
    if not np.all(np.isfinite(fine)) or not np.any(fine > 0):
        raise ValueError("cut must contain finite nonzero data")
    peak_idx = int(np.argmax(fine))
    peak = fine[peak_idx]
    power = fine**2

    # -3 dB width around the peak. The crossing falls between samples, so
    # stepping out to the first sample below the half-power level would
    # bias the width high by up to one fine sample; interpolate linearly
    # between the bracketing pair instead.
    half = peak / np.sqrt(2.0)
    left = peak_idx
    while left > 0 and fine[left] > half:
        left -= 1
    right = peak_idx
    while right < len(fine) - 1 and fine[right] > half:
        right += 1

    def crossing(inner: int, outer: int) -> float:
        """Sub-sample position where the cut crosses `half`, between the
        last sample above it and the first below."""
        hi, lo = fine[inner], fine[outer]
        if hi == lo:
            return float(outer)
        return outer + (half - lo) / (hi - lo) * (inner - outer)

    lo_edge = crossing(min(left + 1, peak_idx), left) if fine[left] <= half \
        else float(left)
    hi_edge = crossing(max(right - 1, peak_idx), right) \
        if fine[right] <= half else float(right)
    irw = (hi_edge - lo_edge) / factor

    # Mainlobe extent: first minima on both sides.
    lo = peak_idx
    while lo > 0 and fine[lo - 1] < fine[lo]:
        lo -= 1
    hi = peak_idx
    while hi < len(fine) - 1 and fine[hi + 1] < fine[hi]:
        hi += 1
    sidelobes = np.concatenate([fine[:lo], fine[hi + 1:]])
    if sidelobes.size == 0:
        raise ValueError("cut has no samples outside the mainlobe")
    pslr = 20.0 * np.log10(sidelobes.max() / peak + 1e-30)

    mainlobe_energy = power[lo:hi + 1].sum()
    sidelobe_energy = power.sum() - mainlobe_energy
    islr = 10.0 * np.log10(sidelobe_energy / mainlobe_energy + 1e-30)

    return {"irw": irw, "pslr": pslr, "islr": islr}


def urban_contrast(image: np.ndarray,
                   tile: int = 512,
                   looks: int = 4) -> float:
    """Scene sharpness metric for real-data runs: std/mean over the
    brightest `tile x tile` window of the `looks x looks` multilooked
    magnitude image (higher is sharper; used for the ALOS numbers in
    the README and for autofocus calibration)."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D array")
    if looks < 1 or tile < looks or tile % looks != 0:
        raise ValueError("looks must be positive and divide tile")
    mag = np.abs(image)
    h, w = mag.shape
    if h < tile or w < tile:
        raise ValueError("image must be at least tile x tile")
    ml = mag[:h - h % looks, :w - w % looks]
    ml = ml.reshape(h // looks, looks, w // looks, looks).mean(axis=(1, 3))
    t = tile // looks
    best, pos = -1.0, (0, 0)
    for i in range(0, ml.shape[0] - t + 1, t):
        for j in range(0, ml.shape[1] - t + 1, t):
            s = ml[i:i + t, j:j + t].sum()
            if s > best:
                best, pos = s, (i, j)
    window = ml[pos[0]:pos[0] + t, pos[1]:pos[1] + t]
    mean = window.mean()
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("selected tile must have positive finite magnitude")
    return float(window.std() / mean)


def measure_image(image: np.ndarray, expected_peak=None) -> dict:
    """Range/azimuth metrics of a point-target image."""
    magnitude = np.abs(np.asarray(image))
    if magnitude.ndim != 2 or magnitude.size == 0:
        raise ValueError("image must be a non-empty 2-D array")
    i, j = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    metrics = {
        "peak": (i, j),
        "range": measure_cut(magnitude[i, :]),
        "azimuth": measure_cut(magnitude[:, j]),
    }
    if expected_peak is not None:
        metrics["peak_error"] = (i - expected_peak[0], j - expected_peak[1])
    return metrics
