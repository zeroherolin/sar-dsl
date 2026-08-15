"""Point-target image-quality metrics.

    IRW   impulse response width (-3 dB), in samples, measured on a
          band-limited upsampled cut through the peak
    PSLR  peak sidelobe ratio (dB), highest sidelobe outside the mainlobe
    ISLR  integrated sidelobe ratio (dB), sidelobe over mainlobe energy

Pure measurement code: no algorithm imports, no CLI. Used by the
`run_quality` / `run_figures` runners and by `test/python/test_quality.py`.
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
    n = len(cut)
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(cut)))
    padded = np.zeros(n * factor, dtype=complex)
    padded[(n * factor - n) // 2:(n * factor + n) // 2] = spectrum
    return np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(padded)))) \
        * factor


def measure_cut(cut: np.ndarray, factor: int = UPSAMPLE) -> dict:
    """IRW / PSLR / ISLR of a 1-D impulse-response cut through the peak."""
    fine = upsample(cut.astype(complex), factor)
    peak_idx = int(np.argmax(fine))
    peak = fine[peak_idx]
    power = fine**2

    # -3 dB width around the peak.
    half = peak / np.sqrt(2.0)
    left = peak_idx
    while left > 0 and fine[left] > half:
        left -= 1
    right = peak_idx
    while right < len(fine) - 1 and fine[right] > half:
        right += 1
    irw = (right - left) / factor

    # Mainlobe extent: first minima on both sides.
    lo = peak_idx
    while lo > 0 and fine[lo - 1] < fine[lo]:
        lo -= 1
    hi = peak_idx
    while hi < len(fine) - 1 and fine[hi + 1] < fine[hi]:
        hi += 1
    sidelobes = np.concatenate([fine[:lo], fine[hi + 1:]])
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
    mag = np.abs(image)
    h, w = mag.shape
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
    return float(window.std() / window.mean())


def measure_image(image: np.ndarray, expected_peak=None) -> dict:
    """Range/azimuth metrics of a point-target image."""
    i, j = np.unravel_index(np.argmax(image), image.shape)
    metrics = {
        "peak": (i, j),
        "range": measure_cut(image[i, :]),
        "azimuth": measure_cut(image[:, j]),
    }
    if expected_peak is not None:
        metrics["peak_error"] = (i - expected_peak[0], j - expected_peak[1])
    return metrics
