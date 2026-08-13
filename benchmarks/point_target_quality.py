#!/usr/bin/env python3
"""Point-target quality analysis for the imaging algorithms.

Measures the standard SAR image-quality metrics on a single scene-center
scatterer for every imaging algorithm (cpu-compiled kernels):

    IRW   impulse response width (-3 dB), in samples, via 32x FFT
          upsampling of the peak row/column
    PSLR  peak sidelobe ratio (dB)
    ISLR  integrated sidelobe ratio (dB)
    peak  position error relative to the scene center

Usage:
    python benchmarks/point_target_quality.py [--n 512]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "python"), str(REPO / "examples")]

from common.params import synthetic_params      # noqa: E402
from common.simulate import single_target_scene  # noqa: E402

__all__ = ["measure_cut", "measure_image"]


def _upsample(cut: np.ndarray, factor: int) -> np.ndarray:
    """Band-limited upsampling by zero-padding the spectrum."""
    n = len(cut)
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(cut)))
    padded = np.zeros(n * factor, dtype=complex)
    padded[(n * factor - n) // 2:(n * factor + n) // 2] = spectrum
    return np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(padded)))) \
        * factor


def measure_cut(cut: np.ndarray, factor: int = 32) -> dict:
    """IRW / PSLR / ISLR of a 1-D impulse-response cut through the peak."""
    fine = _upsample(cut.astype(complex), factor)
    peak_idx = int(np.argmax(fine))
    peak = fine[peak_idx]
    power = fine ** 2

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


def _report(name: str, metrics: dict) -> None:
    rng, azm = metrics["range"], metrics["azimuth"]
    print(f"{name:>6}: peak_err={metrics['peak_error']}  "
          f"range[IRW={rng['irw']:5.2f} PSLR={rng['pslr']:7.2f} dB "
          f"ISLR={rng['islr']:7.2f} dB]  "
          f"azimuth[IRW={azm['irw']:5.2f} PSLR={azm['pslr']:7.2f} dB "
          f"ISLR={azm['islr']:7.2f} dB]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    args = parser.parse_args()

    n = args.n
    params = synthetic_params(n)
    raw = single_target_scene(n, params)
    center = (n // 2, n // 2)

    from csa.algorithm import build_kernel as build_csa
    from csa.algorithm import make_inputs as csa_inputs
    from rda.algorithm import build_kernel as build_rda
    from rda.algorithm import make_inputs as rda_inputs
    from wka.algorithm import build_kernel as build_wka
    from wka.algorithm import make_inputs as wka_inputs

    for name, build, inputs in (
            ("WKA", build_wka, wka_inputs),
            ("RDA", build_rda, rda_inputs),
            ("CSA", build_csa, csa_inputs)):
        kernel = build(n, params).compile("cpu")
        image = kernel(raw, *inputs(n, params)).astype(np.float64)
        _report(name, measure_image(image, expected_peak=center))


if __name__ == "__main__":
    main()
