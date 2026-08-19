#!/usr/bin/env python3
"""Independent NumPy reference for the FP32 WKA implementation.

The default reduced-size run is intended for regression testing.  Full-size
input can be checked in row ranges with --row-limit before a costly full run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

C0 = np.float64(299792458.0)
FC = np.float64(1269999750.06)
FS = np.float64(32000000.0)
PRF = np.float64(2155.172)
VR = np.float64(7072.0)
R0 = np.float64(843013.994)
KR = np.float64(-1.037e12)
TAP_START = -3
TAP_END = 4
LUT_SIZE = 1024
PULSE_LEN = np.float64(27.0e-6)
MAX_NRMSE = 1.0e-3
MIN_CORRELATION = 0.999


def synthetic_input(n: int) -> np.ndarray:
    rows = np.arange(n, dtype=np.float64)[:, None]
    cols = np.arange(n, dtype=np.float64)[None, :]
    phase = 0.03125 * np.mod(rows, 17) + 0.0625 * np.mod(cols, 29)
    amplitude = 0.25 + 0.05 * np.mod(rows + cols, 11)
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def stolt_weights() -> np.ndarray:
    frac = np.arange(LUT_SIZE, dtype=np.float64) / LUT_SIZE
    taps = np.arange(TAP_START, TAP_END + 1, dtype=np.float64)
    dist = frac[:, None] - taps[None, :]
    weight = np.sinc(dist)
    inside = np.abs(dist) < 4.0
    window = 0.5 + 0.5 * np.cos(np.pi * dist / 4.0)
    return np.where(inside, weight * window, 0.0).astype(np.float32)


def band_window(n: int, fraction: float) -> np.ndarray:
    width = max(4, min(n, int(round(n * fraction))))
    result = np.zeros(n)
    start = (n - width) // 2
    result[start:start + width] = np.hanning(width)
    return result.astype(np.float32)


def band_windows(n: int) -> tuple[np.ndarray, np.ndarray]:
    range_fraction = min(1.0, abs(KR) * PULSE_LEN / FS)
    doppler_span = 2.0 * VR**2 * (n / PRF) / ((C0 / FC) * R0)
    azimuth_fraction = min(1.0, doppler_span / PRF)
    return (band_window(n, range_fraction), band_window(n, azimuth_fraction))


def time_shift(n: int) -> np.float64:
    recorded_sample = 4800.0
    if recorded_sample + 0.5 * PULSE_LEN * FS < n:
        return np.float64(recorded_sample / FS)
    return np.float64(n / (2.0 * FS))


def bulk_and_stolt(data: np.ndarray,
                   row_limit: int | None = None) -> np.ndarray:
    n = data.shape[0]
    df_a = PRF / n
    df_r = FS / n
    fa = -PRF / 2.0 + np.arange(n, dtype=np.float64) * df_a
    fr = -FS / 2.0 + np.arange(n, dtype=np.float64) * df_r
    x = FC + fr
    shift = time_shift(n)
    range_phase = 2.0 * np.pi * fr * shift
    weights = stolt_weights()

    output = np.zeros_like(data, dtype=np.complex64)
    rows_to_run = n if row_limit is None else min(n, row_limit)
    coeff = 4.0 * np.pi * R0 / C0
    pi_over_kr = np.pi / KR
    c_over_2v = C0 / (2.0 * VR)

    for i in range(rows_to_run):
        term2 = (c_over_2v * fa[i])**2
        bulk_root = np.sqrt(np.maximum(x * x - term2, 0.0))
        bulk_diff = -term2 / (bulk_root + x)
        bulk_phase = coeff * bulk_diff + fr * fr * pi_over_kr
        source = data[i].astype(np.complex128)
        source *= np.exp(1j * (bulk_phase + range_phase))

        # Stable sqrt(x*x+term2)-FC form around the destination bin.
        stolt_root = np.sqrt(x * x + term2)
        delta = term2 / (stolt_root + x)
        idx_float = np.arange(n, dtype=np.float64) + delta / df_r
        idx_int = np.floor(idx_float).astype(np.int64)
        frac = idx_float - idx_int
        lut_idx = np.clip((frac * LUT_SIZE).astype(np.int64), 0, LUT_SIZE - 1)

        accum = np.zeros(n, dtype=np.complex128)
        for tap_slot, tap in enumerate(range(TAP_START, TAP_END + 1)):
            src_idx = idx_int + tap
            valid = (src_idx >= 0) & (src_idx < n)
            accum[valid] += (source[src_idx[valid]] *
                             weights[lut_idx[valid], tap_slot])
        output[i] = (accum * np.exp(-1j * 2.0 * np.pi * fr * shift)).astype(
            np.complex64)
    return output


def wka_reference_complex(data: np.ndarray,
                          row_limit: int | None = None) -> np.ndarray:
    n = data.shape[0]
    spectrum = np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)
    spectrum = np.fft.fftshift(np.fft.fft(spectrum, axis=0), axes=0)
    migrated = bulk_and_stolt(spectrum.astype(np.complex64), row_limit)
    range_window, azimuth_window = band_windows(n)
    migrated *= azimuth_window[:, None] * range_window[None, :]
    image = np.fft.ifft(np.fft.ifftshift(migrated, axes=1), axis=1)
    image = np.fft.ifft(np.fft.ifftshift(image, axes=0), axis=0)
    return image.astype(np.complex64)


def wka_reference(data: np.ndarray,
                  row_limit: int | None = None) -> np.ndarray:
    return np.abs(wka_reference_complex(data, row_limit)).astype(np.float32)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.complex128).ravel()
    got = candidate.astype(np.complex128).ravel()
    error = got - ref
    ref_energy = np.vdot(ref, ref).real
    err_energy = np.vdot(error, error).real
    nrmse = np.sqrt(err_energy / max(ref_energy, np.finfo(float).tiny))
    correlation = abs(np.vdot(ref, got)) / np.sqrt(
        max(ref_energy * np.vdot(got, got).real,
            np.finfo(float).tiny))
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "nrmse": float(nrmse),
        "snr_db": float(-20.0 * np.log10(max(nrmse,
                                             np.finfo(float).tiny))),
        "complex_correlation": float(correlation),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--row-limit", type=int)
    args = parser.parse_args()

    if args.input:
        raw = np.fromfile(args.input, dtype=np.complex64)
        expected = args.n * args.n
        if raw.size != expected:
            raise ValueError(
                f"expected {expected} complex samples, got {raw.size}")
        data = raw.reshape(args.n, args.n)
    else:
        data = synthetic_input(args.n)

    result = wka_reference(data, args.row_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.tofile(args.output)

    summary = {
        "n": args.n,
        "finite": bool(np.all(np.isfinite(result))),
        "nonzero": int(np.count_nonzero(np.abs(result) > 1.0e-6)),
    }
    passed = summary["finite"] and bool(summary["nonzero"])
    if args.compare:
        candidate = np.fromfile(args.compare, dtype=np.float32)
        if candidate.size != result.size:
            raise ValueError(f"expected {result.size} candidate samples, got "
                             f"{candidate.size}")
        candidate = candidate.reshape(result.shape)
        candidate_finite = bool(np.all(np.isfinite(candidate)))
        summary.update(metrics(result, candidate))
        summary["candidate_finite"] = candidate_finite
        summary["comparison_passed"] = bool(
            candidate_finite and summary["nrmse"] <= MAX_NRMSE
            and summary["complex_correlation"] >= MIN_CORRELATION)
        passed &= summary["comparison_passed"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
