"""Signal-processing vocabulary composed from the core primitives.

These functions carry the working vocabulary of Matlab / scipy.signal
users (windows, dB conversions, 2-D transforms, dechirp, matched
filtering, ...) but decompose into the small orthogonal IR op set at
trace time: the DSL surface is deliberately richer than the IR. All
indexing is 0-based.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import (Tensor, TraceError, _require_tensor, _resolve_axis, _tracing,
               absolute, broadcast, cast, concatenate, conj, constant, exp,
               expj, f32, fft, ifft, interp1d, log, maximum, op, sin, sqrt,
               where)
from . import sum as _sum

__all__ = [
    "fft2",
    "ifft2",
    "circshift",
    "mean",
    "std",
    "var",
    "multilook",
    "stolt_interp",
    "log2",
    "log10",
    "mag2db",
    "pow2db",
    "db",
    "db2mag",
    "db2pow",
    "sinc",
    "hypot",
    "dechirp",
    "matched_filter",
    "window",
    "hanning",
    "hamming",
    "blackman",
    "kaiser",
    "taylorwin",
]

# --------------------------------------------------------------------- #
# Transforms and data movement
# --------------------------------------------------------------------- #


@op
def fft2(x):
    """2-D forward FFT (Matlab `fft2`): both axes, rows then columns."""
    return fft(fft(x, axis=1), axis=0)


@op
def ifft2(x):
    """2-D inverse FFT (Matlab `ifft2`)."""
    return ifft(ifft(x, axis=1), axis=0)


@op
def circshift(x: Tensor,
              shift: int,
              dim: Optional[int] = None,
              axis: Optional[int] = None) -> Tensor:
    """Circularly shifts elements along one axis (Matlab `circshift`,
    numpy `roll`): `result[i] = input[(i - shift) mod n]`."""
    dim = _resolve_axis("circshift", dim, axis)
    x = _require_tensor(x, "sar.circshift")
    if not 0 <= dim < x.rank:
        raise TraceError(f"circshift dim {dim} out of range for rank {x.rank}")
    n = x.shape[dim]
    k = int(shift) % n
    if k == 0:
        return x
    if x.rank == 1:
        rolled = concatenate((x[n - k:], x[:n - k]), dim=0)
    elif dim == 0:
        rolled = concatenate((x[n - k:, :], x[:n - k, :]), dim=0)
    else:
        rolled = concatenate((x[:, n - k:], x[:, :n - k]), dim=1)
    # Slice/concat are structural, so they drop the spectral state. A roll
    # keeps every axis in the domain it was in -- only the position of the
    # zero-frequency bin on the shifted axis becomes unknown -- so restore
    # it here and keep the domain diagnostics live across a circshift.
    axes = list(x._axes)
    axes[dim] = (axes[dim][0], None)
    rolled._axes = tuple(axes)
    return rolled


@op
def mean(x: Tensor,
         dim: Optional[int] = None,
         axis: Optional[int] = None) -> Tensor:
    """Arithmetic mean (Matlab/numpy `mean`); see `sar.sum` for the
    shape conventions."""
    d = _resolve_axis("mean", dim, axis, required=False)
    count = (float(np.prod(x.shape))
             if d is None else float(x.shape[d if x.rank == 2 else 0]))
    return _sum(x, dim=d) / count


def _repeat_vector(x: Tensor, size: int) -> Tensor:
    """Repeats a length-one reduction result without adding an IR primitive."""
    result = x
    while result.shape[0] < size:
        take = min(result.shape[0], size - result.shape[0])
        result = concatenate((result, result[:take]), dim=0)
    return result


def _broadcast_mean(mu: Tensor, x: Tensor, dim: Optional[int]) -> Tensor:
    if x.rank == 1:
        return _repeat_vector(mu, x.shape[0])
    if dim == 0:
        return broadcast(mu, x.shape, dim=1)
    if dim == 1:
        return broadcast(mu, x.shape, dim=0)
    row = _repeat_vector(mu, x.shape[1])
    return broadcast(row, x.shape, dim=1)


@op
def var(x: Tensor,
        dim: Optional[int] = None,
        axis: Optional[int] = None) -> Tensor:
    """Two-pass population variance, normalized by N like numpy.

    Complex inputs use squared magnitude deviations and therefore return
    a real tensor, matching numpy.
    """
    d = _resolve_axis("var", dim, axis, required=False)
    mu = mean(x, dim=d)
    centered = x - _broadcast_mean(mu, x, d)
    if centered.dtype.is_complex:
        centered = absolute(centered)
    return mean(centered * centered, dim=d)


@op
def std(x: Tensor,
        dim: Optional[int] = None,
        axis: Optional[int] = None) -> Tensor:
    """Standard deviation (numpy convention: normalized by N)."""
    return sqrt(var(x, dim=dim, axis=axis))


# --------------------------------------------------------------------- #
# Element-wise math
# --------------------------------------------------------------------- #

_LN2 = float(np.log(2.0))
_LN10 = float(np.log(10.0))


@op
def log2(x):
    """Base-2 logarithm of a float tensor."""
    return log(x) * (1.0 / _LN2)


@op
def log10(x):
    """Base-10 logarithm of a float tensor."""
    return log(x) * (1.0 / _LN10)


@op
def mag2db(x):
    """Magnitude to decibels, `20 log10(x)` (Matlab `mag2db`)."""
    return log(x) * (20.0 / _LN10)


@op
def pow2db(x):
    """Power to decibels, `10 log10(x)` (Matlab `pow2db`)."""
    return log(x) * (10.0 / _LN10)


#: Matlab `db(x)` defaults to the voltage (magnitude) convention.
db = mag2db


@op
def db2mag(x):
    """Decibels to magnitude, `10^(x/20)` (Matlab `db2mag`)."""
    return exp(x * (_LN10 / 20.0))


@op
def db2pow(x):
    """Decibels to power, `10^(x/10)` (Matlab `db2pow`)."""
    return exp(x * (_LN10 / 10.0))


@op
def sinc(x):
    """Normalized sinc, `sin(pi x) / (pi x)` with `sinc(0) = 1` (numpy
    convention, like Matlab `sinc`)."""
    safe = where(x == 0.0, 1.0, x)
    return where(x == 0.0, 1.0, sin(safe * np.pi) / (safe * np.pi))


@op
def hypot(a, b):
    """`sqrt(a^2 + b^2)` element-wise (Matlab/numpy `hypot`)."""
    return sqrt(a * a + b * b)


@op
def multilook(x: Tensor, factors) -> Tensor:
    """Non-overlapping box average over `factors = (azimuth, range)`
    blocks (multilook speckle reduction): the mean of the strided
    sub-grids; element-wise fusion collapses it into one pass."""
    x = _require_tensor(x, "sar.multilook")
    if x.rank != 2 or not x.dtype.is_float:
        raise TraceError("sar.multilook expects a rank-2 float tensor")
    fa, fr = (int(f) for f in factors)
    if fa < 1 or fr < 1:
        raise TraceError("multilook factors must be positive")
    if x.shape[0] % fa or x.shape[1] % fr:
        raise TraceError(
            f"multilook factors {(fa, fr)} must divide the shape {x.shape}")
    acc = None
    for di in range(fa):
        for dj in range(fr):
            piece = x[di::fa, dj::fr]
            acc = piece if acc is None else acc + piece
    return acc * (1.0 / (fa * fr))


@op(const=("fa", "fr"))
def stolt_interp(data: Tensor,
                 fa: np.ndarray,
                 fr: np.ndarray,
                 *,
                 c: float,
                 fc: float,
                 vr: float,
                 t_shift: float,
                 kernel: str = "sinc",
                 taps: int = 8,
                 window: str = "hann",
                 beta: float = 2.5) -> Tensor:
    """Stolt interpolation (omega-K frequency remapping), composed from
    primitives: an in-kernel position computation feeding `sar.interp1d`,
    wrapped in the fast-time re-referencing ramps.

    `fa` / `fr` are the uniform azimuth/range frequency axes as numpy
    arrays; they bake into the kernel as constants (acquisition
    metadata, like the scalar radar parameters). `kernel`/`taps`/
    `window`/`beta` select the interpolator (see `sar.interp1d`).
    """
    data = _require_tensor(data, "sar.stolt_interp")
    if data.rank != 2 or not data.dtype.is_complex:
        raise TraceError("stolt_interp expects rank-2 complex data")
    fa = np.asarray(fa, dtype=np.float64)
    fr = np.asarray(fr, dtype=np.float64)
    if fa.shape != (data.shape[0], ) or fr.shape != (data.shape[1], ):
        raise TraceError(
            f"stolt_interp axis lengths {fa.shape}/{fr.shape} do not "
            f"match data shape {data.shape}")
    df = float(fr[1] - fr[0])

    shape = data.shape
    fa2 = broadcast(constant(fa), shape, dim=0)
    fr2 = broadcast(constant(fr), shape, dim=1)

    # Output-grid frequency mapped onto the input axis:
    #   f_q = sqrt((fr + fc)^2 + (c fa / 2 vr)^2) - fc
    fr_shifted = fr2 + fc
    fa_scaled = fa2 * (c / (2.0 * vr))
    f_q = sqrt(maximum(fr_shifted * fr_shifted + fa_scaled * fa_scaled,
                       1e-10)) - fc
    positions = (f_q - float(fr[0])) * (1.0 / df)

    # Smoothing ramp re-references fast time to the scene center before
    # the nonlinear remapping; the inverse ramp restores it on the
    # output grid (see the omega-K reference for the derivation).
    # Positions stay f64 above (interp1d's contract). The ramps multiply the
    # data, so building them at f64 would promote a c64 pipeline to c128 and
    # double the arithmetic downstream; keep them at the data's precision.
    ramp_axis = fr2 if data.dtype.name == "c128" else cast(fr2, f32)
    ramp = expj(ramp_axis * (2.0 * np.pi * t_shift))
    deramp = expj(ramp_axis * (-2.0 * np.pi * t_shift))
    return deramp * interp1d(data * ramp,
                             positions,
                             axis=1,
                             kernel=kernel,
                             taps=taps,
                             window=window,
                             beta=beta)


# --------------------------------------------------------------------- #
# Radar staples
# --------------------------------------------------------------------- #


@op
def dechirp(x: Tensor, ref: Tensor) -> Tensor:
    """Dechirp-on-receive: mixes each pulse with the conjugated
    reference (Matlab `dechirp`). `ref` is a fast-time vector (or a full
    raster); rank-1 references broadcast along the last axis."""
    x = _require_tensor(x, "sar.dechirp")
    ref = _require_tensor(ref, "sar.dechirp")
    if ref.rank == 1 and x.rank == 2:
        ref = broadcast(ref, x.shape, dim=1)
    return x * conj(ref)


@op
def matched_filter(data: Tensor,
                   replica,
                   dim: Optional[int] = None,
                   axis: Optional[int] = None) -> Tensor:
    """Frequency-domain matched filtering along one axis:
    `ifft(fft(data) * conj(fft(replica)))` with the (1-D) replica
    broadcast across the other axis. Circular convolution semantics;
    zero-pad the replica to the data length on the host."""
    d = _resolve_axis("matched_filter", dim, axis, required=False)
    d = 1 if d is None else d
    data = _require_tensor(data, "sar.matched_filter")
    replica = _require_tensor(replica, "sar.matched_filter")
    if replica.rank != 1 or replica.shape[0] != data.shape[d]:
        raise TraceError(
            "matched_filter expects a rank-1 replica matching the data "
            "length along the filtered axis")
    spectrum = conj(fft(replica, axis=0))
    if data.rank == 2:
        spectrum = broadcast(spectrum, data.shape, dim=d)
    return ifft(fft(data, axis=d) * spectrum, axis=d)


# --------------------------------------------------------------------- #
# Window constants (baked into the IR at trace time)
# --------------------------------------------------------------------- #


def _taylor(n: int, nbar: int = 4, sll: float = -30.0) -> np.ndarray:
    """Taylor window (Matlab `taylorwin`), the SAR standard: near-uniform
    aperture efficiency with the first `nbar - 1` sidelobes at `sll` dB."""
    a = np.arccosh(10.0**(-sll / 20.0)) / np.pi
    sigma2 = nbar**2 / (a**2 + (nbar - 0.5)**2)
    m = np.arange(1, nbar)
    f = np.array([(-1)**(mi + 1) / 2.0 *
                  np.prod(1.0 - mi**2 / (sigma2 * (a**2 + (m - 0.5)**2))) /
                  np.prod([1.0 - mi**2 / k**2 for k in m if k != mi])
                  for mi in m])
    x = (np.arange(n) + 0.5) / n - 0.5
    w = 1.0 + 2.0 * np.sum(
        f[:, np.newaxis] * np.cos(2.0 * np.pi * m[:, np.newaxis] * x), axis=0)
    return w / w.max()


_WINDOWS = {
    "hanning": lambda n: np.hanning(n),
    "hamming": lambda n: np.hamming(n),
    "blackman": lambda n: np.blackman(n),
    "kaiser": lambda n, beta=6.0: np.kaiser(n, beta),
    "taylor": _taylor,
    "rect": lambda n: np.ones(n),
}


@op
def window(kind: str, n: int, **params) -> Tensor:
    """A length-`n` window: `window("taylor", 512, sll=-35)`; kinds:
    hanning, hamming, blackman, kaiser, taylor, rect.

    Windows are constant *constructors*, not operators: they take a length
    rather than a tensor. Inside a kernel the samples bake in as a
    `sar.constant`; outside one this returns the plain numpy array, so the
    same call works in host code (`np.fft.fft(x) * sar.hanning(n)`).
    """
    if kind not in _WINDOWS:
        raise TraceError(
            f"unknown window '{kind}'; available: {sorted(_WINDOWS)}")
    samples = _WINDOWS[kind](int(n), **params)
    return constant(samples) if _tracing() else samples


@op
def hanning(n: int) -> Tensor:
    """Length-`n` Hann window constant (Matlab/numpy `hanning`)."""
    return window("hanning", n)


@op
def hamming(n: int) -> Tensor:
    """Length-`n` Hamming window constant."""
    return window("hamming", n)


@op
def blackman(n: int) -> Tensor:
    """Length-`n` Blackman window constant."""
    return window("blackman", n)


@op
def kaiser(n: int, beta: float = 6.0) -> Tensor:
    """Length-`n` Kaiser window constant with shape parameter `beta`."""
    return window("kaiser", n, beta=beta)


@op
def taylorwin(n: int, nbar: int = 4, sll: float = -30.0) -> Tensor:
    """Length-`n` Taylor window constant (Matlab `taylorwin`): `nbar - 1`
    near-in sidelobes held at `sll` dB."""
    return window("taylor", n, nbar=nbar, sll=sll)
