"""SAR-DSL: an MLIR-based domain-specific compiler for SAR imaging.

Quick start::

    import numpy as np
    import sar

    N = 512

    @sar.func
    def scale_add(x: sar.c64[N, N], y: sar.c64[N, N]) -> sar.c64[N, N]:
        return x * 2.0 + y

    out = scale_add(x_np, y_np)                    # run on CPU
    hls = scale_add.compile(backend="hls")    # or emit HLS C++
"""

from ._version import __version__
from .backends import get_backend, list_backends
from .backends.hls import HLSConfigError
from .compiler import compile  # noqa: A001 - deliberate builtin shadow
from .errors import (CompilationError, LaunchError, SARError, ToolchainError,
                     TraceError)
from .language import DomainWarning, PrecisionWarning, func, op
from .language import (GenericKernel, Kernel, Tensor, absolute, angle, argmax,
                       argmin, atan2, broadcast, c64, c128, cast, ceil, clip,
                       concat, concatenate, conj, conjugate, constant, cos,
                       exp, expj, dynamic_slice, dynamic_update_slice, f32,
                       f64, fft, fftshift, floor, gather2d, i32, i64, ifft,
                       ifftshift, imag, interp1d, iterate, log, make_complex,
                       maximum, minimum, pad, real, sign, sin, sqrt, transpose,
                       where)
from .language import abs, max, min, round, sum  # noqa: A004 - numpy
from .language import flip, cumsum, rank_filter, median_filter, sort
from .language.signal import (blackman, circshift, db, db2mag, db2pow, dechirp,
                              fft2, hamming, hanning, hypot, ifft2, kaiser,
                              log2, log10, mag2db, matched_filter, mean,
                              multilook, pow2db, sinc, std, stolt_interp,
                              taylorwin, var, window)

__all__ = [
    "__version__",
    # types
    "f32",
    "f64",
    "i32",
    "i64",
    "c64",
    "c128",
    # language
    "func",
    "op",
    "GenericKernel",
    "Kernel",
    "Tensor",
    "constant",
    "sqrt",
    "cos",
    "sin",
    "exp",
    "log",
    "atan2",
    "absolute",
    "abs",
    "expj",
    "cast",
    "maximum",
    "minimum",
    "clip",
    "where",
    "conj",
    "conjugate",
    "real",
    "imag",
    "angle",
    "make_complex",
    "sum",
    "max",
    "min",
    "argmax",
    "argmin",
    "sign",
    "floor",
    "ceil",
    "round",
    "transpose",
    "broadcast",
    "dynamic_slice",
    "dynamic_update_slice",
    "concatenate",
    "concat",
    "pad",
    "multilook",
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftshift",
    "ifftshift",
    "flip",
    "circshift",
    "mean",
    "std",
    "var",
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
    "interp1d",
    "gather2d",
    "iterate",
    "stolt_interp",
    "cumsum",
    "rank_filter",
    "median_filter",
    "sort",
    # driver
    "compile",
    "get_backend",
    "list_backends",
    # errors and diagnostics
    "SARError",
    "TraceError",
    "ToolchainError",
    "CompilationError",
    "LaunchError",
    "HLSConfigError",
    "DomainWarning",
    "PrecisionWarning",
]
