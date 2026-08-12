"""SAR-DSL: an MLIR-based domain-specific compiler for SAR imaging.

Quick start::

    import numpy as np
    import sar

    N = 512

    @sar.jit
    def scale_add(x: sar.c64[N, N], y: sar.c64[N, N]) -> sar.c64[N, N]:
        return x * 2.0 + y

    out = scale_add(x_np, y_np)                    # run on CPU
    hls = scale_add.compile(backend="scalehls")    # or emit HLS C++
"""

from ._version import __version__
from .backends import get_backend, list_backends
from .compiler import compile  # noqa: A001 - deliberate builtin shadow
from .errors import CompilationError, LaunchError, SARError, ToolchainError
from .language import (Kernel, Tensor, absolute, broadcast, c64, c128, cast,
                       constant, cos, expj, f32, f64, fft, fftshift, i32,
                       i64, ifft, ifftshift, interp1d, jit, kernel, maximum,
                       sin, sqrt, stolt_interp, transpose)

__all__ = [
    "__version__",
    # types
    "f32", "f64", "i32", "i64", "c64", "c128",
    # language
    "jit", "kernel", "Kernel", "Tensor",
    "constant", "sqrt", "cos", "sin", "absolute", "expj", "cast", "maximum",
    "transpose", "broadcast", "fft", "ifft", "fftshift", "ifftshift",
    "interp1d", "stolt_interp",
    # driver
    "compile", "get_backend", "list_backends",
    # errors
    "SARError", "ToolchainError", "CompilationError", "LaunchError",
]
