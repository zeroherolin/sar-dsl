"""Trace-time diagnostics and source attribution for SAR kernels."""

from __future__ import annotations

import inspect
import os
import warnings
from typing import Sequence

from ..ir import DType

__all__ = ["DomainWarning", "PrecisionWarning"]


class PrecisionWarning(UserWarning):
    """Host data widened a kernel's working precision."""


class DomainWarning(UserWarning):
    """A tensor expression mixes suspicious spectral-domain states."""


_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UNKNOWN_AXIS = (None, None)


def source_location():
    """Returns the nearest user frame as an MLIR file/line/column tuple."""
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame else None
        while frame is not None:
            path = os.path.abspath(frame.f_code.co_filename)
            if not path.startswith(_PACKAGE_ROOT):
                return path, frame.f_lineno, 1
            frame = frame.f_back
    finally:
        del frame
    return None


def warn_user(message: str, category: type) -> None:
    """Emits a warning attributed to the nearest frame outside the package."""
    depth = 2
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame else None
        while frame is not None:
            path = os.path.abspath(frame.f_code.co_filename)
            if not path.startswith(_PACKAGE_ROOT):
                break
            depth += 1
            frame = frame.f_back
    finally:
        del frame
    warnings.warn(message, category, stacklevel=depth)


def propagate_axes(op: str, operands: Sequence, attributes, unit_attributes,
                   result_rank: int):
    """Derives conservative per-axis spectral state for an op result."""
    unknown = (_UNKNOWN_AXIS, ) * result_rank
    preserve = operands[0]._axes if operands else unknown

    if op in ("fft", "ifft", "fftshift"):
        dim = attributes["dim"]
        axes = list(preserve)
        spectral, centered = axes[dim]
        if op == "fft":
            if spectral is True:
                warn_user(
                    f"sar.fft along dim {dim}: axis is already in the "
                    "frequency domain", DomainWarning)
            axes[dim] = (True, False)
        elif op == "ifft":
            if spectral is False:
                warn_user(
                    f"sar.ifft along dim {dim}: axis is in the time domain",
                    DomainWarning)
            elif centered is True:
                warn_user(
                    f"sar.ifft along dim {dim}: spectrum is still centered; "
                    "apply ifftshift first", DomainWarning)
            axes[dim] = (False, None)
        else:
            inverse = "inverse" in unit_attributes
            axes[dim] = (spectral, False if inverse else True)
        return tuple(axes)

    if op == "transpose":
        return preserve[::-1]
    if op == "broadcast":
        axes = [_UNKNOWN_AXIS, _UNKNOWN_AXIS]
        axes[attributes["dim"]] = preserve[0]
        return tuple(axes)
    if op in ("add", "sub", "mul", "div", "complex", "atan2", "cmp", "where"):
        merged = []
        for dim, states in enumerate(
                zip(*(tensor._axes for tensor in operands))):
            known = {state for state in states if state[0] is not None}
            if len({state[0] for state in known}) > 1:
                warn_user(
                    f"sar.{op}: operands mix time-domain and "
                    "frequency-domain axes", DomainWarning)
            else:
                centered = {
                    state[1]
                    for state in known if state[1] is not None
                }
                if len(centered) > 1:
                    warn_user(
                        f"sar.{op} along dim {dim}: one operand is a centered "
                        "spectrum and the other is not; apply "
                        "fftshift/ifftshift to align them", DomainWarning)
            merged.append(
                next(iter(known)) if len(known) == 1 else _UNKNOWN_AXIS)
        return tuple(merged)
    if op in ("add_scalar", "mul_scalar", "sqrt", "cos", "sin", "exp", "log",
              "abs", "cast", "conj", "real", "imag"):
        return preserve[:result_rank]
    return unknown


def is_double(dtype: DType) -> bool:
    """Whether a dtype carries double-precision components."""
    return dtype.name in ("f64", "c128")


def warn_if_host_widens(a, b, target: DType, op: str) -> None:
    """Reports host data that promotes a narrow expression to double."""
    if not is_double(target):
        return
    for wide, narrow in ((a, b), (b, a)):
        if (getattr(wide, "_from_host", False) and is_double(wide.dtype)
                and not is_double(narrow.dtype)):
            warn_user(
                f"sar.{op}: host data of dtype {wide.dtype.name} widens a "
                f"{narrow.dtype.name} pipeline to {target.name}; cast the "
                "host array to single precision to keep it narrow",
                PrecisionWarning)
            return
