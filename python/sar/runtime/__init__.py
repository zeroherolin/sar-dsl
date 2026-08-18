"""Execution runtime: ctypes marshalling of numpy arrays to MLIR memrefs.

Compiled CPU kernels export ``_mlir_ciface_<name>`` taking one pointer per
tensor: first the inputs, then one output pointer per result
(destination-passing style produced by `buffer-results-to-out-params`).
Each pointer refers to a StridedMemRef descriptor::

    { T *allocated; T *aligned;
      int64 offset; int64 sizes[R]; int64 strides[R]; }
"""

from __future__ import annotations

import ctypes
from typing import List, Sequence

import numpy as np

from ..errors import LaunchError
from ..ir import TensorType

__all__ = ["CompiledKernel", "make_descriptor"]

_descriptor_cache = {}


def _descriptor_type(rank: int):
    if rank not in _descriptor_cache:

        class Descriptor(ctypes.Structure):
            _fields_ = [
                ("allocated", ctypes.c_void_p),
                ("aligned", ctypes.c_void_p),
                ("offset", ctypes.c_int64),
                ("sizes", ctypes.c_int64 * rank),
                ("strides", ctypes.c_int64 * rank),
            ]

        Descriptor.__name__ = f"MemRefDescriptor{rank}D"
        _descriptor_cache[rank] = Descriptor
    return _descriptor_cache[rank]


def make_descriptor(array: np.ndarray):
    """Builds the MLIR C-interface memref descriptor for `array`
    (strides in elements). The array must stay alive while the
    descriptor is in use."""
    rank = array.ndim
    desc = _descriptor_type(rank)()
    address = array.ctypes.data
    desc.allocated = address
    desc.aligned = address
    desc.offset = 0
    itemsize = array.itemsize
    for i in range(rank):
        desc.sizes[i] = array.shape[i]
        desc.strides[i] = array.strides[i] // itemsize
    return desc


def _check_argument(index: int, array, expected: TensorType) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise LaunchError(
            f"argument #{index} must be a numpy array, got {type(array)!r}")
    if tuple(array.shape) != expected.shape:
        raise LaunchError(
            f"argument #{index} has shape {tuple(array.shape)}, expected "
            f"{expected.shape}")
    expected_np = expected.dtype.to_numpy()
    if array.dtype != expected_np:
        raise LaunchError(
            f"argument #{index} has dtype {array.dtype}, expected "
            f"{expected_np} ({expected.dtype.name})")
    return np.ascontiguousarray(array)


class CompiledKernel:
    """Callable wrapper around a compiled CPU kernel shared library."""

    def __init__(self, library_path: str, name: str,
                 arg_types: Sequence[TensorType],
                 result_types: Sequence[TensorType]):
        self.library_path = str(library_path)
        self.name = name
        self.arg_types = list(arg_types)
        self.result_types = list(result_types)
        try:
            self._library = ctypes.CDLL(self.library_path)
        except OSError as exc:
            raise LaunchError(
                f"cannot load compiled kernel library {self.library_path}: "
                f"{exc}") from exc
        symbol = f"_mlir_ciface_{name}"
        try:
            self._fn = getattr(self._library, symbol)
        except AttributeError as exc:
            raise LaunchError(
                f"compiled kernel library {self.library_path} does not export "
                f"{symbol}") from exc
        self._fn.restype = None
        # One descriptor pointer per tensor, inputs then results. Declaring
        # them lets ctypes reject an arity mismatch instead of corrupting
        # the stack when metadata and library disagree.
        self._fn.argtypes = [
            ctypes.POINTER(_descriptor_type(t.rank))
            for t in list(arg_types) + list(result_types)
        ]

    def __call__(self, *arrays):
        """Runs the kernel on numpy arrays, returning new output arrays.

        Inputs must match the declared shapes and dtypes exactly. A
        non-contiguous input (a slice, a transpose view) is copied whole
        into a contiguous buffer before the call; pass contiguous arrays
        to avoid the copy on large planes.
        """
        if len(arrays) != len(self.arg_types):
            raise LaunchError(
                f"kernel '{self.name}' takes {len(self.arg_types)} "
                f"argument(s), got {len(arrays)}")

        inputs = [
            _check_argument(i, a, t)
            for i, (a, t) in enumerate(zip(arrays, self.arg_types))
        ]
        outputs: List[np.ndarray] = [
            np.empty(t.shape, dtype=t.dtype.to_numpy())
            for t in self.result_types
        ]

        descriptors = [make_descriptor(a) for a in inputs + outputs]
        self._fn(*[ctypes.byref(d) for d in descriptors])

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def __repr__(self) -> str:
        args = ", ".join(t.mlir for t in self.arg_types)
        return f"CompiledKernel({self.name}({args}) @ {self.library_path})"
