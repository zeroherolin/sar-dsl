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
import functools
import warnings
from typing import List, Sequence

import numpy as np

from ..errors import LaunchError
from ..ir import TensorType

__all__ = [
    "CompiledKernel", "clear_fft_plan_cache", "fft_plan_cache_size",
    "make_descriptor", "thread_config"
]

_descriptor_cache = {}


@functools.lru_cache(maxsize=None)
def _load_runtime(path: str):
    """dlopens the runtime once per path and declares its prototypes.

    Path resolution stays uncached (see toolchain.find_runtime_library):
    a changed path loads the new library, the same path reuses the
    handle.
    """
    library = ctypes.CDLL(path)
    library.sar_rt_fft_plan_cache_clear.restype = None
    library.sar_rt_fft_plan_cache_clear.argtypes = []
    for symbol in ("sar_rt_fft_plan_cache_size", "sar_rt_worker_count",
                   "sar_rt_available_worker_count",
                   "sar_rt_current_affinity_count"):
        fn = getattr(library, symbol)
        fn.restype = ctypes.c_int64
        fn.argtypes = []
    return library


def _runtime_library():
    from ..compiler.toolchain import find_runtime_library
    return _load_runtime(find_runtime_library())


def _runtime_error_api(library):
    clear = library.sar_rt_error_clear
    clear.restype = None
    clear.argtypes = []
    get = library.sar_rt_error_get
    get.restype = ctypes.c_char_p
    get.argtypes = []
    return clear, get


def clear_fft_plan_cache() -> None:
    """Drops cached immutable CPU FFT plans; active calls keep their plan."""
    _runtime_library().sar_rt_fft_plan_cache_clear()


def fft_plan_cache_size() -> int:
    """Number of transform lengths currently held by the CPU runtime."""
    return int(_runtime_library().sar_rt_fft_plan_cache_size())


def thread_config() -> dict:
    """Runtime-pool participants and CPUs visible through process affinity."""
    library = _runtime_library()
    return {
        "runtime_workers": int(library.sar_rt_worker_count()),
        "available_workers": int(library.sar_rt_available_worker_count()),
        "current_affinity_workers":
        int(library.sar_rt_current_affinity_count()),
    }


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
        try:
            self._error_clear, self._error_get = _runtime_error_api(
                self._library)
        except AttributeError:
            # The runtime reports failures by setting an error and
            # returning early, so without this channel an invalid call
            # returns uninitialized output with no diagnostic.
            self._error_clear = self._error_get = None
            warnings.warn(
                f"compiled kernel library {self.library_path} does not "
                "expose sar_rt_error_get; runtime error reporting is "
                "unavailable and invalid runtime calls will fail silently",
                RuntimeWarning,
                stacklevel=2)

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
        if self._error_clear is not None:
            self._error_clear()
        self._fn(*[ctypes.byref(d) for d in descriptors])
        if self._error_get is not None:
            error = self._error_get()
            if error:
                raise LaunchError(
                    f"kernel '{self.name}' runtime failure: {error.decode()}")

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def __repr__(self) -> str:
        args = ", ".join(t.mlir for t in self.arg_types)
        return f"CompiledKernel({self.name}({args}) @ {self.library_path})"
