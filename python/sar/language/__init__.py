"""User-facing SAR DSL: types, tensor expressions and the @jit decorator.

Kernels are plain Python functions whose parameters carry tensor type
annotations. Calling the traced function symbolically records `sar` dialect
operations::

    import sar

    N = 512

    @sar.jit
    def scale(x: sar.c64[N, N]) -> sar.c64[N, N]:
        return x * 2.0

    y = scale(np_array)                      # JIT on the CPU backend
    design = scale.compile(backend="scalehls")   # emit HLS C++ instead
"""

from __future__ import annotations

import functools
import inspect
import threading
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .. import ir
from ..ir import (C64, C128, COMPLEX_OF, DTYPES, F32, F64,
                  FLOAT_PRECISION_OF, I32, I64, DType, TensorType)

__all__ = [
    "f32", "f64", "i32", "i64", "c64", "c128",
    "Tensor", "jit", "kernel", "Kernel",
    "constant", "sqrt", "cos", "sin", "absolute", "expj", "cast", "maximum",
    "transpose", "broadcast", "fft", "ifft", "fftshift", "ifftshift",
    "interp1d", "stolt_interp",
]


class TraceError(TypeError):
    """Raised when a kernel is traced with inconsistent types or shapes."""


# --------------------------------------------------------------------------- #
# Type annotation sugar: sar.c64[512, 512]
# --------------------------------------------------------------------------- #

class _DTypeSpec:
    """`sar.c64` etc.; indexing produces a TensorType annotation."""

    def __init__(self, dtype: DType):
        self.dtype = dtype

    def __getitem__(self, shape) -> TensorType:
        if not isinstance(shape, tuple):
            shape = (shape,)
        return TensorType(tuple(int(d) for d in shape), self.dtype)

    def __repr__(self) -> str:
        return repr(self.dtype)


f32 = _DTypeSpec(F32)
f64 = _DTypeSpec(F64)
i32 = _DTypeSpec(I32)
i64 = _DTypeSpec(I64)
c64 = _DTypeSpec(C64)
c128 = _DTypeSpec(C128)

_SPEC_BY_DTYPE = {F32: f32, F64: f64, I32: i32, I64: i64, C64: c64, C128: c128}


# --------------------------------------------------------------------------- #
# Tracing context
# --------------------------------------------------------------------------- #

_state = threading.local()


def _current_function() -> ir.Function:
    fn = getattr(_state, "function", None)
    if fn is None:
        raise TraceError(
            "SAR operations can only be used inside a function decorated "
            "with @sar.jit while it is being traced")
    return fn


class Tensor:
    """Symbolic tensor value recorded during tracing."""

    __array_priority__ = 1000  # keep numpy from hijacking operators

    def __init__(self, value: ir.Value):
        self._value = value

    # -- introspection ------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._value.type.shape

    @property
    def dtype(self) -> DType:
        return self._value.type.dtype

    @property
    def rank(self) -> int:
        return self._value.type.rank

    def __repr__(self) -> str:
        return f"sar.Tensor({self._value.type.mlir})"

    # -- helpers ------------------------------------------------------------

    def _emit(self, op: str, operands: Sequence["Tensor"],
              result_type: TensorType, attributes=None,
              unit_attributes=()) -> "Tensor":
        fn = _current_function()
        value = fn.emit(f"sar.{op}", [t._value for t in operands],
                        result_type, attributes, unit_attributes)
        return Tensor(value)

    def _binary(self, op: str, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TraceError(f"expected a SAR tensor, got {type(other)!r}")
        if other._value.type != self._value.type:
            raise TraceError(
                f"sar.{op}: operand types differ "
                f"({self._value.type.mlir} vs {other._value.type.mlir}); "
                "use sar.cast to align element types")
        return self._emit(op, [self, other], self._value.type)

    def _scalar(self, op: str, scalar: float) -> "Tensor":
        if self.dtype.is_int:
            raise TraceError(f"sar.{op} does not support integer tensors")
        return self._emit(op, [self], self._value.type,
                          {"scalar": float(scalar)})

    # -- operators ----------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self._scalar("add_scalar", other)
        return self._binary("add", other)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self._scalar("add_scalar", -float(other))
        return self._binary("sub", other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return (-self)._scalar("add_scalar", float(other))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self._scalar("mul_scalar", other)
        return self._binary("mul", other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self._scalar("mul_scalar", 1.0 / float(other))
        return self._binary("div", other)

    def __neg__(self):
        if self.dtype.is_int:
            raise TraceError("sar.neg does not support integer tensors")
        return self._emit("neg", [self], self._value.type)

    @property
    def T(self) -> "Tensor":
        return transpose(self)


# --------------------------------------------------------------------------- #
# Free functions
# --------------------------------------------------------------------------- #

def _require_tensor(x, what: str) -> Tensor:
    if not isinstance(x, Tensor):
        raise TraceError(f"{what} expects a SAR tensor, got {type(x)!r}")
    return x


def constant(value, dtype: Optional[_DTypeSpec] = None,
             shape: Optional[Sequence[int]] = None) -> Tensor:
    """Materializes a constant tensor from a numpy array or a scalar."""
    if isinstance(value, (int, float, complex)):
        if shape is None:
            raise TraceError("scalar constants require an explicit shape")
        if dtype is None:
            dtype = c64 if isinstance(value, complex) else f64
        array = np.full(tuple(shape), value,
                        dtype=dtype.dtype.to_numpy())
    else:
        array = np.asarray(value)
        if dtype is not None:
            array = array.astype(dtype.dtype.to_numpy())
        np_name = array.dtype.name
        matches = [d for d in DTYPES.values() if d.np_dtype == np_name]
        if not matches:
            raise TraceError(
                f"unsupported constant dtype {np_name}; supported: "
                f"{sorted(d.np_dtype for d in DTYPES.values())}")
        dtype = _SPEC_BY_DTYPE[matches[0]]

    ttype = TensorType(tuple(array.shape), dtype.dtype)
    attr = ir.DenseAttr.from_array(array, ttype)
    fn = _current_function()
    value_ = fn.emit("sar.constant", [], ttype, {"value": attr})
    return Tensor(value_)


def _float_unary(op: str, x: Tensor) -> Tensor:
    x = _require_tensor(x, f"sar.{op}")
    if not x.dtype.is_float:
        raise TraceError(f"sar.{op} expects a float tensor")
    return x._emit(op, [x], x._value.type)


def sqrt(x: Tensor) -> Tensor:
    return _float_unary("sqrt", x)


def cos(x: Tensor) -> Tensor:
    return _float_unary("cos", x)


def sin(x: Tensor) -> Tensor:
    return _float_unary("sin", x)


def absolute(x: Tensor) -> Tensor:
    """|x| element-wise; complex magnitude for complex tensors."""
    x = _require_tensor(x, "sar.abs")
    if x.dtype.is_int:
        raise TraceError("sar.abs does not support integer tensors")
    out_dtype = FLOAT_PRECISION_OF[x.dtype]
    return x._emit("abs", [x], TensorType(x.shape, out_dtype))


def expj(x: Tensor) -> Tensor:
    """exp(j*x) for a float tensor; returns a complex tensor."""
    x = _require_tensor(x, "sar.expj")
    if not x.dtype.is_float:
        raise TraceError("sar.expj expects a float tensor")
    return x._emit("expj", [x], TensorType(x.shape, COMPLEX_OF[x.dtype]))


def cast(x: Tensor, dtype: _DTypeSpec) -> Tensor:
    """Casts the element type (float<->float, complex<->complex,
    float->complex)."""
    x = _require_tensor(x, "sar.cast")
    target = dtype.dtype
    if x.dtype == target:
        return x
    if x.dtype.is_int or target.is_int:
        raise TraceError("sar.cast does not support integer tensors")
    if x.dtype.is_complex and not target.is_complex:
        raise TraceError(
            "cannot cast complex to float; use sar.absolute for magnitudes")
    return x._emit("cast", [x], TensorType(x.shape, target))


def maximum(x: Tensor, scalar: float) -> Tensor:
    x = _require_tensor(x, "sar.maximum")
    if not x.dtype.is_float:
        raise TraceError("sar.maximum expects a float tensor")
    return x._emit("max_scalar", [x], x._value.type,
                   {"scalar": float(scalar)})


def transpose(x: Tensor) -> Tensor:
    x = _require_tensor(x, "sar.transpose")
    if x.rank != 2:
        raise TraceError("sar.transpose expects a rank-2 tensor")
    out = TensorType((x.shape[1], x.shape[0]), x.dtype)
    return x._emit("transpose", [x], out)


def broadcast(x: Tensor, shape: Sequence[int], dim: int) -> Tensor:
    """Broadcasts a 1-D tensor to a 2-D shape; the vector lies along `dim`."""
    x = _require_tensor(x, "sar.broadcast")
    if x.rank != 1:
        raise TraceError("sar.broadcast expects a rank-1 tensor")
    shape = tuple(int(d) for d in shape)
    if len(shape) != 2 or dim not in (0, 1):
        raise TraceError("sar.broadcast currently supports rank-2 results "
                         "with dim in {0, 1}")
    if shape[dim] != x.shape[0]:
        raise TraceError(
            f"sar.broadcast: result dim {dim} is {shape[dim]} but the vector "
            f"has length {x.shape[0]}")
    return x._emit("broadcast", [x], TensorType(shape, x.dtype),
                   {"dim": dim})


def _fftshift(x: Tensor, dim: int, inverse: bool) -> Tensor:
    x = _require_tensor(x, "sar.fftshift")
    if not 0 <= dim < x.rank:
        raise TraceError(f"fftshift dim {dim} out of range for rank {x.rank}")
    return x._emit("fftshift", [x], x._value.type, {"dim": dim},
                   ("inverse",) if inverse else ())


def fftshift(x: Tensor, dim: int) -> Tensor:
    return _fftshift(x, dim, inverse=False)


def ifftshift(x: Tensor, dim: int) -> Tensor:
    return _fftshift(x, dim, inverse=True)


def _fft(op: str, x: Tensor, dim: int) -> Tensor:
    x = _require_tensor(x, f"sar.{op}")
    if not x.dtype.is_complex:
        raise TraceError(f"sar.{op} expects a complex tensor; cast first")
    if not 0 <= dim < x.rank:
        raise TraceError(f"{op} dim {dim} out of range for rank {x.rank}")
    n = x.shape[dim]
    if n < 2 or n & (n - 1):
        raise TraceError(f"sar.{op} size along dim must be a power of two, "
                         f"got {n}")
    return x._emit(op, [x], x._value.type, {"dim": dim})


def fft(x: Tensor, dim: int) -> Tensor:
    return _fft("fft", x, dim)


def ifft(x: Tensor, dim: int) -> Tensor:
    return _fft("ifft", x, dim)


def interp1d(data: Tensor, positions: Tensor) -> Tensor:
    """Windowed-sinc resampling of each row of `data` at fractional sample
    `positions` (both rank-2, same shape). The orthogonal primitive behind
    Stolt remapping and range cell migration correction."""
    data = _require_tensor(data, "sar.interp1d")
    positions = _require_tensor(positions, "sar.interp1d")
    if data.rank != 2 or not data.dtype.is_complex:
        raise TraceError("interp1d expects rank-2 complex data")
    if positions.dtype != F64 or positions.shape != data.shape:
        raise TraceError(
            "interp1d positions must be an f64 tensor with the data shape")
    return data._emit("interp1d", [data, positions], data._value.type)


def stolt_interp(data: Tensor, fa: Tensor, fr: Tensor, *, c: float,
                 fc: float, vr: float, t_shift: float) -> Tensor:
    """Stolt interpolation (omega-K frequency remapping); see the dialect
    documentation for the exact semantics."""
    data = _require_tensor(data, "sar.stolt_interp")
    fa = _require_tensor(fa, "sar.stolt_interp")
    fr = _require_tensor(fr, "sar.stolt_interp")
    if data.rank != 2 or not data.dtype.is_complex:
        raise TraceError("stolt_interp expects rank-2 complex data")
    if fa.dtype != F64 or fr.dtype != F64:
        raise TraceError("stolt_interp expects f64 frequency axes")
    if fa.shape != (data.shape[0],) or fr.shape != (data.shape[1],):
        raise TraceError(
            f"stolt_interp axis lengths {fa.shape[0]}/{fr.shape[0]} do not "
            f"match data shape {data.shape}")
    return data._emit("stolt_interp", [data, fa, fr], data._value.type,
                      {"c": float(c), "fc": float(fc), "vr": float(vr),
                       "t_shift": float(t_shift)})


# --------------------------------------------------------------------------- #
# Kernel tracing
# --------------------------------------------------------------------------- #

class Kernel:
    """A traced SAR kernel; compiles lazily per backend."""

    def __init__(self, fn: Callable):
        self._fn = fn
        self.name = fn.__name__
        functools.update_wrapper(self, fn)
        self.arg_types, self.declared_result_types = self._parse_signature(fn)
        self._module_text: Optional[str] = None
        self._compiled = {}

    @staticmethod
    def _resolve_annotation(annotation, fn: Callable):
        """Evaluates postponed (PEP 563) string annotations, including
        closure variables of factory functions building kernels."""
        if not isinstance(annotation, str):
            return annotation
        namespace = {}
        if fn.__closure__:
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                try:
                    namespace[name] = cell.cell_contents
                except ValueError:
                    pass
        return eval(annotation, getattr(fn, "__globals__", {}), namespace)

    @classmethod
    def _parse_signature(cls, fn: Callable):
        sig = inspect.signature(fn)
        arg_types: List[TensorType] = []
        for p in sig.parameters.values():
            if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                raise TraceError(
                    f"kernel '{fn.__name__}': parameter '{p.name}' must be "
                    "positional")
            annotation = cls._resolve_annotation(p.annotation, fn)
            if not isinstance(annotation, TensorType):
                raise TraceError(
                    f"kernel '{fn.__name__}': parameter '{p.name}' needs a "
                    "SAR type annotation such as sar.c64[512, 512]")
            arg_types.append(annotation)

        ret = sig.return_annotation
        if ret is inspect.Signature.empty:
            raise TraceError(
                f"kernel '{fn.__name__}' needs a return type annotation")
        ret = cls._resolve_annotation(ret, fn)
        rets = ret if isinstance(ret, tuple) else (ret,)
        for r in rets:
            if not isinstance(r, TensorType):
                raise TraceError(
                    f"kernel '{fn.__name__}': invalid return annotation {r!r}")
        return arg_types, list(rets)

    def trace(self) -> str:
        """Traces the Python function and returns the MLIR module text."""
        if self._module_text is not None:
            return self._module_text

        fn_ir = ir.Function(self.name, self.arg_types)
        prev = getattr(_state, "function", None)
        _state.function = fn_ir
        try:
            result = self._fn(*[Tensor(a) for a in fn_ir.arguments])
        finally:
            _state.function = prev

        results = result if isinstance(result, tuple) else (result,)
        if len(results) != len(self.declared_result_types):
            raise TraceError(
                f"kernel '{self.name}' returned {len(results)} values but "
                f"declares {len(self.declared_result_types)}")
        for i, (value, declared) in enumerate(
                zip(results, self.declared_result_types)):
            if not isinstance(value, Tensor):
                raise TraceError(
                    f"kernel '{self.name}' result #{i} is not a SAR tensor")
            if value._value.type != declared:
                raise TraceError(
                    f"kernel '{self.name}' result #{i} has type "
                    f"{value._value.type.mlir} but declares {declared.mlir}")

        fn_ir.set_return([v._value for v in results])
        self._module_text = ir.Module([fn_ir]).render()
        return self._module_text

    def to_mlir(self) -> str:
        return self.trace()

    def compile(self, backend: str = "cpu", options: Optional[dict] = None):
        """Compiles the kernel for `backend` and returns the launcher."""
        from ..compiler import compile as _compile
        key = (backend, tuple(sorted((options or {}).items())))
        if key not in self._compiled:
            self._compiled[key] = _compile(self, backend=backend,
                                           options=options)
        return self._compiled[key]

    def __call__(self, *arrays):
        """JIT-compiles for the CPU backend and executes."""
        return self.compile("cpu")(*arrays)


def jit(fn: Callable) -> Kernel:
    """Decorator turning an annotated Python function into a SAR kernel."""
    return Kernel(fn)


#: Alias kept for readers who prefer Triton-like naming.
kernel = jit
