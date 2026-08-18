"""User-facing SAR DSL: types, tensor expressions and the @sar.func
decorator.

Kernels are plain Python functions whose parameters carry tensor type
annotations. Calling the traced function symbolically records `sar` dialect
operations::

    import sar

    N = 512

    @sar.func
    def scale(x: sar.c64[N, N]) -> sar.c64[N, N]:
        return x * 2.0

    y = scale(np_array)                      # JIT on the CPU backend
    design = scale.compile(backend="hls")   # emit HLS C++ instead
"""

from __future__ import annotations

import functools
import inspect
import math
import numbers
import os
import re
import threading
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import ir
from ..errors import TraceError
from ..ir import (C64, C128, COMPLEX_OF, DTYPES, F32, F64, FLOAT_PRECISION_OF,
                  I32, I64, DType, TensorType)
from .diagnostics import (DomainWarning, PrecisionWarning, _UNKNOWN_AXIS,
                          is_double as _is_double, propagate_axes as
                          _propagate_axes, source_location as _source_location,
                          warn_if_host_widens as _warn_if_host_widens)


def _is_real_scalar(value) -> bool:
    return isinstance(value, numbers.Real)


__all__ = [
    "f32",
    "f64",
    "i32",
    "i64",
    "c64",
    "c128",
    "Tensor",
    "TraceError",
    "func",
    "Kernel",
    "GenericKernel",
    "DomainWarning",
    "PrecisionWarning",
    "op",
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
    "concatenate",
    "concat",
    "pad",
    "fft",
    "ifft",
    "fftshift",
    "ifftshift",
    "interp1d",
    "gather2d",
    "iterate",
    "cumsum",
    "rank_filter",
    "median_filter",
    "sort",
    "flip",
]

# --------------------------------------------------------------------------- #
# Type annotation sugar: sar.c64[512, 512]
# --------------------------------------------------------------------------- #


class _DTypeSpec:
    """`sar.c64` etc.; indexing produces a TensorType annotation."""

    def __init__(self, dtype: DType):
        self.dtype = dtype

    def __getitem__(self, shape) -> TensorType:
        if not isinstance(shape, tuple):
            shape = (shape, )
        dims = []
        for d in shape:
            # int(2.5) would silently truncate a mistyped dimension.
            if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
                raise TypeError(
                    f"sar.{self.dtype.name}[...]: dimension {d!r} is not an "
                    "integer; tensor shapes are static")
            dims.append(int(d))
        return TensorType(tuple(dims), self.dtype)

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
            "with @sar.func while it is being traced")
    return fn


def _promoted_dtype(a: DType, b: DType) -> DType:
    """NumPy-style promotion over the supported lattice: precision widens
    (32 -> 64) and float promotes to complex.

    Width is decided by component precision, not by the digits in the type
    name: `c64` is a pair of f32, so `f32 x c64` stays at `c64`.
    """
    if a == b:
        return a
    if a.is_int or b.is_int:
        raise TraceError("integer tensors do not promote implicitly")
    wide = _is_double(a) or _is_double(b)
    if a.is_complex or b.is_complex:
        return C128 if wide else C64
    return F64 if wide else F32


def _coerce_pair(a: "Tensor", b: "Tensor", op: str):
    """Aligns two operands numpy-style: dtype promotion plus rank-1 ->
    rank-2 broadcasting along the last axis."""
    if a.rank != b.rank:

        def expand(vec: "Tensor", mat: "Tensor") -> "Tensor":
            if vec.rank != 1 or mat.rank != 2 or \
                    vec.shape[0] != mat.shape[1]:
                raise TraceError(
                    f"sar.{op}: shapes {a.shape} and {b.shape} do not "
                    "broadcast (rank-1 operands broadcast along the last "
                    "axis; use sar.broadcast for explicit placement)")
            return broadcast(vec, mat.shape, dim=1)

        if a.rank < b.rank:
            a = expand(a, b)
        else:
            b = expand(b, a)
    if a.shape != b.shape:
        raise TraceError(
            f"sar.{op}: operand shapes differ ({a.shape} vs {b.shape})")
    target = _promoted_dtype(a.dtype, b.dtype)
    _warn_if_host_widens(a, b, target, op)
    if a.dtype != target:
        a = cast(a, _SPEC_BY_DTYPE[target])
    if b.dtype != target:
        b = cast(b, _SPEC_BY_DTYPE[target])
    return a, b


class Tensor:
    """Symbolic tensor value recorded during tracing."""

    __array_priority__ = 1000  # keep numpy from hijacking operators
    # Refuse numpy ufuncs outright: without this, `np.conj(x)` silently
    # builds an object array instead of tracing an op. Binary expressions
    # (`np_array * x`) still work -- numpy defers and Python falls back to
    # the reflected operator.
    __array_ufunc__ = None

    def __init__(self, value: ir.Value, axes=None, from_host=False):
        self._value = value
        self._axes = (tuple(axes) if axes is not None else
                      (_UNKNOWN_AXIS, ) * len(value.type.shape))
        # Set for values that entered the kernel as host data rather than as
        # an argument. Their dtype is the host's choice, so it is the one
        # worth reporting when a promotion widens the pipeline.
        self._from_host = from_host

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

    @property
    def ndim(self) -> int:
        """numpy's spelling of `rank`."""
        return self._value.type.rank

    @property
    def size(self) -> int:
        """Total element count (numpy `size`)."""
        return int(np.prod(self._value.type.shape))

    def __len__(self) -> int:
        return self._value.type.shape[0]

    def __repr__(self) -> str:
        return f"sar.Tensor({self._value.type.mlir})"

    # -- helpers ------------------------------------------------------------

    def _emit(self,
              op: str,
              operands: Sequence["Tensor"],
              result_type: TensorType,
              attributes=None,
              unit_attributes=()) -> "Tensor":
        fn = _current_function()
        value = fn.emit(f"sar.{op}", [t._value for t in operands], result_type,
                        attributes, unit_attributes, _source_location())
        axes = _propagate_axes(op, operands, attributes or {}, unit_attributes,
                               len(result_type.shape))
        # A value computed only from host data is still host data, so the
        # precision diagnostic can name the array that set its width even
        # when several ops separate the two.
        from_host = bool(operands) and any(
            getattr(t, "_from_host", False) for t in operands)
        return Tensor(value, axes, from_host=from_host)

    def _binary(self, op: str, other, reflected: bool = False) -> "Tensor":
        if isinstance(other, np.ndarray):
            other = constant(other)
        if not isinstance(other, Tensor):
            raise TraceError(f"expected a SAR tensor, got {type(other)!r}")
        lhs, rhs = _coerce_pair(self, other, op)
        if reflected:
            lhs, rhs = rhs, lhs
        return lhs._emit(op, [lhs, rhs], lhs._value.type)

    def _scalar(self, op: str, scalar: float) -> "Tensor":
        if self.dtype.is_int:
            raise TraceError(f"sar.{op} does not support integer tensors")
        return self._emit(op, [self], self._value.type,
                          {"scalar": float(scalar)})

    # -- operators ----------------------------------------------------------

    def __add__(self, other):
        if _is_real_scalar(other):
            return self._scalar("add_scalar", other)
        return self._binary("add", other)

    __radd__ = __add__

    def __sub__(self, other):
        if _is_real_scalar(other):
            return self._scalar("add_scalar", -float(other))
        return self._binary("sub", other)

    def __rsub__(self, other):
        if _is_real_scalar(other):
            return (-self)._scalar("add_scalar", float(other))
        return self._binary("sub", other, reflected=True)

    def __mul__(self, other):
        if _is_real_scalar(other):
            return self._scalar("mul_scalar", other)
        return self._binary("mul", other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if _is_real_scalar(other):
            return self._scalar("mul_scalar", 1.0 / float(other))
        return self._binary("div", other)

    def __rtruediv__(self, other):
        if _is_real_scalar(other):
            other = np.full(self.shape, other, dtype=self.dtype.to_numpy())
        return self._binary("div", other, reflected=True)

    def __neg__(self):
        if self.dtype.is_int:
            raise TraceError("negation does not support integer tensors")
        return self._scalar("mul_scalar", -1.0)

    def __pos__(self):
        return self

    def __abs__(self):
        return absolute(self)

    def _no_scalar_conversion(self, what: str):
        raise TraceError(
            f"a SAR tensor cannot convert to a Python {what}: values exist "
            "only when the compiled kernel runs. Reduce first (sar.sum, "
            "sar.max, ...) and read the result array outside the kernel")

    def __float__(self):
        self._no_scalar_conversion("float")

    def __int__(self):
        self._no_scalar_conversion("int")

    def __complex__(self):
        self._no_scalar_conversion("complex")

    # -- comparisons (0.0 / 1.0 masks for sar.where) -------------------------

    def _compare(self, predicate: str, other) -> "Tensor":
        if not self.dtype.is_float:
            raise TraceError("comparisons require float tensors "
                             "(compare magnitudes or components)")
        if isinstance(other, (int, float)):
            other = np.full(self.shape,
                            float(other),
                            dtype=self.dtype.to_numpy())
        if isinstance(other, np.ndarray):
            other = constant(other)
        lhs, rhs = _coerce_pair(self, other, "cmp")
        return lhs._emit("cmp", [lhs, rhs], lhs._value.type,
                         {"predicate": predicate})

    def __gt__(self, other):
        return self._compare("gt", other)

    def __ge__(self, other):
        return self._compare("ge", other)

    def __lt__(self, other):
        return self._compare("lt", other)

    def __le__(self, other):
        return self._compare("le", other)

    def __eq__(self, other):  # noqa: D105 - mask semantics, like numpy
        if isinstance(other, (Tensor, int, float, np.ndarray)):
            return self._compare("eq", other)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, (Tensor, int, float, np.ndarray)):
            return self._compare("ne", other)
        return NotImplemented

    __hash__ = object.__hash__  # __eq__ builds masks, not truth values

    def __bool__(self):
        raise TraceError(
            "a SAR tensor has no truth value: comparisons build element-wise "
            "masks, not booleans. Use sar.where(mask, a, b) for selection; "
            "kernel control flow must depend on Python values, not traced "
            "tensors")

    def __pow__(self, exponent):
        if exponent == 2:
            return self._binary("mul", self)
        if exponent == 0.5:
            return sqrt(self)
        if isinstance(exponent, int) and 1 <= exponent <= 8:
            result = self
            for _ in range(exponent - 1):
                result = result._binary("mul", self)
            return result
        raise TraceError(
            "tensor powers support small integer exponents and 0.5")

    # -- numpy-style methods --------------------------------------------------

    @property
    def real(self) -> "Tensor":
        return real(self) if self.dtype.is_complex else self

    @property
    def imag(self) -> "Tensor":
        return imag(self)

    def conj(self) -> "Tensor":
        return conj(self)

    #: numpy spells the same operation both ways.
    conjugate = conj

    def transpose(self) -> "Tensor":
        return transpose(self)

    def clip(self, lo: float, hi: float) -> "Tensor":
        return clip(self, lo, hi)

    def round(self) -> "Tensor":
        return round(self)

    def cumsum(self, axis: Optional[int] = None) -> "Tensor":
        return cumsum(self, axis=axis)

    def std(self, axis: Optional[int] = None) -> "Tensor":
        from .signal import std
        return std(self, dim=axis)

    def var(self, axis: Optional[int] = None) -> "Tensor":
        from .signal import var
        return var(self, dim=axis)

    def sum(self, axis: Optional[int] = None) -> "Tensor":
        return sum(self, dim=axis)

    def max(self, axis: Optional[int] = None) -> "Tensor":
        return max(self, dim=axis)

    def min(self, axis: Optional[int] = None) -> "Tensor":
        return min(self, dim=axis)

    def argmax(self, axis: Optional[int] = None) -> "Tensor":
        return argmax(self, dim=axis)

    def argmin(self, axis: Optional[int] = None) -> "Tensor":
        return argmin(self, dim=axis)

    def mean(self, axis: Optional[int] = None) -> "Tensor":
        from .signal import mean
        return mean(self, dim=axis)

    def astype(self, dtype) -> "Tensor":
        """numpy-style dtype conversion (see `sar.cast`)."""
        return cast(self, dtype)

    def __getitem__(self, key) -> "Tensor":
        """Basic slicing with unit steps: `x[2:6]`, `x[:, 4:8]`, ..."""
        if not isinstance(key, tuple):
            key = (key, )
        if len(key) > self.rank:
            raise TraceError(f"too many indices for a rank-{self.rank} tensor")
        key = key + (slice(None), ) * (self.rank - len(key))
        offsets, sizes, strides = [], [], []
        for d, (item, size) in enumerate(zip(key, self.shape)):
            if not isinstance(item, slice):
                raise TraceError(
                    "sar tensors support slice indexing only (integer "
                    "indexing would drop a dimension; slice a length-1 "
                    "range instead)")
            start, stop, step = item.indices(size)
            if step < 1:
                raise TraceError(
                    "sar tensor slices require a positive step (use "
                    "sar.flip to reverse)")
            if stop <= start:
                raise TraceError(f"empty slice along dim {d}")
            offsets.append(start)
            sizes.append((stop - start + step - 1) // step)
            strides.append(step)
        if tuple(sizes) == self.shape:
            return self
        return self._emit("slice", [self], TensorType(tuple(sizes),
                                                      self.dtype), {
                                                          "offsets": offsets,
                                                          "sizes": sizes,
                                                          "strides": strides
                                                      })

    @property
    def T(self) -> "Tensor":
        return transpose(self)


# --------------------------------------------------------------------------- #
# Free functions
# --------------------------------------------------------------------------- #


def _require_tensor(x, what: str) -> Tensor:
    if isinstance(x, np.ndarray):
        return constant(x)  # numpy arrays lift to constants
    if not isinstance(x, Tensor):
        raise TraceError(f"{what} expects a SAR tensor, got {type(x)!r}")
    return x


def _resolve_axis(what: str,
                  dim: Optional[int],
                  axis: Optional[int],
                  required: bool = True) -> Optional[int]:
    """`axis=` is the numpy-style alias of `dim=`; exactly one (if any)."""
    if dim is not None and axis is not None:
        raise TraceError(f"{what}: pass either dim or axis, not both")
    value = dim if dim is not None else axis
    if value is None and required:
        raise TraceError(f"{what} requires a dim (or axis) argument")
    return value


def constant(value,
             dtype: Optional[_DTypeSpec] = None,
             shape: Optional[Sequence[int]] = None) -> Tensor:
    """Materializes a constant tensor from a numpy array or a scalar."""
    if isinstance(value, (int, float, complex, np.generic)):
        if shape is None:
            raise TraceError("scalar constants require an explicit shape")
        if dtype is None:
            if isinstance(value, np.generic):
                np_name = value.dtype.name
                matches = [d for d in DTYPES.values() if d.np_dtype == np_name]
                if not matches:
                    raise TraceError(
                        f"unsupported constant dtype {np_name}; supported: "
                        f"{sorted(d.np_dtype for d in DTYPES.values())}")
                dtype = _SPEC_BY_DTYPE[matches[0]]
            else:
                dtype = c64 if isinstance(value, complex) else f64
        array = np.full(tuple(shape), value, dtype=dtype.dtype.to_numpy())
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
    value_ = fn.emit("sar.constant", [],
                     ttype, {"value": attr},
                     location=_source_location())
    return Tensor(value_, from_host=True)


def _float_unary(op: str, x: Tensor) -> Tensor:
    x = _require_tensor(x, f"sar.{op}")
    if not x.dtype.is_float:
        raise TraceError(f"sar.{op} expects a float tensor")
    return x._emit(op, [x], x._value.type)


def sqrt(x: Tensor) -> Tensor:
    """Element-wise square root of a float tensor."""
    return _float_unary("sqrt", x)


def cos(x: Tensor) -> Tensor:
    """Element-wise cosine of a float tensor (radians)."""
    return _float_unary("cos", x)


def sin(x: Tensor) -> Tensor:
    """Element-wise sine of a float tensor (radians)."""
    return _float_unary("sin", x)


def exp(x: Tensor) -> Tensor:
    """Element-wise natural exponential of a float tensor."""
    return _float_unary("exp", x)


def log(x: Tensor) -> Tensor:
    """Element-wise natural logarithm of a float tensor."""
    return _float_unary("log", x)


def atan2(y: Tensor, x: Tensor) -> Tensor:
    """atan2(y, x) element-wise (numpy `arctan2` argument order)."""
    y = _require_tensor(y, "sar.atan2")
    x = _require_tensor(x, "sar.atan2")
    if not y.dtype.is_float or not x.dtype.is_float:
        raise TraceError("sar.atan2 expects float tensors")
    if y._value.type != x._value.type:
        raise TraceError("sar.atan2 operand types must match")
    return y._emit("atan2", [y, x], y._value.type)


def conj(x: Tensor) -> Tensor:
    """Complex conjugate element-wise."""
    x = _require_tensor(x, "sar.conj")
    if not x.dtype.is_complex:
        raise TraceError("sar.conj expects a complex tensor")
    return x._emit("conj", [x], x._value.type)


def _complex_to_float(op: str, x: Tensor) -> Tensor:
    x = _require_tensor(x, f"sar.{op}")
    if not x.dtype.is_complex:
        raise TraceError(f"sar.{op} expects a complex tensor")
    out_dtype = FLOAT_PRECISION_OF[x.dtype]
    return x._emit(op, [x], TensorType(x.shape, out_dtype))


#: numpy spelling (`np.conjugate`) of the same operation.
conjugate = conj


def real(x: Tensor) -> Tensor:
    """Real part as a float tensor."""
    return _complex_to_float("real", x)


def imag(x: Tensor) -> Tensor:
    """Imaginary part as a float tensor."""
    return _complex_to_float("imag", x)


def angle(x: Tensor) -> Tensor:
    """Phase angle atan2(imag, real) as a float tensor (numpy `angle`)."""
    x = _require_tensor(x, "sar.angle")
    if not x.dtype.is_complex:
        raise TraceError("sar.angle expects a complex tensor")
    return atan2(imag(x), real(x))


def make_complex(re: Tensor, im: Tensor) -> Tensor:
    """Assembles a complex tensor from float real/imaginary parts (the
    inverse of `sar.real` / `sar.imag`; cf. `torch.complex`)."""
    re = _require_tensor(re, "sar.complex")
    im = _require_tensor(im, "sar.complex")
    if not re.dtype.is_float:
        raise TraceError("sar.complex expects float tensors")
    if re._value.type != im._value.type:
        raise TraceError("sar.complex plane types must match")
    return re._emit("complex", [re, im],
                    TensorType(re.shape, COMPLEX_OF[re.dtype]))


def absolute(x: Tensor) -> Tensor:
    """|x| element-wise; complex magnitude for complex tensors."""
    x = _require_tensor(x, "sar.abs")
    if x.dtype.is_int:
        raise TraceError("sar.abs does not support integer tensors")
    out_dtype = FLOAT_PRECISION_OF[x.dtype]
    return x._emit("abs", [x], TensorType(x.shape, out_dtype))


def expj(x: Tensor) -> Tensor:
    """exp(j*x) for a float tensor; returns a complex tensor (composes
    to cos + j sin)."""
    x = _require_tensor(x, "sar.expj")
    if not x.dtype.is_float:
        raise TraceError("sar.expj expects a float tensor")
    return make_complex(cos(x), sin(x))


def cast(x: Tensor, dtype: _DTypeSpec) -> Tensor:
    """Casts the element type. float<->float, complex<->complex,
    float->complex, int<->int, int->float, and float->int (truncation
    toward zero, C semantics)."""
    x = _require_tensor(x, "sar.cast")
    target = dtype.dtype
    if x.dtype == target:
        return x
    if x.dtype.is_complex and not target.is_complex:
        raise TraceError(
            "cannot cast complex to a real dtype; use sar.absolute, "
            "sar.real or sar.imag")
    if x.dtype.is_int and target.is_complex:
        raise TraceError(
            "cannot cast an integer tensor directly to complex; cast to "
            "float first")
    return x._emit("cast", [x], TensorType(x.shape, target))


def where(mask: Tensor, a, b) -> Tensor:
    """Element-wise selection (numpy `where`): `a` where `mask` is
    nonzero, `b` elsewhere. The mask comes from tensor comparisons
    (`x > 0.0`, `sar.absolute(z) >= t`, ...); selection is exact, not an
    arithmetic blend."""
    mask = _require_tensor(mask, "sar.where")
    if not mask.dtype.is_float:
        raise TraceError("sar.where expects a float mask")

    def splat(value, template):
        if isinstance(template, Tensor):
            np_dtype = template.dtype.to_numpy()
        elif isinstance(template, np.ndarray):
            np_dtype = template.dtype
        else:
            # Both branches are Python scalars, so there is no tensor to take
            # the precision from. Follow the mask, which carries the working
            # precision of the surrounding computation: inferring f64 from a
            # Python float would silently promote an f32 pipeline (and, on
            # HLS, synthesize double-precision hardware).
            np_dtype = mask.dtype.to_numpy()
        return np.full(mask.shape, value, dtype=np_dtype)

    if isinstance(a, (int, float)):
        a = splat(a, b)
    if isinstance(b, (int, float)):
        b = splat(b, a)
    a = _require_tensor(a, "sar.where")
    b = _require_tensor(b, "sar.where")
    a, b = _coerce_pair(a, b, "where")
    if a.dtype.is_int:
        raise TraceError(
            "sar.where does not support integer branch tensors; cast the "
            "branches to a float dtype first")
    if a.shape != mask.shape:
        raise TraceError(
            f"sar.where: mask shape {mask.shape} does not match the "
            f"branch shape {a.shape}")
    if FLOAT_PRECISION_OF[a.dtype] != mask.dtype:
        mask = cast(mask, _SPEC_BY_DTYPE[FLOAT_PRECISION_OF[a.dtype]])
    return a._emit("where", [mask, a, b], a._value.type)


#: numpy-style alias (the builtin `abs(x)` works too via `__abs__`).
abs = absolute  # noqa: A001


def maximum(x: Tensor, scalar: float) -> Tensor:
    """Element-wise maximum against a scalar (numpy `maximum`)."""
    x = _require_tensor(x, "sar.maximum")
    if not x.dtype.is_float:
        raise TraceError("sar.maximum expects a float tensor")
    selected = where(x > scalar, x, scalar)
    return where(x != x, x, selected)


def minimum(x: Tensor, scalar: float) -> Tensor:
    """Element-wise minimum against a scalar (numpy `minimum`)."""
    x = _require_tensor(x, "sar.minimum")
    if not x.dtype.is_float:
        raise TraceError("sar.minimum expects a float tensor")
    selected = where(x < scalar, x, scalar)
    return where(x != x, x, selected)


def clip(x: Tensor, lo: float, hi: float) -> Tensor:
    """Clamps into [lo, hi] (numpy `clip`)."""
    return minimum(maximum(x, lo), hi)


def _reduce_emit(x: Tensor, kind: str, dim: int) -> Tensor:
    out = TensorType((x.shape[1 - dim], ), x.dtype)
    return x._emit("reduce", [x], out, {"kind": kind, "dim": dim})


def _reduce(kind: str, x: Tensor, dim: Optional[int]) -> Tensor:
    x = _require_tensor(x, f"sar.{kind}")
    if kind != "sum" and not x.dtype.is_float:
        raise TraceError(f"sar.{kind} expects a float tensor")
    if x.dtype.is_int:
        raise TraceError(f"sar.{kind} does not support integer tensors")
    if x.rank == 1:
        if dim not in (None, 0):
            raise TraceError(f"sar.{kind}: dim {dim} out of range for rank 1")
        row = broadcast(x, (1, x.shape[0]), dim=1)
        return _reduce_emit(row, kind, dim=1)
    if x.rank != 2:
        raise TraceError(f"sar.{kind} expects a rank-1 or rank-2 tensor")
    if dim is None:
        return _reduce(kind, _reduce_emit(x, kind, dim=1), dim=None)
    if dim not in (0, 1):
        raise TraceError(f"sar.{kind}: dim must be 0 or 1 for rank 2")
    return _reduce_emit(x, kind, dim)


def sum(
        x: Tensor,
        dim: Optional[int] = None,  # noqa: A001
        axis: Optional[int] = None) -> Tensor:
    """Sum reduction (numpy `sum`). With `dim`/`axis` the axis is reduced
    to a rank-1 result; without it the full reduction has shape `(1,)`."""
    return _reduce("sum", x, _resolve_axis("sum", dim, axis, required=False))


def max(
        x: Tensor,
        dim: Optional[int] = None,  # noqa: A001
        axis: Optional[int] = None) -> Tensor:
    """Maximum reduction (numpy `max`); float tensors only."""
    return _reduce("max", x, _resolve_axis("max", dim, axis, required=False))


def min(
        x: Tensor,
        dim: Optional[int] = None,  # noqa: A001
        axis: Optional[int] = None) -> Tensor:
    """Minimum reduction (numpy `min`); float tensors only."""
    return _reduce("min", x, _resolve_axis("min", dim, axis, required=False))


def argmax(x: Tensor,
           dim: Optional[int] = None,
           axis: Optional[int] = None) -> Tensor:
    """Index of the maximum (numpy `argmax`: first occurrence on ties).

    Rank-1 tensors reduce to a `(1,)` i64 tensor; rank-2 tensors require
    an explicit `dim`/`axis` and return the rank-1 indices along it.
    """
    dim = _resolve_axis("argmax", dim, axis, required=False)
    x = _require_tensor(x, "sar.argmax")
    if not x.dtype.is_float:
        raise TraceError("sar.argmax expects a float tensor")
    if x.rank == 1:
        if dim not in (None, 0):
            raise TraceError(f"sar.argmax: dim {dim} out of range for rank 1")
        x = broadcast(x, (1, x.shape[0]), dim=1)
        dim = 1
    elif x.rank != 2:
        raise TraceError("sar.argmax expects a rank-1 or rank-2 tensor")
    elif dim is None:
        raise TraceError("sar.argmax on a rank-2 tensor requires dim")
    out = TensorType((x.shape[1 - dim], ), I64)
    return x._emit("argmax", [x], out, {"dim": dim})


def argmin(x: Tensor,
           dim: Optional[int] = None,
           axis: Optional[int] = None) -> Tensor:
    """Index of the minimum (numpy `argmin`: first occurrence on ties);
    composes to `argmax(-x)`."""
    x = _require_tensor(x, "sar.argmin")
    if not x.dtype.is_float:
        raise TraceError("sar.argmin expects a float tensor")
    return argmax(-x, dim=dim, axis=axis)


def sign(x: Tensor) -> Tensor:
    """numpy `sign`: -1, 0 or +1 per element."""
    x = _require_tensor(x, "sar.sign")
    if not x.dtype.is_float:
        raise TraceError("sar.sign expects a float tensor")
    return where(x > 0.0, 1.0, where(x < 0.0, -1.0, 0.0))


def _trunc(x: Tensor) -> Tensor:
    """Truncation toward zero via an i64 round trip.

    Limitation: values must fit in i64 (|x| < 2^63). Inputs outside that
    range -- including inf and NaN -- produce undefined results rather
    than an error, matching C float-to-int semantics. SAR position and
    index computations are always far below this bound.
    """
    return cast(cast(x, i64), _SPEC_BY_DTYPE[x.dtype])


def floor(x: Tensor) -> Tensor:
    """numpy `floor` as a composition.

    Values must fit in i64; inf/NaN inputs are undefined (see `_trunc`).
    """
    x = _require_tensor(x, "sar.floor")
    if not x.dtype.is_float:
        raise TraceError("sar.floor expects a float tensor")
    t = _trunc(x)
    return where(x < t, t - 1.0, t)


def ceil(x: Tensor) -> Tensor:
    """numpy `ceil` as a composition.

    Values must fit in i64; inf/NaN inputs are undefined (see `_trunc`).
    """
    x = _require_tensor(x, "sar.ceil")
    if not x.dtype.is_float:
        raise TraceError("sar.ceil expects a float tensor")
    t = _trunc(x)
    return where(x > t, t + 1.0, t)


def round(x: Tensor) -> Tensor:  # noqa: A001 - numpy-style rounding
    """Rounds half away from zero (Matlab `round`; note numpy rounds
    half to even).

    Values must fit in i64; inf/NaN inputs are undefined (see `_trunc`).
    """
    x = _require_tensor(x, "sar.round")
    if not x.dtype.is_float:
        raise TraceError("sar.round expects a float tensor")
    return sign(x) * floor(absolute(x) + 0.5)


def concatenate(tensors: Sequence[Tensor],
                dim: Optional[int] = None,
                axis: Optional[int] = None) -> Tensor:
    """Concatenates tensors along `dim` (numpy `concatenate`)."""
    dim = _resolve_axis("concatenate", dim, axis, required=False)
    dim = 0 if dim is None else dim
    if len(tensors) < 2:
        raise TraceError("sar.concatenate expects at least two tensors")
    tensors = [_require_tensor(t, "sar.concat") for t in tensors]
    first = tensors[0]
    if not 0 <= dim < first.rank:
        raise TraceError(f"concatenate dim {dim} out of range for rank "
                         f"{first.rank}")
    result = first
    for other in tensors[1:]:
        if other.dtype != result.dtype or other.rank != result.rank:
            raise TraceError(
                "sar.concatenate operands must share dtype and rank")
        shape = list(result.shape)
        shape[dim] += other.shape[dim]
        for d in range(result.rank):
            if d != dim and other.shape[d] != result.shape[d]:
                raise TraceError(
                    "sar.concatenate shapes must match outside dim")
        result = result._emit("concat", [result, other],
                              TensorType(tuple(shape), result.dtype),
                              {"dim": dim})
    return result


#: numpy 2.0 spelling of the same operation (Array API standard), and the
#: name the IR op carries.
concat = concatenate


def pad(x: Tensor, pad_width, value: float = 0.0) -> Tensor:
    """Pads with a constant; `pad_width` follows numpy `pad`:
    ((low0, high0), ...) per axis, or a single (low, high) for all axes."""
    x = _require_tensor(x, "sar.pad")
    if x.dtype.is_int:
        raise TraceError("sar.pad does not support integer tensors")
    fmt = ("pad_width takes (low, high) per axis -- ((l0, h0), ...) for "
           f"all {x.rank} dims, or a single (low, high) applied to each")
    if isinstance(pad_width, (int, np.integer)):
        raise TraceError(f"sar.pad: {fmt}")
    try:
        widths = list(pad_width)
    except TypeError:
        raise TraceError(f"sar.pad: {fmt}") from None
    if widths and isinstance(widths[0], (int, np.integer)):
        widths = [tuple(widths)] * x.rank
    if len(widths) != x.rank:
        raise TraceError(f"sar.pad: {fmt}")
    try:
        low = [int(lo) for lo, _ in widths]
        high = [int(hi) for _, hi in widths]
    except (TypeError, ValueError):
        raise TraceError(f"sar.pad: {fmt}") from None
    if any(v < 0 for v in low + high):
        raise TraceError("padding amounts must be non-negative")
    shape = tuple(s + lo + hi for s, lo, hi in zip(x.shape, low, high))
    return x._emit("pad", [x], TensorType(shape, x.dtype), {
        "low": low,
        "high": high,
        "value": float(value)
    })


def transpose(x: Tensor) -> Tensor:
    """Corner turn of a rank-2 tensor (also available as `x.T`)."""
    x = _require_tensor(x, "sar.transpose")
    if x.rank != 2:
        raise TraceError("sar.transpose expects a rank-2 tensor")
    out = TensorType((x.shape[1], x.shape[0]), x.dtype)
    return x._emit("transpose", [x], out)


def broadcast(x: Tensor,
              shape: Sequence[int],
              dim: Optional[int] = None,
              axis: Optional[int] = None) -> Tensor:
    """Broadcasts a 1-D tensor to a 2-D shape; the vector lies along `dim`."""
    dim = _resolve_axis("broadcast", dim, axis)
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
    return x._emit("broadcast", [x], TensorType(shape, x.dtype), {"dim": dim})


def _fftshift(x: Tensor, dim: int, inverse: bool) -> Tensor:
    x = _require_tensor(x, "sar.fftshift")
    if not 0 <= dim < x.rank:
        raise TraceError(f"fftshift dim {dim} out of range for rank {x.rank}")
    return x._emit("fftshift", [x], x._value.type, {"dim": dim},
                   ("inverse", ) if inverse else ())


def fftshift(x: Tensor,
             dim: Optional[int] = None,
             axis: Optional[int] = None) -> Tensor:
    """numpy/Matlab `fftshift`; without `dim`/`axis` every axis shifts."""
    dim = _resolve_axis("fftshift", dim, axis, required=False)
    if dim is not None:
        return _fftshift(x, dim, inverse=False)
    x = _require_tensor(x, "sar.fftshift")
    for d in range(x.rank):
        x = _fftshift(x, d, inverse=False)
    return x


def ifftshift(x: Tensor,
              dim: Optional[int] = None,
              axis: Optional[int] = None) -> Tensor:
    """numpy/Matlab `ifftshift`; without `dim`/`axis` every axis shifts."""
    dim = _resolve_axis("ifftshift", dim, axis, required=False)
    if dim is not None:
        return _fftshift(x, dim, inverse=True)
    x = _require_tensor(x, "sar.ifftshift")
    for d in range(x.rank):
        x = _fftshift(x, d, inverse=True)
    return x


def flip(x: Tensor,
         dim: Optional[int] = None,
         axis: Optional[int] = None) -> Tensor:
    """Reverses the element order along one axis (numpy/Matlab `flip`)."""
    dim = _resolve_axis("flip", dim, axis)
    x = _require_tensor(x, "sar.flip")
    if not 0 <= dim < x.rank:
        raise TraceError(f"flip dim {dim} out of range for rank {x.rank}")
    return x._emit("reverse", [x], x._value.type, {"dim": dim})


def _fft(op: str, x: Tensor, dim: int) -> Tensor:
    x = _require_tensor(x, f"sar.{op}")
    if not x.dtype.is_complex:
        x = cast(x, c128 if x.dtype == F64 else c64)  # numpy fft promotes
    if not 0 <= dim < x.rank:
        raise TraceError(f"{op} dim {dim} out of range for rank {x.rank}")
    if x.shape[dim] < 2:
        raise TraceError(f"sar.{op} size along dim must be at least 2")
    return x._emit(op, [x], x._value.type, {"dim": dim})


def _fft_norm(out: Tensor, dim: int, norm: Optional[str],
              inverse: bool) -> Tensor:
    """numpy `norm=` conventions on top of the unscaled-forward / 1/N-
    inverse primitives; the scale multiply fuses with neighbours."""
    if norm in (None, "backward"):
        return out
    n = out.shape[dim]
    if norm == "ortho":
        return out * (math.sqrt(n) if inverse else 1.0 / math.sqrt(n))
    if norm == "forward":
        return out * (float(n) if inverse else 1.0 / n)
    raise TraceError(
        f"fft norm must be one of None, 'backward', 'ortho', 'forward'; "
        f"got {norm!r}")


def fft(x: Tensor,
        dim: Optional[int] = None,
        axis: Optional[int] = None,
        norm: Optional[str] = None) -> Tensor:
    """Forward DFT along `dim`/`axis` (numpy/Matlab convention:
    unscaled); `norm` selects "backward" (default), "ortho" or
    "forward". Any size >= 2 on both backends (non-powers of two go
    through Bluestein's chirp-z reduction on the HLS path)."""
    dim = _resolve_axis("fft", dim, axis)
    return _fft_norm(_fft("fft", x, dim), dim, norm, inverse=False)


def ifft(x: Tensor,
         dim: Optional[int] = None,
         axis: Optional[int] = None,
         norm: Optional[str] = None) -> Tensor:
    """Inverse DFT along `dim`/`axis`, scaled by `1/N` (numpy/Matlab
    convention); `norm` as in `sar.fft`."""
    dim = _resolve_axis("ifft", dim, axis)
    return _fft_norm(_fft("ifft", x, dim), dim, norm, inverse=True)


_INTERP_KERNELS = ("nearest", "linear", "cubic", "sinc")
_INTERP_WINDOWS = ("rect", "hann", "hamming", "kaiser")


def interp1d(data: Tensor,
             positions: Tensor,
             dim: Optional[int] = None,
             axis: Optional[int] = None,
             kernel: str = "sinc",
             taps: int = 8,
             window: str = "hann",
             beta: float = 2.5,
             boundary: str = "zero") -> Tensor:
    """Resamples `data` along `dim`/`axis` (default 1) at fractional
    sample `positions` (both rank-2, same shape). The orthogonal
    primitive behind Stolt remapping and RCMC.

    `kernel` selects the interpolator: `nearest`, `linear`, `cubic`
    (Keys convolution) or `sinc` (default). For `sinc`, `taps` sets the
    support width and `window` tapers it (`rect`, `hann`, `hamming` or
    `kaiser` with shape `beta`).

    `boundary` controls out-of-range taps: `zero` (default: contribute zero),
    `edge` (clamp to the nearest sample), or `reflect` (mirror about
    the boundary, repeating the edge sample -- numpy `symmetric`)."""
    dim = _resolve_axis("interp1d", dim, axis, required=False)
    dim = 1 if dim is None else dim
    data = _require_tensor(data, "sar.interp1d")
    positions = _require_tensor(positions, "sar.interp1d")
    if data.rank != 2 or not data.dtype.is_complex:
        raise TraceError("interp1d expects rank-2 complex data")
    if positions.dtype != F64 or positions.shape != data.shape:
        raise TraceError(
            "interp1d positions must be an f64 tensor with the data shape")
    if dim not in (0, 1):
        raise TraceError("interp1d dim must be 0 or 1")
    if kernel not in _INTERP_KERNELS:
        raise TraceError(f"interp1d kernel must be one of {_INTERP_KERNELS}, "
                         f"got {kernel!r}")
    taps = int(taps)
    if kernel == "sinc":
        if taps % 2 or not 4 <= taps <= 32:
            raise TraceError("interp1d taps must be even and in [4, 32]")
        if window not in _INTERP_WINDOWS:
            raise TraceError(
                f"interp1d window must be one of {_INTERP_WINDOWS}, "
                f"got {window!r}")
        if window == "kaiser" and not 0.0 < float(beta) <= 12.0:
            raise TraceError("interp1d beta must be in (0, 12]")
    if boundary not in ("zero", "edge", "reflect"):
        raise TraceError(
            f"interp1d boundary must be one of ('zero', 'edge', 'reflect'), "
            f"got {boundary!r}")
    return data._emit(
        "interp1d", [data, positions], data._value.type, {
            "dim": dim,
            "kernel": kernel,
            "taps": taps,
            "window": window,
            "beta": float(beta),
            "boundary": boundary
        })


def gather2d(data: Tensor,
             rows: Tensor,
             cols: Tensor,
             kernel: str = "linear",
             boundary: str = "zero") -> Tensor:
    """Samples `data` at fractional 2-D positions:
    `out[i, j] = data[rows[i, j], cols[i, j]]`.

    Both coordinates may be arbitrary functions of the output position,
    which is the access pattern of time-domain backprojection; per-line
    resampling is `sar.interp1d`. The output takes the shape of the
    position tensors (rank-2 f64, equal shapes), independent of the data
    shape.

    `kernel` is `nearest` or `linear` (bilinear); `boundary` resolves
    out-of-range taps: `zero` (contribute zero) or `edge` (clamp)."""
    data = _require_tensor(data, "sar.gather2d")
    rows = _require_tensor(rows, "sar.gather2d")
    cols = _require_tensor(cols, "sar.gather2d")
    if data.rank != 2 or not data.dtype.is_complex:
        raise TraceError("gather2d expects rank-2 complex data")
    for name, pos in (("rows", rows), ("cols", cols)):
        if pos.rank != 2 or pos.dtype != F64:
            raise TraceError(f"gather2d {name} must be a rank-2 f64 tensor")
    if rows.shape != cols.shape:
        raise TraceError("gather2d rows and cols must share a shape")
    if kernel not in ("nearest", "linear"):
        raise TraceError(
            f"gather2d kernel must be 'nearest' or 'linear', got {kernel!r}")
    if boundary not in ("zero", "edge"):
        raise TraceError(
            f"gather2d boundary must be 'zero' or 'edge', got {boundary!r}")
    out = TensorType(rows.shape, data.dtype)
    return data._emit("gather2d", [data, rows, cols], out, {
        "kernel": kernel,
        "boundary": boundary
    })


def cumsum(x: Tensor,
           dim: Optional[int] = None,
           axis: Optional[int] = None) -> Tensor:
    """Inclusive prefix sum along `dim`/`axis` (numpy `cumsum`).

    Rank-1 or rank-2 float or complex tensors. Unlike numpy, an omitted
    axis does not flatten -- it defaults to the last one.
    """
    dim = _resolve_axis("cumsum", dim, axis, required=False)
    x = _require_tensor(x, "sar.cumsum")
    if x.dtype.is_int:
        raise TraceError("sar.cumsum does not support integer tensors")
    if x.rank == 1:
        if dim not in (None, 0):
            raise TraceError(f"sar.cumsum: dim {dim} out of range for rank 1")
        return x._emit("cumsum", [x], x._value.type, {"dim": 0})
    if x.rank != 2:
        raise TraceError("sar.cumsum expects a rank-1 or rank-2 tensor")
    dim = 1 if dim is None else dim
    if dim not in (0, 1):
        raise TraceError("sar.cumsum: dim must be 0 or 1 for rank 2")
    return x._emit("cumsum", [x], x._value.type, {"dim": dim})


def rank_filter(x: Tensor,
                window: int,
                rank: int,
                dim: Optional[int] = None,
                axis: Optional[int] = None) -> Tensor:
    """Windowed order-statistic filter along `dim`/`axis`
    (scipy `ndimage.rank_filter` restricted to one axis).

    Each output element is the `rank`-th smallest (0-based) of the
    `window` samples centred on it; edges replicate the boundary sample.
    Float tensors only -- ordering is undefined for complex.
    """
    dim = _resolve_axis("rank_filter", dim, axis, required=False)
    x = _require_tensor(x, "sar.rank_filter")
    if not x.dtype.is_float:
        raise TraceError("sar.rank_filter expects a float tensor")
    window = int(window)
    rank = int(rank)
    if window < 1 or window % 2 == 0:
        raise TraceError("sar.rank_filter window must be a positive odd "
                         f"integer, got {window}")
    if not 0 <= rank < window:
        raise TraceError(
            f"sar.rank_filter rank must be in [0, {window}), got {rank}")
    if x.rank == 1:
        if dim not in (None, 0):
            raise TraceError(
                f"sar.rank_filter: dim {dim} out of range for rank 1")
        return x._emit("rank_filter", [x], x._value.type, {
            "window": window,
            "rank": rank,
            "dim": 0
        })
    if x.rank != 2:
        raise TraceError("sar.rank_filter expects a rank-1 or rank-2 tensor")
    dim = 1 if dim is None else dim
    if dim not in (0, 1):
        raise TraceError("sar.rank_filter: dim must be 0 or 1 for rank 2")
    return x._emit("rank_filter", [x], x._value.type, {
        "window": window,
        "rank": rank,
        "dim": dim
    })


def median_filter(x: Tensor,
                  window: int,
                  dim: Optional[int] = None,
                  axis: Optional[int] = None) -> Tensor:
    """Running median along `dim`/`axis` (scipy `ndimage.median_filter`
    restricted to one axis): `rank_filter` at rank `window // 2`."""
    return rank_filter(x, window, int(window) // 2, dim=dim, axis=axis)


def sort(x: Tensor,
         dim: Optional[int] = None,
         axis: Optional[int] = None) -> Tensor:
    """Sorts each line along `dim`/`axis` into ascending order
    (numpy `sort`).

    Rank-1 or rank-2 float tensors only -- ordering is undefined for
    complex.
    """
    dim = _resolve_axis("sort", dim, axis, required=False)
    x = _require_tensor(x, "sar.sort")
    if not x.dtype.is_float:
        raise TraceError("sar.sort expects a float tensor")
    if x.rank == 1:
        if dim not in (None, 0):
            raise TraceError(f"sar.sort: dim {dim} out of range for rank 1")
        return x._emit("sort", [x], x._value.type, {"dim": 0})
    if x.rank != 2:
        raise TraceError("sar.sort expects a rank-1 or rank-2 tensor")
    dim = 1 if dim is None else dim
    if dim not in (0, 1):
        raise TraceError("sar.sort: dim must be 0 or 1 for rank 2")
    return x._emit("sort", [x], x._value.type, {"dim": dim})


def iterate(trips: int, body: Callable, *carries: Tensor, index: bool = False):
    """Compiled counted loop with tensor-carried state: applies `body`
    `trips` times, feeding each iteration's results to the next.

    ``iterate(8, step, x0)`` compiles to a single loop in the design --
    unlike a Python ``for``, which unrolls the body into the IR once per
    iteration at trace time. `body` receives one tensor per carry and
    must return as many, with matching types; more than one carry
    returns a tuple::

        smoothed = sar.iterate(steps, lambda x: x - mu * grad(x), x0)

    With ``index=True`` the body receives the iteration index first, as
    an ``i64[1]`` tensor (0-based, not a carry), so per-iteration terms
    can be computed in the kernel::

        out = sar.iterate(n, lambda i, acc: acc * phase(i), x0, index=True)

    The trip count is a compile-time constant.
    """
    fn = _current_function()
    if isinstance(trips, bool) or not isinstance(trips, (int, np.integer)):
        raise TraceError("sar.iterate trips must be a positive integer")
    trips = int(trips)
    if trips < 1:
        raise TraceError("sar.iterate trips must be a positive integer")
    carried = tuple(_require_tensor(c, "sar.iterate") for c in carries)
    if not carried:
        raise TraceError("sar.iterate needs at least one carried tensor")

    arg_types = [c._value.type for c in carried]
    if index:
        arg_types = [TensorType((1, ), I64)] + arg_types
    args = fn.push_region(arg_types)
    try:
        returned = body(*[Tensor(a) for a in args])
        results = returned if isinstance(returned, tuple) else (returned, )
        if len(results) != len(carried):
            raise TraceError(
                f"sar.iterate body returned {len(results)} value(s) for "
                f"{len(carried)} carried tensor(s)")
        for i, (out, carry) in enumerate(zip(results, carried)):
            if not isinstance(out, Tensor):
                raise TraceError(
                    f"sar.iterate body result #{i} is not a SAR tensor")
            if out._value.type != carry._value.type:
                raise TraceError(
                    f"sar.iterate body result #{i} has type "
                    f"{out._value.type.mlir} but the carry it feeds is "
                    f"{carry._value.type.mlir}")
    except BaseException:
        fn.abort_region()
        raise
    region = fn.pop_region([r._value for r in results])
    values = fn.emit_region_op("sar.iterate", [c._value for c in carried],
                               [c._value.type for c in carried], region,
                               {"trips": trips}, ("index", ) if index else (),
                               _source_location())
    tensors = tuple(Tensor(v) for v in values)
    return tensors[0] if len(tensors) == 1 else tensors


# --------------------------------------------------------------------------- #
# User-defined operators
# --------------------------------------------------------------------------- #


def _tracing() -> bool:
    """True when inside a kernel trace, False when running eagerly."""
    return getattr(_state, "function", None) is not None


def _freeze_for_key(value):
    """A hashable stand-in for a compile-time keyword argument.

    Keyword arguments are baked into the trace, so they belong in the
    specialization key. Most are scalars or strings; numpy arrays (Stolt
    frequency axes, replicas) are not hashable, so they are summarized by
    their bytes.
    """
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, str(value.dtype), value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_key(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, _freeze_for_key(v)) for k, v in sorted(value.items()))
    return value


def op(fn=None, *, const=()):
    """Defines an operator as a composition of existing ops.

    The body inlines into kernels at trace time like any built-in (so it
    fuses with its surroundings and compiles to every backend) and runs
    eagerly on numpy arrays outside kernels (specialized and JIT-compiled
    per argument signature)::

        @sar.op
        def range_compress(data, replica):
            return sar.ifft(sar.fft(data, axis=1) * replica, axis=1)

        range_compress(raw_np, replica_np)   # eager, numpy in / numpy out
        @sar.func
        def chain(raw):                      # inlined, fuses with neighbours
            return abs(range_compress(raw, replica_np))

    Outside a kernel the arguments split two ways: numpy arrays become the
    specialized kernel's parameters, everything else (axes, tap counts,
    flags) is a compile-time constant baked into the trace, so each
    distinct combination compiles its own variant. An operator with no
    array arguments at all is a *constructor* (`sar.hanning(512)`): there
    is nothing to specialize on, so it runs directly and returns its numpy
    samples, which stay usable in host code.

    `const` names array parameters that are *metadata* rather than data --
    acquisition axes whose values the body inspects with host arithmetic
    (`fr[1] - fr[0]`). They bake in as constants instead of becoming
    kernel parameters::

        @sar.op(const=("fa", "fr"))
        def stolt(data, fa, fr, *, vr): ...
    """
    if fn is None:  # used as @op(const=(...))
        return functools.partial(op, const=const)

    const_names = frozenset(const)
    signature = inspect.signature(fn)
    #: One GenericKernel per (tensor parameters, constant values).
    variants: Dict[tuple, GenericKernel] = {}
    variants_lock = threading.RLock()

    def eager(args, kwargs):
        bound = signature.bind(*args, **kwargs)
        # Defaults join the specialization key: without them a call spelling
        # a default explicitly would compile a second, identical variant.
        bound.apply_defaults()
        tensor_names, constants = [], {}
        for name, value in bound.arguments.items():
            if isinstance(value, np.ndarray) and name not in const_names:
                tensor_names.append(name)
            else:
                constants[name] = value

        if not tensor_names:
            # Nothing to specialize on: the operator is a *constructor*
            # (windows take a length, not a tensor). There is no kernel to
            # compile, so the body runs directly and hands back whatever it
            # builds -- plain numpy samples, usable in host code.
            return fn(*args, **kwargs)

        key = (tuple(tensor_names),
               tuple(
                   sorted(
                       (k, _freeze_for_key(v)) for k, v in constants.items())))
        with variants_lock:
            generic = variants.get(key)
            if generic is None:

                def bound_fn(*tensors,
                             _names=tuple(tensor_names),
                             _consts=dict(constants)):
                    call = dict(_consts)
                    call.update(zip(_names, tensors))
                    return fn(**call)

                bound_fn.__name__ = fn.__name__
                bound_fn.__doc__ = fn.__doc__
                generic = GenericKernel(bound_fn)
                variants[key] = generic
        return generic(*[bound.arguments[n] for n in tensor_names])

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if getattr(_state, "function", None) is not None:
            return fn(*args, **kwargs)  # inside a trace: plain inlining
        return eager(args, kwargs)

    #: Handle for emission backends:
    #: ``f.func.specialize(sar.c64[512, 512], ...)``.
    wrapper.func = GenericKernel(fn)
    wrapper.specialize = wrapper.func.specialize
    return wrapper


# --------------------------------------------------------------------------- #
# Kernel tracing
# --------------------------------------------------------------------------- #

#: MLIR bare identifiers, which is what a kernel name becomes (`@name`).
_MLIR_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_$.]*\Z")


def _validated_kernel_name(name: str) -> str:
    """Rejects names that cannot be emitted as an MLIR symbol.

    Without this the invalid name reaches `sar-opt` and surfaces as a parse
    error about the serialized IR ("@ identifier expected to start with
    letter"), which says nothing about the Python that caused it. Lambdas
    (`<lambda>`) and non-ASCII identifiers are the practical cases.
    """
    if not _MLIR_IDENTIFIER.match(name):
        raise TraceError(
            f"kernel name {name!r} is not a valid MLIR symbol; kernels need "
            "an ASCII identifier name -- define the kernel with `def` "
            "instead of a lambda, or rename it")
    return name


class Kernel:
    """A traced SAR kernel; compiles lazily per backend.

    `arg_types` may be passed directly (annotation-free kernels
    specialized from call-site arrays); result types are then inferred
    from the trace instead of checked against annotations."""

    def __init__(self,
                 fn: Callable,
                 arg_types: Optional[Sequence[TensorType]] = None):
        self._fn = fn
        self._lock = threading.RLock()
        self._name = _validated_kernel_name(fn.__name__)
        functools.update_wrapper(self, fn)
        if arg_types is None:
            self.arg_types, self.declared_result_types = \
                self._parse_signature(fn)
        else:
            self.arg_types = list(arg_types)
            self.declared_result_types: Optional[List[TensorType]] = None
        self.param_names = self._parse_param_names(fn, len(self.arg_types))
        self._module_text: Optional[str] = None
        self._compiled = {}

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        value = _validated_kernel_name(value)
        with self._lock:
            if value == self._name:
                return
            self._name = value
            self._module_text = None
            self._compiled.clear()

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
        # Shapes are expressions (`sar.c64[N, N]`), so evaluating the
        # string is the only way to resolve it; what can be improved is
        # the report when it fails.
        try:
            return eval(annotation, getattr(fn, "__globals__", {}), namespace)
        except Exception as exc:
            raise TraceError(
                f"kernel '{fn.__name__}': cannot evaluate the type "
                f"annotation {annotation!r}: {exc}") from exc

    @staticmethod
    def _parse_param_names(fn: Callable,
                           arg_count: int) -> Optional[List[str]]:
        """The kernel's parameter names, or None when the signature does
        not expose one per argument (e.g. a `*tensors` wrapper built by
        `@sar.op`). Emission backends use them to name design ports."""
        try:
            params = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):
            return None
        if len(params) != arg_count or any(
                p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                for p in params):
            return None
        return [p.name for p in params]

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
        rets = ret if isinstance(ret, tuple) else (ret, )
        for r in rets:
            if not isinstance(r, TensorType):
                raise TraceError(
                    f"kernel '{fn.__name__}': invalid return annotation {r!r}")
        return arg_types, list(rets)

    def trace(self) -> str:
        """Traces the Python function and returns the MLIR module text."""
        with self._lock:
            fn_ir = ir.Function(self.name, self.arg_types, self.param_names)
            prev = getattr(_state, "function", None)
            _state.function = fn_ir
            try:
                result = self._fn(*[Tensor(a) for a in fn_ir.arguments])
            finally:
                _state.function = prev

            results = result if isinstance(result, tuple) else (result, )
            for i, value in enumerate(results):
                if not isinstance(value, Tensor):
                    raise TraceError(
                        f"kernel '{self.name}' result #{i} is not a SAR tensor"
                    )
            if self.declared_result_types is None:
                self.declared_result_types = [v._value.type for v in results]
            else:
                if len(results) != len(self.declared_result_types):
                    raise TraceError(
                        f"kernel '{self.name}' returned {len(results)} values "
                        f"but declares {len(self.declared_result_types)}")
                for i, (value, declared) in enumerate(
                        zip(results, self.declared_result_types)):
                    if value._value.type != declared:
                        raise TraceError(
                            f"kernel '{self.name}' result #{i} has type "
                            f"{value._value.type.mlir} but declares "
                            f"{declared.mlir}")

            fn_ir.set_return([v._value for v in results])
            self._module_text = ir.Module([fn_ir]).render()
            self._compiled.clear()
            return self._module_text

    def to_mlir(self) -> str:
        """The kernel's MLIR module text, traced on first use and reused
        after that (the trace depends only on the function and the
        argument types, both fixed per kernel)."""
        with self._lock:
            if self._module_text is None:
                return self.trace()
            return self._module_text

    def compile(self, backend: str = "cpu", options: Optional[dict] = None):
        """Compiles the kernel for `backend` and returns the launcher.

        A repeated compile for the same backend and options reuses the
        launcher. Option sets that cannot serve as a memo key skip the
        memo and compile directly (the on-disk artifact cache still
        applies): a `config` entry points at a file whose contents can
        change between calls, an environment-named config file can change
        the same way, and an unhashable value cannot be looked up at all.
        """

        from ..compiler import compile as _compile
        key = None
        env_config = os.environ.get("SAR_DSL_HLS_CONFIG")
        if not (options or {}).get("config") and not env_config:
            try:
                key = (backend, tuple(sorted((options or {}).items())))
                hash(key)
            except TypeError:
                key = None
        with self._lock:
            if key is None:
                return _compile(self, backend=backend, options=options)
            if key not in self._compiled:
                self._compiled[key] = _compile(self,
                                               backend=backend,
                                               options=options)
            return self._compiled[key]

    def __call__(self, *arrays):
        """JIT-compiles for the CPU backend and executes."""
        return self.compile("cpu")(*arrays)


def _type_of_array(value) -> TensorType:
    if not isinstance(value, np.ndarray):
        raise TraceError(
            "annotation-free kernels specialize from numpy array arguments; "
            f"got {type(value)!r}")
    name = value.dtype.name
    for dtype in DTYPES.values():
        if dtype.np_dtype == name:
            return TensorType(tuple(value.shape), dtype)
    raise TraceError(f"unsupported array dtype {name}; supported dtypes: "
                     f"{sorted(d.np_dtype for d in DTYPES.values())}")


class GenericKernel:
    """An annotation-free kernel: each distinct argument signature
    (shapes and dtypes of the call-site arrays) compiles its own
    specialization, numba/Triton style."""

    def __init__(self, fn: Callable):
        self._fn = fn
        functools.update_wrapper(self, fn)
        self._variants: Dict[tuple, Kernel] = {}
        self._lock = threading.RLock()

    def specialize(self, *arg_types: TensorType) -> Kernel:
        """Returns the kernel instance for explicit argument types
        (`sar.c64[512, 512]`, ...); useful for emission backends."""
        key = tuple(arg_types)
        with self._lock:
            if key not in self._variants:
                self._variants[key] = Kernel(self._fn, arg_types=key)
            return self._variants[key]

    def __call__(self, *arrays):
        return self.specialize(*[_type_of_array(a) for a in arrays])(*arrays)

    def compile(self, backend: str = "cpu", options: Optional[dict] = None):
        raise TraceError(
            "annotation-free kernels specialize per argument signature: "
            "call them with arrays directly, or pin one with "
            ".specialize(sar.c64[512, 512], ...) before compiling")


def func(fn: Callable):
    """Decorator turning a Python function into a compiled SAR kernel.

    Type annotations (`x: sar.c64[512, 512]`) pin the signature; without
    them the kernel specializes from the arrays of each call."""
    sig = inspect.signature(fn)
    annotated = [
        p.annotation is not inspect.Parameter.empty
        for p in sig.parameters.values()
    ]
    if annotated and not any(annotated):
        return GenericKernel(fn)
    return Kernel(fn)


# The composed signal-processing vocabulary lives in a submodule (the
# DSL surface is richer than the IR). It is imported and re-exported by
# sar/__init__.py rather than here: signal.py spells the decorator
# `@sar.op` -- the same way user code does -- which needs `sar.op` to be
# bound before the submodule is executed.
