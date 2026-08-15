"""Lightweight textual MLIR builder for the `sar` dialect.

SAR-DSL deliberately emits MLIR as *text* rather than going through the MLIR
Python bindings: the surface between the Python frontend and the C++ core is
just the serialized module, which keeps the frontend importable without any
compiled component and decouples it from LLVM API churn. Correctness is still
enforced by the C++ dialect verifiers the moment `sar-opt` parses the module.

Operations are printed in MLIR's *generic* form (``"sar.add"(%0, %1) ...``),
which is stable regardless of custom assembly formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DType",
    "TensorType",
    "Value",
    "DenseAttr",
    "Operation",
    "Function",
    "Module",
]


@dataclass(frozen=True)
class DType:
    """Element type of a SAR tensor."""

    name: str  # short name used in the DSL, e.g. "c64"
    mlir: str  # MLIR element type, e.g. "complex<f32>"
    np_dtype: str  # numpy dtype name, e.g. "complex64"

    @property
    def is_complex(self) -> bool:
        return self.name.startswith("c")

    @property
    def is_float(self) -> bool:
        return self.name.startswith("f")

    @property
    def is_int(self) -> bool:
        return self.name.startswith("i")

    def to_numpy(self):
        return np.dtype(self.np_dtype)

    def __repr__(self) -> str:
        return f"sar.{self.name}"


# The closed set of element types supported by the dialect.
F32 = DType("f32", "f32", "float32")
F64 = DType("f64", "f64", "float64")
I32 = DType("i32", "i32", "int32")
I64 = DType("i64", "i64", "int64")
C64 = DType("c64", "complex<f32>", "complex64")
C128 = DType("c128", "complex<f64>", "complex128")

DTYPES: Dict[str, DType] = {d.name: d for d in (F32, F64, I32, I64, C64, C128)}

#: Maps a complex dtype to the float dtype of its components (and floats to
#: themselves). Used by e.g. `sar.abs`.
FLOAT_PRECISION_OF = {C64: F32, C128: F64, F32: F32, F64: F64}

#: Maps a float dtype to the complex dtype of the same precision.
COMPLEX_OF = {F32: C64, F64: C128}


@dataclass(frozen=True)
class TensorType:
    """A ranked tensor type with a static shape."""

    shape: Tuple[int, ...]
    dtype: DType

    def __post_init__(self):
        if len(self.shape) < 1:
            raise ValueError("SAR tensors must have rank >= 1")
        for dim in self.shape:
            if not isinstance(dim, (int, np.integer)) or dim <= 0:
                raise ValueError(
                    f"SAR tensor shapes must be static and positive, got "
                    f"{self.shape}")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def mlir(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        return f"tensor<{dims}x{self.dtype.mlir}>"

    def __repr__(self) -> str:
        return self.mlir


@dataclass(frozen=True)
class Value:
    """An SSA value inside a function under construction."""

    name: str  # e.g. "%arg0" or "%3"
    type: TensorType


def _format_float(value: float) -> str:
    """Formats a float so MLIR parses it back to the identical double."""
    return f"{float(value):.17e}"


def _format_attr(value) -> str:
    """Formats a python attribute value as an MLIR attribute."""
    if isinstance(value, bool):
        raise TypeError("use unit attributes for booleans")
    if isinstance(value, (int, np.integer)):
        return f"{int(value)} : i64"
    if isinstance(value, (float, np.floating)):
        return f"{_format_float(value)} : f64"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        elems = ", ".join(str(int(v)) for v in value)
        return f"array<i64: {elems}>"
    if isinstance(value, DenseAttr):
        return value.text
    raise TypeError(f"unsupported attribute value: {value!r}")


#: Above this element count, non-splat float/complex constants are emitted
#: as raw hex blobs (compact and exact; same layout numpy uses in memory).
_HEX_DENSE_THRESHOLD = 64


@dataclass(frozen=True)
class DenseAttr:
    """A pre-rendered dense elements attribute."""

    text: str

    @staticmethod
    def from_array(array: np.ndarray, type: TensorType) -> "DenseAttr":
        if tuple(array.shape) != type.shape:
            raise ValueError(
                f"constant shape {tuple(array.shape)} does not match type "
                f"{type.shape}")

        def render(x) -> str:
            if type.dtype.is_complex:
                return f"({_format_float(x.real)},{_format_float(x.imag)})"
            if type.dtype.is_float:
                return _format_float(x)
            return str(int(x))

        flat = array.reshape(-1)
        if flat.size > 0 and np.all(flat == flat[0]):
            body = render(flat[0])  # splat
        elif (flat.size > _HEX_DENSE_THRESHOLD
              and (type.dtype.is_float or type.dtype.is_complex)):
            # MLIR hex blobs store elements in little-endian byte order,
            # which matches numpy's contiguous layout exactly.
            data = np.ascontiguousarray(
                array.astype(type.dtype.to_numpy(), copy=False))
            body = '"0x' + data.tobytes().hex().upper() + '"'
        else:

            def nest(arr) -> str:
                if arr.ndim == 1:
                    return "[" + ", ".join(render(x) for x in arr) + "]"
                return "[" + ", ".join(nest(sub) for sub in arr) + "]"

            body = nest(array)
        return DenseAttr(f"dense<{body}> : {type.mlir}")


@dataclass
class Operation:
    """A single operation in generic MLIR form."""

    op_name: str  # e.g. "sar.add"
    operands: List[Value]
    result_types: List[object]
    attributes: Dict[str, object] = field(default_factory=dict)
    unit_attributes: Tuple[str, ...] = ()
    results: List[Value] = field(default_factory=list)

    def render(self) -> str:
        results = ", ".join(v.name for v in self.results)
        operands = ", ".join(v.name for v in self.operands)

        attr_parts = [
            f"{k} = {_format_attr(v)}" for k, v in self.attributes.items()
        ]
        attr_parts += list(self.unit_attributes)
        props = f" <{{{', '.join(attr_parts)}}}>" if attr_parts else ""

        operand_types = ", ".join(v.type.mlir for v in self.operands)
        result_types = ", ".join(t.mlir for t in self.result_types)
        signature = f"({operand_types}) -> ({result_types})"

        prefix = f"{results} = " if results else ""
        return f'{prefix}"{self.op_name}"({operands}){props} : {signature}'


class Function:
    """A function under construction; owns SSA numbering."""

    def __init__(self, name: str, arg_types: Sequence[TensorType]):
        self.name = name
        self.arguments: List[Value] = [
            Value(f"%arg{i}", t) for i, t in enumerate(arg_types)
        ]
        self.operations: List[Operation] = []
        self.returned: Optional[List[Value]] = None
        self._next_id = 0

    def _new_value(self, type: TensorType) -> Value:
        value = Value(f"%{self._next_id}", type)
        self._next_id += 1
        return value

    def emit(
        self,
        op_name: str,
        operands: Sequence[Value],
        result_type: TensorType,
        attributes: Optional[Dict[str, object]] = None,
        unit_attributes: Sequence[str] = ()
    ) -> Value:
        """Appends an operation with a single result and returns its value."""
        op = Operation(op_name, list(operands), [result_type],
                       dict(attributes or {}), tuple(unit_attributes))
        op.results = [self._new_value(result_type)]
        self.operations.append(op)
        return op.results[0]

    def set_return(self, values: Sequence[Value]) -> None:
        self.returned = list(values)

    @property
    def result_types(self) -> List[TensorType]:
        assert self.returned is not None, "function has no return yet"
        return [v.type for v in self.returned]

    def render(self, indent: str = "  ") -> str:
        assert self.returned is not None, "function has no return yet"
        args = ", ".join(f"{v.name}: {v.type.mlir}" for v in self.arguments)
        rets = ", ".join(t.mlir for t in self.result_types)
        lines = [f"{indent}func.func @{self.name}({args}) -> ({rets}) {{"]
        for op in self.operations:
            lines.append(f"{indent}  {op.render()}")
        ret_vals = ", ".join(v.name for v in self.returned)
        ret_types = ", ".join(t.mlir for t in self.result_types)
        lines.append(f"{indent}  return {ret_vals} : {ret_types}")
        lines.append(f"{indent}}}")
        return "\n".join(lines)


class Module:
    """A top-level MLIR module holding one or more kernels."""

    def __init__(self, functions: Sequence[Function]):
        self.functions = list(functions)

    def render(self) -> str:
        body = "\n\n".join(f.render() for f in self.functions)
        return f"module {{\n{body}\n}}\n"
