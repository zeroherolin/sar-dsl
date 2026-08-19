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

import json
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
            if isinstance(
                    dim,
                    bool) or not isinstance(dim,
                                            (int, np.integer)) or dim <= 0:
                raise ValueError(
                    f"SAR tensor shapes must be static positive integers, "
                    f"got {self.shape}")

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
#: as raw hex blobs (compact and exact).
_HEX_DENSE_THRESHOLD = 64


@dataclass(frozen=True)
class DenseAttr:
    """A pre-rendered dense elements attribute."""

    text: str

    @staticmethod
    def _render(value, dtype: DType) -> str:
        if dtype.is_complex:
            return (f"({_format_float(value.real)},"
                    f"{_format_float(value.imag)})")
        if dtype.is_float:
            return _format_float(value)
        return str(int(value))

    @classmethod
    def splat(cls, value, type: TensorType) -> "DenseAttr":
        """Build a scalar dense attribute without allocating the tensor."""
        scalar = np.asarray(value, dtype=type.dtype.to_numpy()).item()
        return cls(f"dense<{cls._render(scalar, type.dtype)}> : {type.mlir}")

    @staticmethod
    def from_array(array: np.ndarray, type: TensorType) -> "DenseAttr":
        if tuple(array.shape) != type.shape:
            raise ValueError(
                f"constant shape {tuple(array.shape)} does not match type "
                f"{type.shape}")

        flat = array.reshape(-1)
        # Compare object representations, not numeric equality: +0.0 and
        # -0.0 are equal numerically but are observably different constants.
        bits = np.ascontiguousarray(flat).view(
            np.dtype((np.void, array.dtype.itemsize)))
        if flat.size > 0 and np.all(bits == bits[0]):
            body = DenseAttr._render(flat[0], type.dtype)
        elif (flat.size > _HEX_DENSE_THRESHOLD
              and (type.dtype.is_float or type.dtype.is_complex)):
            # MLIR hex blobs are little-endian regardless of the host.
            little_endian = np.dtype(type.dtype.to_numpy()).newbyteorder("<")
            data = np.ascontiguousarray(array.astype(little_endian,
                                                     copy=False))
            body = '"0x' + data.tobytes().hex().upper() + '"'
        else:

            def nest(arr) -> str:
                if arr.ndim == 1:
                    values = (DenseAttr._render(x, type.dtype) for x in arr)
                    return "[" + ", ".join(values) + "]"
                return "[" + ", ".join(nest(sub) for sub in arr) + "]"

            body = nest(array)
        return DenseAttr(f"dense<{body}> : {type.mlir}")


@dataclass
class Region:
    """A single-block region: block arguments, body, and the values the
    terminating `sar.yield` carries."""

    arguments: List[Value]
    operations: List[Operation] = field(default_factory=list)
    yields: List[Value] = field(default_factory=list)

    def render(self, indent: str) -> str:
        args = ", ".join(f"{v.name}: {v.type.mlir}" for v in self.arguments)
        lines = ["({", f"{indent}^bb0({args}):"]
        for op in self.operations:
            lines.append(f"{indent}  {op.render(indent + '  ')}")
        names = ", ".join(v.name for v in self.yields)
        types = ", ".join(v.type.mlir for v in self.yields)
        lines.append(f'{indent}  "sar.yield"({names}) : ({types}) -> ()')
        lines.append(f"{indent}}})")
        return "\n".join(lines)


@dataclass
class Operation:
    """A single operation in generic MLIR form."""

    op_name: str  # e.g. "sar.add"
    operands: List[Value]
    result_types: List[object]
    attributes: Dict[str, object] = field(default_factory=dict)
    unit_attributes: Tuple[str, ...] = ()
    results: List[Value] = field(default_factory=list)
    region: Optional[Region] = None
    location: Optional[Tuple[str, int, int]] = None

    def render(self, indent: str = "  ") -> str:
        # Several results print as one group (`%3:2 = ...`), which later
        # uses name per index (`%3#0`).
        if len(self.results) > 1:
            base = self.results[0].name.split("#")[0]
            prefix = f"{base}:{len(self.results)} = "
        elif self.results:
            prefix = f"{self.results[0].name} = "
        else:
            prefix = ""
        operands = ", ".join(v.name for v in self.operands)

        attr_parts = [
            f"{k} = {_format_attr(v)}" for k, v in self.attributes.items()
        ]
        attr_parts += list(self.unit_attributes)
        props = f" <{{{', '.join(attr_parts)}}}>" if attr_parts else ""
        region = f" {self.region.render(indent)}" if self.region else ""

        operand_types = ", ".join(v.type.mlir for v in self.operands)
        result_types = ", ".join(t.mlir for t in self.result_types)
        signature = f"({operand_types}) -> ({result_types})"
        loc = ""
        if self.location is not None:
            path, line, column = self.location
            loc = f" loc({json.dumps(path)}:{line}:{column})"

        return (f'{prefix}"{self.op_name}"({operands}){props}{region}'
                f" : {signature}{loc}")


class Function:
    """A function under construction; owns SSA numbering."""

    def __init__(self,
                 name: str,
                 arg_types: Sequence[TensorType],
                 arg_names: Optional[Sequence[str]] = None):
        self.name = name
        self.arguments: List[Value] = [
            Value(f"%arg{i}", t) for i, t in enumerate(arg_types)
        ]
        #: Python parameter names, carried into the IR as `sar.arg_names`
        #: so emission backends can name ports after them.
        self.arg_names: Optional[List[str]] = (list(arg_names)
                                               if arg_names else None)
        self.operations: List[Operation] = []
        self.returned: Optional[List[Value]] = None
        self._next_id = 0
        #: Emission targets: the function body, plus one frame per open
        #: region (`push_region`). `emit` appends to the innermost.
        self._frames: List[List[Operation]] = [self.operations]
        self._open_regions: List[Region] = []

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
        unit_attributes: Sequence[str] = (),
        location: Optional[Tuple[str, int, int]] = None,
    ) -> Value:
        """Appends an operation with a single result and returns its value."""
        op = Operation(op_name,
                       list(operands), [result_type],
                       dict(attributes or {}),
                       tuple(unit_attributes),
                       location=location)
        op.results = [self._new_value(result_type)]
        self._frames[-1].append(op)
        return op.results[0]

    # -- Region-carrying operations (`sar.iterate`) --------------------------

    def push_region(self, arg_types: Sequence[TensorType]) -> List[Value]:
        """Opens a region: ops emitted until the matching `pop_region` land
        in it. Returns the region's block arguments."""
        region = Region([self._new_value(t) for t in arg_types])
        self._open_regions.append(region)
        self._frames.append(region.operations)
        return region.arguments

    def pop_region(self, yields: Sequence[Value]) -> Region:
        """Closes the innermost region, attaching its yielded values."""
        self._frames.pop()
        region = self._open_regions.pop()
        region.yields = list(yields)
        return region

    def abort_region(self) -> None:
        """Discards the innermost region (tracing its body raised)."""
        self._frames.pop()
        self._open_regions.pop()

    def emit_region_op(
        self,
        op_name: str,
        operands: Sequence[Value],
        result_types: Sequence[TensorType],
        region: Region,
        attributes: Optional[Dict[str, object]] = None,
        unit_attributes: Sequence[str] = (),
        location: Optional[Tuple[str, int, int]] = None,
    ) -> List[Value]:
        """Appends an operation holding `region`, with one result group."""
        op = Operation(op_name,
                       list(operands),
                       list(result_types),
                       dict(attributes or {}),
                       tuple(unit_attributes),
                       location=location)
        op.region = region
        base = self._next_id
        self._next_id += 1
        if len(result_types) == 1:
            op.results = [Value(f"%{base}", result_types[0])]
        else:
            op.results = [
                Value(f"%{base}#{i}", t) for i, t in enumerate(result_types)
            ]
        self._frames[-1].append(op)
        return op.results

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
        attrs = ""
        if self.arg_names:
            names = ", ".join(f'"{n}"' for n in self.arg_names)
            attrs = f" attributes {{sar.arg_names = [{names}]}}"
        lines = [
            f"{indent}func.func @{self.name}({args}) -> ({rets}){attrs} {{"
        ]
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
