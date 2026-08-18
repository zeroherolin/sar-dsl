"""Differential fuzzing: random DSL programs vs a NumPy oracle.

Each iteration builds a random DAG of SAR operations, compiles it on the
CPU backend and compares against a mirrored NumPy evaluation. Seeds are
fixed for reproducibility; a failing seed prints its program.

Environment:
    SAR_FUZZ_ITERS -- number of random programs (default 25)
"""

import builtins
import inspect
import os
import random

import numpy as np

import sar
from sar.ir import C64, C128, F32, F64

from conftest import requires_cpu

pytestmark = requires_cpu


def _fuzz_iterations() -> int:
    raw = os.environ.get("SAR_FUZZ_ITERS", "25")
    try:
        iterations = int(raw)
    except ValueError as err:
        raise ValueError("SAR_FUZZ_ITERS must be a positive integer") from err
    if iterations < 1:
        raise ValueError("SAR_FUZZ_ITERS must be a positive integer")
    return iterations


_ITERS = _fuzz_iterations()


class _Program:
    """A random DAG over (dtype, shape) tensors mirrored in numpy."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        # (6, 12) exercises the Bluestein path of the FFT ops.
        self.shape = rng.choice([(4, 8), (8, 8), (16, 4), (8, 16), (6, 12)])
        # Working precision per program keeps casts deliberate.
        self.float_dtype = rng.choice([F32, F64])
        self.complex_dtype = C64 if self.float_dtype == F32 else C128
        self.num_inputs = rng.randint(1, 3)
        self.input_kinds = [
            rng.choice(["float", "complex", "int"])
            for _ in range(self.num_inputs)
        ]
        self._build()

    def _build(self):
        rng = self.rng
        # values: list of (kind, expr, shape); kind is "float"|"complex".
        values = [(kind, ("input", i), self.shape)
                  for i, kind in enumerate(self.input_kinds)]
        for _ in range(rng.randint(2, 10)):
            kind, expr, shape = rng.choice(values)
            op = rng.choice(self._ops_for(kind, len(shape)))
            if op in ("add", "sub", "mul", "div", "atan2", "make_complex"):
                partners = [
                    v for v in values if v[0] == kind and v[2] == shape
                ]
                other = rng.choice(partners)
                out_kind = "complex" if op == "make_complex" else kind
                new = (out_kind, (op, expr, other[1]), shape)
            elif op in ("add_scalar", "mul_scalar"):
                new = (kind, (op, expr, round(rng.uniform(-2, 2), 3)), shape)
            elif op == "maximum":
                new = (kind, (op, expr, round(rng.uniform(-1, 1), 3)), shape)
            elif op == "transpose":
                new = (kind, (op, expr), (shape[1], shape[0]))
            elif op in ("sum_r", "max_r", "min_r"):
                dim = rng.choice([0, 1])
                new = (kind, (op, expr, dim), (shape[1 - dim], ))
            elif op in ("argmax_r", "argmin_r"):
                dim = rng.choice([0, 1])
                new = ("int", (op, expr, dim), (shape[1 - dim], ))
            elif op == "slice_tl":
                sub = (builtins.max(1, shape[0] // 2),
                       builtins.max(1, shape[1] // 2))
                new = (kind, (op, expr, sub), sub)
            elif op == "slice_stride":
                # Every other element; needs at least two per axis.
                if shape[0] < 2 or shape[1] < 2:
                    continue
                sub = (shape[0] // 2, shape[1] // 2)
                new = (kind, (op, expr, sub), sub)
            elif op == "concat_h":
                partners = [
                    v for v in values
                    if v[0] == kind and len(v[2]) == 2 and v[2][0] == shape[0]
                ]
                if not partners:
                    continue
                other = rng.choice(partners)
                out = (shape[0], shape[1] + other[2][1])
                new = (kind, (op, expr, other[1]), out)
            elif op == "pad_r":
                out = (shape[0], shape[1] + 3)
                new = (kind, (op, expr), out)
            elif op in ("cast_to_int", "cast_from_int"):
                if op == "cast_to_int":
                    new = ("int", (op, expr), shape)
                else:
                    new = ("float", (op, expr), shape)
            else:
                out_kind = kind
                if op in ("abs", "real", "imag", "angle"):
                    out_kind = "float"
                elif op == "expj":
                    out_kind = "complex"
                new = (out_kind, (op, expr), shape)
            values.append(new)
        kind, expr, shape = values[-1]
        self.result = (kind, expr)
        self.result_shape = shape

    def _ops_for(self, kind, rank):
        if kind == "float":
            ops = [
                "add", "sub", "mul", "div", "add_scalar", "mul_scalar",
                "maximum", "cos", "sin", "neg", "expj", "sqrt_abs",
                "exp_neg_abs", "log_abs1", "atan2", "make_complex",
                "where_clip", "cast_to_int"
            ]
            rank2 = [
                "transpose", "fftshift", "ifftshift", "sum_r", "max_r",
                "min_r", "argmax_r", "argmin_r", "slice_tl", "slice_stride",
                "pad_r", "flip", "concat_h"
            ]
        elif kind == "int":
            # Integer tensors: only elementwise add/sub/mul (SAR_Tensor
            # includes ints) and cast back to float; scalar ops, reductions
            # and argmax/argmin require float elements in the IR.
            ops = ["add", "sub", "mul", "cast_from_int"]
            rank2 = ["transpose", "slice_tl", "slice_stride", "concat_h"]
        else:  # complex
            ops = [
                "add", "sub", "mul", "div", "add_scalar", "mul_scalar", "neg",
                "abs", "conj", "real", "imag", "angle", "where_conj"
            ]
            rank2 = [
                "transpose", "fftshift", "ifftshift", "fft", "ifft", "sum_r",
                "slice_tl", "slice_stride", "pad_r", "flip", "concat_h"
            ]
        return ops + rank2 if rank == 2 else ops

    def trace(self, tensors):
        return self._eval(self.result[1], tensors, symbolic=True)

    def oracle(self, arrays):
        return self._eval(self.result[1], arrays, symbolic=False)

    def _eval(self, expr, env, symbolic):
        op = expr[0]
        if op == "input":
            return env[expr[1]]
        x = self._eval(expr[1], env, symbolic)
        if op in ("add", "sub", "mul", "div", "atan2", "make_complex"):
            y = self._eval(expr[2], env, symbolic)
            if op == "atan2":
                return (sar.atan2(x, y)
                        if symbolic else np.arctan2(x, y).astype(x.dtype))
            if op == "make_complex":
                return (sar.make_complex(x, y) if symbolic else (x + 1j * y))
            if symbolic:
                return {
                    "add": x.__add__,
                    "sub": x.__sub__,
                    "mul": x.__mul__,
                    "div": x.__truediv__
                }[op](y)
            return {
                "add": np.add,
                "sub": np.subtract,
                "mul": np.multiply,
                "div": np.divide
            }[op](x, y)
        if op == "add_scalar":
            return x + expr[2] if symbolic else x + type(x.flat[0])(expr[2])
        if op == "mul_scalar":
            return x * expr[2] if symbolic else x * type(x.flat[0])(expr[2])
        if op == "maximum":
            return (sar.maximum(x, expr[2]) if symbolic else np.maximum(
                x, expr[2]).astype(x.dtype))
        if op == "sqrt_abs":
            return (sar.sqrt(sar.absolute(x)) if symbolic else np.sqrt(
                np.abs(x)))
        if op == "exp_neg_abs":
            return (sar.exp(-sar.absolute(x))
                    if symbolic else np.exp(-np.abs(x)))
        if op == "log_abs1":
            return (sar.log(sar.absolute(x) +
                            1.0) if symbolic else np.log(np.abs(x) + 1.0))
        if op == "where_clip":
            if symbolic:
                return sar.where(x > 0.3, x * 2.0, x)
            return np.where(x > 0.3, x * type(x.flat[0])(2.0), x)
        if op == "where_conj":
            if symbolic:
                return sar.where(sar.real(x) > 0.0, x, sar.conj(x))
            return np.where(x.real > 0.0, x, np.conj(x))
        if op == "cast_to_int":
            return sar.cast(x, sar.i64) if symbolic else x.astype(np.int64)
        if op == "cast_from_int":
            fdt = F32 if self.float_dtype == F32 else F64
            if symbolic:
                from sar.language import _SPEC_BY_DTYPE
                return sar.cast(x, _SPEC_BY_DTYPE[fdt])
            return x.astype(fdt.to_numpy())
        if op == "conj":
            return sar.conj(x) if symbolic else np.conj(x)
        if op == "real":
            return sar.real(x) if symbolic else np.ascontiguousarray(x.real)
        if op == "imag":
            return sar.imag(x) if symbolic else np.ascontiguousarray(x.imag)
        if op == "angle":
            return (sar.angle(x) if symbolic else np.angle(x).astype(
                np.abs(x).dtype))
        if op == "cos":
            return sar.cos(x) if symbolic else np.cos(x)
        if op == "sin":
            return sar.sin(x) if symbolic else np.sin(x)
        if op == "neg":
            return -x
        if op == "transpose":
            return sar.transpose(x) if symbolic else np.ascontiguousarray(x.T)
        if op == "flip":
            return (sar.flip(x, dim=1) if symbolic else np.ascontiguousarray(
                np.flip(x, axis=1)))
        if op in ("sum_r", "max_r", "min_r"):
            dim = expr[2]
            if symbolic:
                fn = {"sum_r": sar.sum, "max_r": sar.max, "min_r": sar.min}
                return fn[op](x, dim=dim)
            fn = {"sum_r": np.sum, "max_r": np.max, "min_r": np.min}
            return np.ascontiguousarray(fn[op](x, axis=dim))
        if op in ("argmax_r", "argmin_r"):
            dim = expr[2]
            if symbolic:
                fn = {"argmax_r": sar.argmax, "argmin_r": sar.argmin}
                return fn[op](x, dim=dim)
            fn = {"argmax_r": np.argmax, "argmin_r": np.argmin}
            return np.ascontiguousarray(fn[op](x, axis=dim))
        if op == "slice_tl":
            r, c = expr[2]
            return (x[0:r,
                      0:c] if symbolic else np.ascontiguousarray(x[:r, :c]))
        if op == "slice_stride":
            r, c = expr[2]
            return (x[0:r * 2:2, 0:c * 2:2] if symbolic else
                    np.ascontiguousarray(x[:r * 2:2, :c * 2:2]))
        if op == "concat_h":
            y = self._eval(expr[2], env, symbolic)
            if symbolic:
                return sar.concatenate([x, y], dim=1)
            return np.ascontiguousarray(np.concatenate([x, y], axis=1))
        if op == "pad_r":
            if symbolic:
                return sar.pad(x, ((0, 0), (1, 2)), value=0.5)
            return np.pad(x, ((0, 0), (1, 2)),
                          constant_values=0.5).astype(x.dtype)
        if op == "fftshift":
            return (sar.fftshift(x, dim=1)
                    if symbolic else np.fft.fftshift(x, axes=1))
        if op == "ifftshift":
            return (sar.ifftshift(x, dim=1)
                    if symbolic else np.fft.ifftshift(x, axes=1))
        if op == "abs":
            return sar.absolute(x) if symbolic else np.abs(x)
        if op == "expj":
            return sar.expj(x) if symbolic else np.exp(1j * x)
        if op == "fft":
            return (sar.fft(x, dim=1)
                    if symbolic else np.fft.fft(x, axis=1).astype(x.dtype))
        if op == "ifft":
            return (sar.ifft(x, dim=1)
                    if symbolic else np.fft.ifft(x, axis=1).astype(x.dtype))
        raise AssertionError(op)


def _run_one(seed: int):
    rng = random.Random(seed)
    program = _Program(rng)

    result_kind = program.result[0]
    result_shape = program.result_shape
    fdt, cdt = program.float_dtype, program.complex_dtype

    def spec(kind, shape):
        if kind == "int":
            from sar.language import _SPEC_BY_DTYPE
            return _SPEC_BY_DTYPE[sar.ir.I64][shape]
        dtype = fdt if kind == "float" else cdt
        from sar.language import _SPEC_BY_DTYPE
        return _SPEC_BY_DTYPE[dtype][shape]

    arg_specs = [spec(kind, program.shape) for kind in program.input_kinds]
    result_spec = spec(result_kind, result_shape)

    def body(*args):
        return program.trace(list(args))

    # Attach the randomly generated positional signature.
    body.__name__ = f"fuzz_{seed}"
    params = [
        inspect.Parameter(f"arg{i}",
                          inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=s) for i, s in enumerate(arg_specs)
    ]
    body.__signature__ = inspect.Signature(params,
                                           return_annotation=result_spec)

    kernel = sar.func(body)

    np_rng = np.random.default_rng(seed)
    arrays = []
    for kind in program.input_kinds:
        if kind == "int":
            data = np_rng.integers(-100, 100, program.shape)
            arrays.append(data.astype(np.int64))
        elif kind == "complex":
            data = np_rng.uniform(-2, 2, program.shape)
            data = data + 1j * np_rng.uniform(-2, 2, program.shape)
            arrays.append(data.astype(cdt.to_numpy()))
        else:
            data = np_rng.uniform(-2, 2, program.shape)
            arrays.append(data.astype(fdt.to_numpy()))

    got = kernel(*arrays)
    want = program.oracle(arrays)

    single = fdt == F32
    rtol = 1e-3 if single else 1e-9
    atol = (1e-3 if single else 1e-9) * (1.0 + np.abs(want).max())
    np.testing.assert_allclose(
        got,
        want.astype(got.dtype),
        rtol=rtol,
        atol=atol,
        err_msg=f"fuzz seed {seed} diverged; program: {program.result}")


def test_fuzz_random_programs():
    # Random op chains legitimately trip the spectral-domain diagnostics
    # (e.g. fftshift feeding ifft); the fuzzer checks numerics only.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sar.DomainWarning)
        for seed in range(_ITERS):
            _run_one(seed)
