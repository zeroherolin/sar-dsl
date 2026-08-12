"""Differential fuzzing: random DSL programs vs a NumPy oracle.

Each iteration builds a random DAG of SAR operations, compiles it on the
CPU backend and compares against a mirrored NumPy evaluation. Seeds are
fixed for reproducibility; a failing seed prints its program.

Environment:
    SAR_FUZZ_ITERS -- number of random programs (default 25)
"""

import inspect
import os
import random

import numpy as np

import sar
from sar.ir import C64, C128, F32, F64

from conftest import requires_cpu

pytestmark = requires_cpu

_ITERS = int(os.environ.get("SAR_FUZZ_ITERS", "25"))


class _Program:
    """A random DAG over (dtype, shape) tensors mirrored in numpy."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.shape = rng.choice([(4, 8), (8, 8), (16, 4), (8, 16)])
        # Working precision per program keeps casts deliberate.
        self.float_dtype = rng.choice([F32, F64])
        self.complex_dtype = C64 if self.float_dtype == F32 else C128
        self.steps = []           # (op, args) mirrored in trace and numpy
        self.num_inputs = rng.randint(1, 3)
        self.input_kinds = [rng.choice(["float", "complex"])
                            for _ in range(self.num_inputs)]
        self._build()

    def _build(self):
        rng = self.rng
        # values: list of (kind, expr, shape); kind is "float"|"complex".
        values = [(kind, ("input", i), self.shape)
                  for i, kind in enumerate(self.input_kinds)]
        for _ in range(rng.randint(2, 10)):
            kind, expr, shape = rng.choice(values)
            op = rng.choice(self._ops_for(kind))
            if op in ("add", "sub", "mul"):
                partners = [v for v in values
                            if v[0] == kind and v[2] == shape]
                other = rng.choice(partners)
                new = (kind, (op, expr, other[1]), shape)
            elif op in ("add_scalar", "mul_scalar"):
                new = (kind, (op, expr, round(rng.uniform(-2, 2), 3)),
                       shape)
            elif op == "maximum":
                new = (kind, (op, expr, round(rng.uniform(-1, 1), 3)),
                       shape)
            elif op == "transpose":
                new = (kind, (op, expr), (shape[1], shape[0]))
            else:
                out_kind = kind
                if op == "abs":
                    out_kind = "float"
                elif op == "expj":
                    out_kind = "complex"
                new = (out_kind, (op, expr), shape)
            values.append(new)
        kind, expr, shape = values[-1]
        self.result = (kind, expr)
        self.result_shape = shape

    def _ops_for(self, kind):
        if kind == "float":
            return ["add", "sub", "mul", "add_scalar", "mul_scalar",
                    "maximum", "cos", "sin", "neg", "transpose", "fftshift",
                    "ifftshift", "expj", "sqrt_abs"]
        return ["add", "sub", "mul", "add_scalar", "mul_scalar",
                "neg", "transpose", "fftshift", "ifftshift", "abs", "fft",
                "ifft"]

    def trace(self, tensors):
        return self._eval(self.result[1], tensors, symbolic=True)

    def oracle(self, arrays):
        return self._eval(self.result[1], arrays, symbolic=False)

    def _eval(self, expr, env, symbolic):
        op = expr[0]
        if op == "input":
            return env[expr[1]]
        x = self._eval(expr[1], env, symbolic)
        if op in ("add", "sub", "mul"):
            y = self._eval(expr[2], env, symbolic)
            if symbolic:
                return {"add": x.__add__, "sub": x.__sub__,
                        "mul": x.__mul__}[op](y)
            return {"add": np.add, "sub": np.subtract,
                    "mul": np.multiply}[op](x, y)
        if op == "add_scalar":
            return x + expr[2] if symbolic else x + type(x.flat[0])(expr[2])
        if op == "mul_scalar":
            return x * expr[2] if symbolic else x * type(x.flat[0])(expr[2])
        if op == "maximum":
            return (sar.maximum(x, expr[2]) if symbolic
                    else np.maximum(x, expr[2]).astype(x.dtype))
        if op == "sqrt_abs":
            # sqrt of |x| keeps the domain valid for random data.
            return (sar.sqrt(sar.absolute(x)) if symbolic
                    else np.sqrt(np.abs(x)))
        if op == "cos":
            return sar.cos(x) if symbolic else np.cos(x)
        if op == "sin":
            return sar.sin(x) if symbolic else np.sin(x)
        if op == "neg":
            return -x
        if op == "transpose":
            return sar.transpose(x) if symbolic else np.ascontiguousarray(x.T)
        if op == "fftshift":
            return (sar.fftshift(x, dim=1) if symbolic
                    else np.fft.fftshift(x, axes=1))
        if op == "ifftshift":
            return (sar.ifftshift(x, dim=1) if symbolic
                    else np.fft.ifftshift(x, axes=1))
        if op == "abs":
            return sar.absolute(x) if symbolic else np.abs(x)
        if op == "expj":
            return sar.expj(x) if symbolic else np.exp(1j * x)
        if op == "fft":
            return (sar.fft(x, dim=1) if symbolic
                    else np.fft.fft(x, axis=1).astype(x.dtype))
        if op == "ifft":
            return (sar.ifft(x, dim=1) if symbolic
                    else np.fft.ifft(x, axis=1).astype(x.dtype))
        raise AssertionError(op)


def _run_one(seed: int):
    rng = random.Random(seed)
    program = _Program(rng)

    result_kind = program.result[0]
    result_shape = program.result_shape
    fdt, cdt = program.float_dtype, program.complex_dtype

    def spec(kind, shape):
        dtype = fdt if kind == "float" else cdt
        from sar.language import _SPEC_BY_DTYPE
        return _SPEC_BY_DTYPE[dtype][shape]

    arg_specs = [spec(kind, program.shape)
                 for kind in program.input_kinds]
    result_spec = spec(result_kind, result_shape)

    def body(*args):
        return program.trace(list(args))

    # Attach the randomly generated positional signature.
    body.__name__ = f"fuzz_{seed}"
    params = [inspect.Parameter(f"arg{i}",
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                annotation=s)
              for i, s in enumerate(arg_specs)]
    body.__signature__ = inspect.Signature(
        params, return_annotation=result_spec)

    kernel = sar.jit(body)

    np_rng = np.random.default_rng(seed)
    arrays = []
    for kind in program.input_kinds:
        data = np_rng.uniform(-2, 2, program.shape)
        if kind == "complex":
            data = data + 1j * np_rng.uniform(-2, 2, program.shape)
            arrays.append(data.astype(cdt.to_numpy()))
        else:
            arrays.append(data.astype(fdt.to_numpy()))

    got = kernel(*arrays)
    want = program.oracle(arrays)

    single = fdt == F32
    rtol = 1e-3 if single else 1e-9
    atol = (1e-3 if single else 1e-9) * (1.0 + np.abs(want).max())
    np.testing.assert_allclose(
        got, want.astype(got.dtype), rtol=rtol, atol=atol,
        err_msg=f"fuzz seed {seed} diverged; program: {program.result}")


def test_fuzz_random_programs():
    for seed in range(_ITERS):
        _run_one(seed)
