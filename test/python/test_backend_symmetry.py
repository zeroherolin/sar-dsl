"""Backend symmetry: whatever the DSL can express, every backend must run.

This is the project's central invariant. A construct that lowers on one
backend and not another would split the language in two -- kernels would
stop being portable, and the choice of target would leak into how an
algorithm is written. So the gate is the whole construct set against
every backend, not a sample of it.

Failures here mean either a missing lowering or, worse, an op that only
one backend can reach. Both are bugs.
"""

import warnings
import re

import numpy as np
import pytest

import sar

from conftest import REPO_ROOT, requires_cpu, requires_hls

N = 16

#: One kernel body per DSL construct group. Complex input throughout: it
#: exercises the split-complex path the HLS backend takes, and the ops
#: that only accept floats get a real value derived from it.
CONSTRUCTS = {
    "arith":
    lambda x: x * 2.0 + 1.0 - x / (x + 3.0),
    "sqrt":
    lambda x: sar.sqrt(sar.absolute(x) + 1.0),
    "trig":
    lambda x: sar.cos(sar.real(x)) + sar.sin(sar.imag(x)),
    "exp_log":
    lambda x: sar.log(sar.exp(sar.absolute(x)) + 1.0),
    "atan2":
    lambda x: sar.atan2(sar.imag(x), sar.real(x)),
    "conj":
    sar.conj,
    "cmp_where":
    lambda x: sar.where(
        sar.absolute(x) > 0.5, sar.absolute(x),
        sar.absolute(x) * 0.0),
    "make_complex":
    lambda x: sar.make_complex(sar.real(x), sar.imag(x)),
    "cast":
    lambda x: sar.cast(sar.absolute(x), sar.f32),
    "reduce_sum":
    lambda x: sar.broadcast(sar.sum(sar.absolute(x), axis=1), (N, N), dim=1),
    "reduce_max":
    lambda x: sar.broadcast(sar.max(sar.absolute(x), axis=1), (N, N), dim=1),
    "reduce_min":
    lambda x: sar.broadcast(sar.min(sar.absolute(x), axis=1), (N, N), dim=1),
    "argmax":
    lambda x: sar.broadcast(
        sar.cast(sar.argmax(sar.absolute(x), axis=1), sar.f64), (N, N), dim=1),
    "transpose":
    sar.transpose,
    "slice_concat":
    lambda x: sar.concatenate((x[:N // 2, :], x[N // 2:, :]), dim=0),
    "dynamic_slice":
    lambda x: sar.dynamic_update_slice(
        x, sar.dynamic_slice(x, (1, 2), (N // 2, N // 2)), (N // 2, N // 2)),
    "pad":
    lambda x: sar.pad(x, ((1, 2), (0, 3)), value=0.5),
    "reverse":
    lambda x: sar.flip(x, axis=1),
    "fftshift":
    lambda x: sar.fftshift(x, axis=1),
    "ifftshift":
    lambda x: sar.ifftshift(x, axis=1),
    "fft":
    lambda x: sar.fft(x, axis=1),
    "ifft":
    lambda x: sar.ifft(x, axis=1),
    "interp1d":
    lambda x: sar.interp1d(
        x, sar.broadcast(np.linspace(0, N - 1, N), (N, N), dim=1)),
    "cumsum":
    lambda x: sar.cumsum(x, axis=1),
    "rank_filter":
    lambda x: sar.rank_filter(sar.absolute(x), window=5, rank=1, axis=1),
    "median_filter":
    lambda x: sar.median_filter(sar.absolute(x), window=3, axis=0),
    "sort":
    lambda x: sar.sort(sar.absolute(x), axis=1),
    "gather2d":
    lambda x: sar.gather2d(
        x, sar.broadcast(np.linspace(0, N - 1, N), (N, N), dim=0),
        sar.broadcast(np.linspace(0, N - 1, N), (N, N), dim=1)),
    "iterate":
    lambda x: sar.iterate(4, lambda acc: acc * x, x),
}


def _declared_sar_ops():
    lines = (REPO_ROOT /
             "include/sar/Dialect/SAR/IR/SAROps.td").read_text().splitlines()
    result = set()
    for index, line in enumerate(lines):
        if not re.match(r"\s*def SAR_\w+Op\b", line):
            continue
        declaration = " ".join(lines[index:index + 5])
        match = re.search(r'<"([^"]+)"', declaration)
        assert match is not None, line
        result.add(match.group(1))
    return result


def test_construct_matrix_covers_every_sar_operation():
    traced = set()
    for name, body in CONSTRUCTS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            @sar.func
            def kernel(x):
                return body(x)

            module = kernel.specialize(sar.c128[N, N]).to_mlir()
        traced.update(re.findall(r'"sar\.([^"]+)"', module))

    internal = {"fft_split", "interp1d_split", "gather2d_split"}
    missing = _declared_sar_ops() - internal - traced
    assert not missing, (
        f"SAR operations absent from symmetry matrix: {missing}")


def _compile(name, body, backend, spec=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        @sar.func
        def kernel(x):
            return body(x)

        kernel.name = f"sym_{name}_{backend}"
        return kernel.specialize(spec
                                 or sar.c128[N, N]).compile(backend=backend)


@pytest.mark.parametrize("construct", sorted(CONSTRUCTS))
@requires_cpu
def test_construct_lowers_on_cpu(construct):
    _compile(construct, CONSTRUCTS[construct], "cpu")


@pytest.mark.parametrize("construct", sorted(CONSTRUCTS))
@requires_hls
def test_construct_lowers_on_hls(construct):
    _compile(construct, CONSTRUCTS[construct], "hls")


# Bluestein is the one place where the two backends take visibly
# different routes -- a runtime call against a compile-time-folded
# chirp-z reduction -- so it is worth naming separately. One test per
# backend so each carries its own availability gate.


@requires_cpu
def test_non_power_of_two_fft_lowers_on_cpu():
    _compile("nonpow2",
             lambda x: sar.fft(x, axis=1),
             "cpu",
             spec=sar.c128[12, 12])


@requires_hls
def test_non_power_of_two_fft_lowers_on_hls():
    _compile("nonpow2",
             lambda x: sar.fft(x, axis=1),
             "hls",
             spec=sar.c128[12, 12])


# Rank-1 is a distinct code path for the ops that sweep lines (there is
# no line dimension to sweep), so the symmetry gate covers it explicitly.


def _rank1_body(x):
    magnitude = sar.absolute(sar.fft(x, axis=0))
    return sar.cumsum(sar.sort(sar.median_filter(magnitude, window=3)))


@requires_cpu
def test_rank1_constructs_lower_on_cpu():
    _compile("rank1", _rank1_body, "cpu", spec=sar.c128[N])


@requires_hls
def test_rank1_constructs_lower_on_hls():
    _compile("rank1", _rank1_body, "hls", spec=sar.c128[N])
