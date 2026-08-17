"""`sar.iterate`: compiled counted loops with tensor-carried state.

The defining property is that the loop stays a loop: a Python `for`
unrolls the body into the IR once per iteration, `sar.iterate` emits one
`sar.iterate` op whatever the trip count. Numerics are pinned against
the trace-time-unrolled equivalent, which must agree exactly.
"""

import subprocess

import numpy as np
import pytest

import sar

from conftest import requires_cpu, requires_hls

N = 16


def _inputs():
    rng = np.random.default_rng(7)
    z = (rng.standard_normal((N, N)) + 1j * rng.standard_normal(
        (N, N))).astype(np.complex128)
    f = np.exp(1j * rng.uniform(-1.0, 1.0, (N, N))).astype(np.complex128)
    return z, f


def _rotate(trips, name):

    @sar.func
    def rotate(z: sar.c128[N, N], f: sar.c128[N, N]) -> sar.c128[N, N]:
        return sar.iterate(trips, lambda acc: acc * f, z)

    rotate.name = name
    return rotate


@requires_cpu
def test_matches_the_unrolled_loop():
    z, f = _inputs()
    got = _rotate(8, "it_cpu")(z, f)
    want = z.copy()
    for _ in range(8):
        want = want * f
    assert np.abs(got - want).max() < 1e-12


def test_the_loop_is_not_unrolled():
    ir = _rotate(64, "it_ir").to_mlir()
    assert ir.count("sar.iterate") == 1
    # One multiply in the body, not sixty-four.
    assert ir.count("sar.mul") == 1


@requires_cpu
def test_multiple_carries_of_mixed_type():
    z, _ = _inputs()
    x = np.abs(z).astype(np.float64)

    @sar.func
    def two(z: sar.c128[N, N],
            x: sar.f64[N, N]) -> (sar.c128[N, N], sar.f64[N, N]):
        return sar.iterate(5, lambda a, b: (a * 0.5, b + sar.absolute(a)), z,
                           x)

    got_z, got_x = two(z, x)
    want_z, want_x = z.copy(), x.copy()
    for _ in range(5):
        want_z, want_x = want_z * 0.5, want_x + np.abs(want_z)
    assert np.abs(got_z - want_z).max() < 1e-12
    assert np.abs(got_x - want_x).max() < 1e-12


@requires_hls
def test_csim_matches_the_reference(tmp_path):
    """The compiled loop must survive the whole HLS flow: decomplexify
    through the region, carry demotion after bufferization, and the
    generated C++ reproducing the reference under csim."""
    from sar.compiler.toolchain import find_tool

    z, f = _inputs()
    want = z.copy()
    for _ in range(8):
        want = want * f

    design = _rotate(8, "it_csim").compile(backend="hls")
    design.write_testbench([z, f], [want], tmp_path, rtol=1e-12)
    clang = find_tool("clang")
    subprocess.run([
        clang + "++", "-O2", "-Wno-unknown-pragmas", "-I", "stubs",
        "it_csim.cpp", "it_csim_tb.cpp", "-o", "csim", "-pthread"
    ],
                   cwd=tmp_path,
                   check=True,
                   capture_output=True)
    result = subprocess.run(["./csim"],
                            cwd=tmp_path,
                            capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stdout
    assert "PASS" in result.stdout


def test_bad_bodies_are_rejected():

    @sar.func
    def zero_trips(z: sar.c128[N, N]) -> sar.c128[N, N]:
        return sar.iterate(0, lambda a: a, z)

    with pytest.raises(sar.TraceError, match="positive"):
        zero_trips.to_mlir()

    @sar.func
    def type_drift(z: sar.c128[N, N]) -> sar.f64[N, N]:
        return sar.iterate(2, lambda a: sar.absolute(a), z)

    with pytest.raises(sar.TraceError, match="type"):
        type_drift.to_mlir()

    @sar.func
    def arity_drift(z: sar.c128[N, N]) -> sar.c128[N, N]:
        return sar.iterate(2, lambda a: (a, a), z)

    with pytest.raises(sar.TraceError, match="carried"):
        arity_drift.to_mlir()
