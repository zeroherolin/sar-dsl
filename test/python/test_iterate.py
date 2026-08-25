"""`sar.iterate`: compiled counted loops with tensor-carried state.

The defining property is that the loop stays a loop: a Python `for`
unrolls the body into the IR once per iteration, `sar.iterate` emits one
`sar.iterate` op whatever the trip count. Numerics are pinned against
the trace-time-unrolled equivalent, which must agree exactly.
"""

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
def test_iterate_hls_csim_matches_the_reference(tmp_path):
    """The compiled loop must survive the whole HLS flow: decomplexify
    through the region, carry demotion after bufferization, and the
    generated C++ reproducing the reference in C-sim through Vitis HLS."""
    from conftest import run_hls_csim

    z, f = _inputs()
    want = z.copy()
    for _ in range(8):
        want = want * f

    design = _rotate(8, "it_hls_csim").compile(backend="hls")
    design.write_testbench([z, f], [want], tmp_path, rtol=1e-12)
    run_hls_csim(tmp_path, "it_hls_csim")


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


@requires_cpu
def test_index_argument_matches_host_loop():
    """With index=True the body's first argument is the 0-based iteration
    index as i64[1]; the compiled loop must agree with the host loop that
    reads Python's own counter."""
    trips = 5

    @sar.func
    def scaled(z: sar.c128[N, N]) -> sar.c128[N, N]:

        def step(i, acc):
            gain = sar.cast(i, sar.f64) + 1.0
            return acc * sar.broadcast(sar.concatenate((gain, ) * N, dim=0),
                                       (N, N),
                                       dim=0)

        return sar.iterate(trips, step, z, index=True)

    z, _ = _inputs()
    expected = z.copy()
    for i in range(trips):
        expected = expected * float(i + 1)
    np.testing.assert_allclose(scaled(z), expected, rtol=1e-12)


@requires_hls
def test_index_argument_emits_on_hls():
    """Backend symmetry for the index form (emission gate)."""

    @sar.func
    def scaled(z: sar.c128[N, N]) -> sar.c128[N, N]:

        def step(i, acc):
            gain = sar.cast(i, sar.f64) + 1.0
            return acc * sar.broadcast(sar.concatenate((gain, ) * N, dim=0),
                                       (N, N),
                                       dim=0)

        return sar.iterate(3, step, z, index=True)

    scaled.name = "iterate_index_hls"
    assert "#pragma HLS" in scaled.compile(backend="hls").source()


def _chunked_kernel(name):
    block = 4

    @sar.func
    def chunked(z: sar.c128[N, N]) -> sar.c128[N, N]:

        def step(i, out):
            offset = i * block
            tile = sar.dynamic_slice(z, (offset, 0), (block, N))
            return sar.dynamic_update_slice(out, tile * 2.0, (offset, 0))

        return sar.iterate(N // block, step, z * 0.0, index=True)

    chunked.name = name
    return chunked


@requires_cpu
def test_index_drives_dynamic_slice_and_update():
    z, _ = _inputs()
    np.testing.assert_allclose(_chunked_kernel("iterate_chunks_cpu")(z),
                               z * 2.0,
                               rtol=1e-12)


@requires_hls
def test_dynamic_slice_loop_hls_csim(tmp_path):
    from conftest import run_hls_csim

    z, _ = _inputs()
    design = _chunked_kernel("iterate_chunks_hls").compile(backend="hls")
    design.write_testbench([z], [z * 2.0], tmp_path, rtol=1e-12)
    run_hls_csim(tmp_path, "iterate_chunks_hls")


def test_dynamic_slice_argument_validation():

    @sar.func
    def bad_offset(z: sar.c128[N, N]) -> sar.c128[4, N]:
        return sar.dynamic_slice(z, (1.5, 0), (4, N))

    with pytest.raises(sar.TraceError, match="offset #0"):
        bad_offset.to_mlir()

    @sar.func
    def bad_size(z: sar.c128[N, N]) -> sar.c128[N + 1, N]:
        return sar.dynamic_slice(z, (0, 0), (N + 1, N))

    with pytest.raises(sar.TraceError, match="span exceeds"):
        bad_size.to_mlir()


def test_index_body_takes_one_extra_argument():
    """index=True hands the body exactly one extra leading argument."""

    @sar.func
    def wrong(z: sar.c128[N, N]) -> sar.c128[N, N]:
        # body written for the no-index signature
        return sar.iterate(2, lambda acc: acc, z, index=True)

    with pytest.raises(TypeError):
        wrong.to_mlir()
