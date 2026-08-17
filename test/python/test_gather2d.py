"""`sar.gather2d`: 2-D gather at data-dependent positions.

Numerics are pinned against a direct numpy evaluation of the documented
formula for every kernel/boundary combination, including positions that
fall outside the data -- the boundary policy is where a backprojection's
edge pulses live or die.
"""

import numpy as np
import pytest

import sar

from conftest import requires_cpu, requires_hls

DATA_SHAPE = (12, 9)
OUT_SHAPE = (7, 5)


def _inputs():
    rng = np.random.default_rng(3)
    data = (rng.standard_normal(DATA_SHAPE) +
            1j * rng.standard_normal(DATA_SHAPE)).astype(np.complex128)
    rows = rng.uniform(-1.5, DATA_SHAPE[0] + 0.5, OUT_SHAPE)
    cols = rng.uniform(-1.5, DATA_SHAPE[1] + 0.5, OUT_SHAPE)
    return data, rows, cols


def _reference(data, rows, cols, kernel, boundary):
    n, m = data.shape
    out = np.zeros(rows.shape, complex)
    clamp = lambda v, hi: min(max(v, 0), hi - 1)  # noqa: E731
    for i, j in np.ndindex(rows.shape):
        r, c = rows[i, j], cols[i, j]
        if kernel == "nearest":
            taps = [(int(np.floor(r + 0.5)), int(np.floor(c + 0.5)), 1.0)]
        else:
            r0, c0 = int(np.floor(r)), int(np.floor(c))
            fr, fc = r - r0, c - c0
            taps = [(r0 + dr, c0 + dc,
                     (fr if dr else 1 - fr) * (fc if dc else 1 - fc))
                    for dr in (0, 1) for dc in (0, 1)]
        for ri, ci, w in taps:
            if boundary == "zero" and not (0 <= ri < n and 0 <= ci < m):
                continue
            out[i, j] += w * data[clamp(ri, n), clamp(ci, m)]
    return out


def _kernel(kernel, boundary, name):
    n, m = DATA_SHAPE
    o0, o1 = OUT_SHAPE

    @sar.func
    def gather(d: sar.c128[n, m], r: sar.f64[o0, o1],
               c: sar.f64[o0, o1]) -> sar.c128[o0, o1]:
        return sar.gather2d(d, r, c, kernel=kernel, boundary=boundary)

    gather.name = name
    return gather


@requires_cpu
@pytest.mark.parametrize("kernel", ["nearest", "linear"])
@pytest.mark.parametrize("boundary", ["zero", "edge"])
def test_matches_the_reference(kernel, boundary):
    data, rows, cols = _inputs()
    got = _kernel(kernel, boundary, f"g2d_{kernel}_{boundary}")(data, rows,
                                                                cols)
    want = _reference(data, rows, cols, kernel, boundary)
    assert np.abs(got - want).max() < 1e-12


@requires_hls
def test_emits_on_hls():
    design = _kernel("linear", "zero", "g2d_hls").compile(backend="hls")
    assert "#pragma HLS" in design.source()


@requires_hls
def test_csim_matches_the_reference(tmp_path):
    """The split-complex lowering must be numerically right, not just
    emittable: the cpu path gathers through the complex form, so only
    csim exercises the plane-pair form the HLS backend runs."""
    import subprocess

    from sar.compiler.toolchain import find_tool

    data, rows, cols = _inputs()
    want = _reference(data, rows, cols, "linear", "zero")

    design = _kernel("linear", "zero", "g2d_csim").compile(backend="hls")
    design.write_testbench([data, rows, cols], [want], tmp_path, rtol=1e-10)
    clang = find_tool("clang")
    subprocess.run([
        clang + "++", "-O2", "-Wno-unknown-pragmas", "-I", "stubs",
        "g2d_csim.cpp", "g2d_csim_tb.cpp", "-o", "csim", "-pthread"
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


def test_bad_arguments_are_rejected():
    n, m = DATA_SHAPE

    @sar.func
    def bad_kernel(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
        return sar.gather2d(d, p, p, kernel="cubic")

    with pytest.raises(sar.TraceError, match="kernel"):
        bad_kernel.to_mlir()

    @sar.func
    def bad_positions(d: sar.c128[n, m], p: sar.f32[n, m]) -> sar.c128[n, m]:
        return sar.gather2d(d, p, p)

    with pytest.raises(sar.TraceError, match="f64"):
        bad_positions.to_mlir()
