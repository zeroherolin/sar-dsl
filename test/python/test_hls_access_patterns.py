"""Cross-row memory access gate for the omega-K HLS lowering.

A cross-row access strides by at least a whole row per innermost
iteration: on an FPGA that is a memory transaction per element rather
than a burst, and it pins the plane on chip instead of letting it
stream. `docs/backends.md` ("HLS memory access patterns") carries the
classification these baselines come from.

Omega-K corner-turns twice in each direction, so a small number of
cross-row accesses is inherent. What this module gates is growth: after
`sar-stage-transposes` runs, the striding side of every corner turn sits
on a small on-chip block, and no full-scene plane is left strided. A
pass that reorders loops, or a fusion that stops firing, shows up here
as either a higher count or a strided plane.
"""

import subprocess

import pytest

from access_audit import cross_row_accesses, walk_accesses
from conftest import requires_cpu

from common.params import synthetic_params
from wka.algorithm import build_kernel as build_wka_kernel

from sar.compiler.toolchain import find_tool

N = 128

#: Cross-row accesses the lowering is allowed to keep. Every one of them is
#: the block-fill store of a staged corner turn -- the one place the stride
#: is meant to live, because the block is small and on chip. Omega-K corner
#: turns four times, on a real and an imaginary plane each, and two of the
#: eight nests carry a fused `fftshift`. See `docs/backends.md`.
BASELINE_CROSS_ROW = 10

#: Element count at or above which a buffer is a full-scene plane rather than
#: a staging block or a line of transform scratch. The staged block is at most
#: a quarter of a plane, which is what separates the two.
_PLANE_ELEMENTS = N * N // 4

#: Bytes a staged transpose block may occupy: exactly what the fixture
#: passes as `transpose-block-bytes` below, so the assertion checks the
#: configured cap rather than a historic default.
_BLOCK_BUDGET = (N * N * 8) // 8

#: Multi-dimensional reads that stay on the non-affine `memref` path. All
#: of them belong to the Stolt band gather, whose address is clamped
#: against the raster edge -- not an affine expression, and computed from
#: the data besides.
BASELINE_NON_AFFINE = 4


@pytest.fixture(scope="module")
def wka_affine():
    """WKA lowered to affine with the options the HLS backend uses.

    Mirroring `sar.backends.hls`'s lower stage keeps the audit on the IR
    that actually reaches the HLS pipeline: the placement thresholds and
    the transpose block size all change which accesses survive.
    """
    params = synthetic_params(N)
    kernel = build_wka_kernel(N, params)
    planes = N * N
    # The staged block has to be a fraction of a plane, or the pass has no
    # reason to stage: a block that swallows the whole raster leaves the
    # corner turn striding the plane itself.
    block_bytes = planes * 8 // 8
    pipeline = (f"--sar-to-affine-pipeline=reuse-buffer-min-elements={planes} "
                f"recompute-min-elements={planes} "
                f"transpose-block-bytes={block_bytes}")
    return subprocess.run([find_tool("sar-opt"), pipeline, "-"],
                          input=kernel.to_mlir(),
                          capture_output=True,
                          text=True,
                          check=True).stdout


@requires_cpu
def test_wka_cross_row_accesses_do_not_grow(wka_affine):
    """The gate: cross-row accesses must not exceed the documented baseline."""
    cross = cross_row_accesses(wka_affine)
    assert len(cross) <= BASELINE_CROSS_ROW, (
        f"cross-row accesses grew to {len(cross)} "
        f"(baseline {BASELINE_CROSS_ROW}); they are:\n" +
        "\n".join(f"  {a}" for a in cross))


@requires_cpu
def test_wka_cross_row_accesses_are_all_staging_block_fills(wka_affine):
    """Every surviving cross-row access must be a block fill, never a read
    of a chain plane.

    This is the property that actually matters, and it is what staging
    buys: a corner turn has to stride something, so `sar-stage-transposes`
    makes that something the write into a small on-chip block while both
    full-size planes are swept along their rows. A cross-row *load* means
    a plane is being read a transaction at a time.
    """
    for access in cross_row_accesses(wka_affine):
        assert access.op == "store", (
            "a cross-row load means a plane is read down its columns rather "
            f"than staged through an on-chip block: {access}")


@requires_cpu
def test_wka_strided_buffers_fit_the_staging_budget(wka_affine):
    """The strided buffers must be blocks the staging pass sized, so the
    stride stays inside on-chip memory however large the scene grows."""
    for access in cross_row_accesses(wka_affine):
        assert access.elements * 8 <= _BLOCK_BUDGET, (
            f"strided buffer exceeds the staged block budget: {access}")


@requires_cpu
def test_wka_non_affine_accesses_do_not_grow(wka_affine):
    """Gather indices must not regress to data-dependent `memref.load`.

    `fftshift` folds its rotation into the indexing map; computing it with
    `arith.remui` instead leaves a `memref.load` that neither the HLS flow
    nor the corner-turn rewrite can see through, forcing whole planes onto
    the strided path.

    The ones that remain are the Stolt interpolation's band gather, whose
    address is clamped (`maxsi`/`minsi`) against the raster edge. Clamping
    is not an affine expression and the position it clamps is computed
    from the data, so those reads are inherently non-affine.
    """
    non_affine = [
        a for a in walk_accesses(wka_affine)
        if a.dialect == "memref" and a.rank >= 2
    ]
    assert len(non_affine) <= BASELINE_NON_AFFINE, (
        f"non-affine multi-dimensional accesses grew to {len(non_affine)} "
        f"(baseline {BASELINE_NON_AFFINE}):\n" +
        "\n".join(f"  {a}" for a in non_affine))


@requires_cpu
def test_wka_audit_sees_the_loop_nest(wka_affine):
    """Guards the auditor itself: a parser that silently stopped tracking
    loops would report zero cross-row accesses and the gate above would
    pass for the wrong reason."""
    accesses = list(walk_accesses(wka_affine))
    assert len(accesses) > 100, "expected the lowered kernel to be access-rich"
    assert any(a.kind() == 'row' for a in accesses), "no row-wise access found"
    assert any(len(a.ivs) >= 2 for a in accesses), "loop nesting not tracked"


@requires_cpu
def test_staging_is_what_keeps_planes_off_the_strided_side():
    """Without `sar-stage-transposes` the corner turns read planes strided.

    Pinning the contrast keeps the gate honest: it shows the surviving
    cross-row accesses are block fills because the pass put them there,
    not because the auditor stopped finding the strided reads.
    """
    params = synthetic_params(N)
    kernel = build_wka_kernel(N, params)
    planes = N * N
    pipeline = (f"--sar-to-affine-pipeline=reuse-buffer-min-elements={planes} "
                f"recompute-min-elements={planes} transpose-block-bytes=0")
    unstaged = subprocess.run([find_tool("sar-opt"), pipeline, "-"],
                              input=kernel.to_mlir(),
                              capture_output=True,
                              text=True,
                              check=True).stdout

    strided_loads = [a for a in cross_row_accesses(unstaged) if a.op == "load"]
    assert strided_loads, (
        "expected unstaged corner turns to read planes down their columns; "
        "if this stopped being true the staging pass may no longer be needed")
