"""Numerical validation of the banded (line-buffer) interpolation gather.

`convert-sar-interp-to-affine` stages each output row through a narrow on-chip
band buffer when displacement-range analysis proves `|positions - j|` bounded.
The banded and full-plane gathers perform identical arithmetic in identical
order -- only the address the sample is fetched from differs -- so the two must
agree exactly, and both must agree with a numpy oracle.

The same IR is compiled twice through `sar-affine-to-llvm-pipeline`, once with
the banded path enabled (the default) and once with
`interp-enable-banded-gather=0`.
"""

import numpy as np
import pytest

from conftest import requires_cpu
from conftest import compile_split_kernel as _compile_split_kernel
from conftest import run_split as _run_split

pytestmark = requires_cpu


def _lit(v):
    s = f"{float(v):.17g}"
    return s if any(c in s for c in ".eEn") else s + ".0"


def _dense(values):
    # MLIR needs a decimal point to read a literal as floating point.
    return "[" + ", ".join(_lit(v) for v in values) + "]"


def _module(n, m, positions_1d, boundary=None):
    """A split-complex interp whose positions are an identity ramp plus a
    bounded per-column shift -- the shape the analysis is meant to prove."""
    attrs = f' {{boundary = "{boundary}"}}' if boundary else ""
    return f"""
func.func @ip(%re: tensor<{n}x{m}xf64>, %im: tensor<{n}x{m}xf64>)
    -> (tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>) {{
  %p1d = "sar.constant"() <{{value = dense<{_dense(positions_1d)}>
      : tensor<{m}xf64>}}> : () -> tensor<{m}xf64>
  %p = "sar.broadcast"(%p1d) <{{dim = 1 : i64}}>
      : (tensor<{m}xf64>) -> tensor<{n}x{m}xf64>
  %r, %i = sar.interp1d_split %re, %im, %p{attrs}
      : (tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>)
      -> (tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>)
  return %r, %i : tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>
}}
"""


def _module_2d(positions):
    n, m = positions.shape
    dense = "[" + ", ".join(_dense(row) for row in positions) + "]"
    return f"""
func.func @ip(%re: tensor<{n}x{m}xf64>, %im: tensor<{n}x{m}xf64>)
    -> (tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>) {{
  %p = "sar.constant"() <{{value = dense<{dense}>
      : tensor<{n}x{m}xf64>}}> : () -> tensor<{n}x{m}xf64>
  %r, %i = sar.interp1d_split %re, %im, %p
      : (tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>)
      -> (tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>)
  return %r, %i : tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>
}}
"""


def _resolve(idx, cols, boundary):
    """The source index a tap resolves to, or None when it contributes zero."""
    if 0 <= idx < cols:
        return idx
    if boundary == "zero":
        return None
    if boundary == "edge":
        return int(np.clip(idx, 0, cols - 1))
    mirrored = -idx - 1 if idx < 0 else 2 * cols - idx - 1
    return int(np.clip(mirrored, 0, cols - 1))


def _oracle(data, positions, taps=8, boundary="zero"):
    """Direct numpy port of the emitted windowed-sinc (hann) gather."""
    n, m = data.shape
    half = taps // 2
    out = np.zeros_like(data)
    for i in range(n):
        for j in range(m):
            pos = positions[i, j]
            idx0 = int(np.floor(pos))
            acc = 0.0 + 0.0j
            for k in range(1 - half, half + 1):
                resolved = _resolve(idx0 + k, m, boundary)
                if resolved is None:
                    continue
                d = pos - (idx0 + k)
                w = np.sinc(d) * (0.5 + 0.5 * np.cos(np.pi * d / half))
                acc += data[i, resolved] * w
            out[i, j] = acc
    return out


def _run(mlir,
         name,
         tmp_path,
         banded,
         top="ip",
         full_row_bytes=0,
         cache_copies=1):
    pipeline = "--sar-affine-to-llvm-pipeline"
    options = []
    if not banded:
        options.append("interp-enable-banded-gather=0")
    if full_row_bytes:
        options.append(f"interp-full-row-max-bytes={full_row_bytes}")
    if cache_copies != 1:
        options.append(f"interp-cache-copies={cache_copies}")
    if options:
        pipeline += "=" + " ".join(options)
    return _compile_split_kernel(mlir, top, tmp_path / name, pipeline=pipeline)


def _runtime_positions_module(n, m):
    return f"""
func.func @ip(%re: tensor<{n}x{m}xf64>, %im: tensor<{n}x{m}xf64>,
              %pos: tensor<{n}x{m}xf64>)
    -> (tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>) {{
  %r, %i = sar.interp1d_split %re, %im, %pos
      : (tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>)
      -> (tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>)
  return %r, %i : tensor<{n}x{m}xf64>, tensor<{n}x{m}xf64>
}}
"""


def test_full_row_cache_matches_runtime_position_gather(tmp_path):
    """An unbounded runtime position field may use a complete row cache.

    The cached and direct paths differ only in where the source sample is
    loaded from, so they must remain bit-identical for arbitrary positions.
    """
    n, m = 4, 32
    mlir = _runtime_positions_module(n, m)
    (tmp_path / "cached").mkdir()
    (tmp_path / "direct").mkdir()
    _, cached = _run(mlir,
                     "cached",
                     tmp_path,
                     banded=True,
                     full_row_bytes=2 * m * 8 * 4,
                     cache_copies=4)
    _, direct = _run(mlir, "direct", tmp_path, banded=False)

    rng = np.random.default_rng(314)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    positions = np.empty((n, m), dtype=np.float64)
    for row in range(n):
        positions[row] = (np.arange(m, dtype=np.float64) +
                          rng.uniform(-5.0, 5.0, size=m))
    args = [
        np.ascontiguousarray(data.real),
        np.ascontiguousarray(data.imag), positions
    ]
    got_re, got_im = _run_split(cached, args, [(n, m), (n, m)], np.float64)
    ref_re, ref_im = _run_split(direct, args, [(n, m), (n, m)], np.float64)
    np.testing.assert_array_equal(got_re, ref_re)
    np.testing.assert_array_equal(got_im, ref_im)


@pytest.mark.parametrize(
    "shift_fn,label",
    [
        (lambda m: np.zeros(m), "zero"),
        (lambda m: np.full(m, 0.5), "half"),
        # Sub-band sinusoidal wander, the shape an RCMC/Stolt shift really has.
        (lambda m: 3.0 * np.sin(np.linspace(0, 6 * np.pi, m)), "wander"),
        # Runs off both row edges: exercises the staging clamp at each end.
        (lambda m: np.full(m, -4.0), "low_edge"),
        (lambda m: np.full(m, 4.0), "high_edge"),
    ])
def test_banded_matches_full_plane_and_oracle(tmp_path, shift_fn, label):
    n, m = 8, 64
    positions_1d = np.arange(m, dtype=np.float64) + shift_fn(m)
    mlir = _module(n, m, positions_1d)

    (tmp_path / "b").mkdir()
    (tmp_path / "f").mkdir()
    _, fn_banded = _run(mlir, "b", tmp_path, banded=True, cache_copies=4)
    _, fn_full = _run(mlir, "f", tmp_path, banded=False)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    re = np.ascontiguousarray(data.real)
    im = np.ascontiguousarray(data.imag)

    got_re, got_im = _run_split(fn_banded, [re, im], [(n, m), (n, m)],
                                np.float64)
    want_re, want_im = _run_split(fn_full, [re, im], [(n, m), (n, m)],
                                  np.float64)

    # Identical arithmetic, so identical bits.
    np.testing.assert_array_equal(got_re, want_re)
    np.testing.assert_array_equal(got_im, want_im)

    positions = np.tile(positions_1d, (n, 1))
    ref = _oracle(data, positions)
    np.testing.assert_allclose(got_re + 1j * got_im,
                               ref,
                               rtol=1e-11,
                               atol=1e-11)


def test_nonlinear_constant_positions_still_band(tmp_path):
    """A position field the affine form cannot follow -- `sqrt` over a ramp,
    the shape every frequency-domain remapping takes -- is still fully
    determined at compile time, so folding it must recover the exact band."""
    n, m = 4, 64
    j = np.arange(m, dtype=np.float64)
    # sqrt((j + a)^2 + b^2) - a: a bounded perturbation of the identity ramp
    # that no affine tracking can see through.
    positions_1d = np.sqrt((j + 500.0)**2 + 40.0**2) - 500.0
    mlir = _module(n, m, positions_1d)

    (tmp_path / "b").mkdir()
    (tmp_path / "f").mkdir()
    _, banded = _run(mlir, "b", tmp_path, banded=True)
    _, full = _run(mlir, "f", tmp_path, banded=False)

    rng = np.random.default_rng(7)
    re = np.ascontiguousarray(rng.standard_normal((n, m)))
    im = np.ascontiguousarray(rng.standard_normal((n, m)))

    got_re, got_im = _run_split(banded, [re, im], [(n, m), (n, m)], np.float64)
    ref_re, ref_im = _run_split(full, [re, im], [(n, m), (n, m)], np.float64)
    np.testing.assert_allclose(got_re, ref_re, rtol=0, atol=0)
    np.testing.assert_allclose(got_im, ref_im, rtol=0, atol=0)

    positions = np.broadcast_to(positions_1d, (n, m))
    oracle = _oracle(re + 1j * im, positions)
    np.testing.assert_allclose(got_re, oracle.real, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(got_im, oracle.imag, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("boundary", ["edge", "reflect"])
@pytest.mark.parametrize("shift", [-4.0, 4.0])
def test_banded_honours_boundary_policy(tmp_path, boundary, shift):
    """The band stages source columns with a clamp, which is exactly `edge`;
    `reflect` cannot be served from the band (a mirrored index leaves the
    resident window) so it falls back to the full plane. Either way the
    result must match the policy's oracle, and the two paths must agree.

    The shift runs the tap support off a row edge, which is the only place a
    policy is observable.
    """
    n, m = 8, 64
    positions_1d = np.arange(m, dtype=np.float64) + shift
    mlir = _module(n, m, positions_1d, boundary=boundary)

    (tmp_path / "b").mkdir()
    (tmp_path / "f").mkdir()
    _, fn_banded = _run(mlir, "b", tmp_path, banded=True)
    _, fn_full = _run(mlir, "f", tmp_path, banded=False)

    rng = np.random.default_rng(101)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    re = np.ascontiguousarray(data.real)
    im = np.ascontiguousarray(data.imag)

    got_re, got_im = _run_split(fn_banded, [re, im], [(n, m), (n, m)],
                                np.float64)
    want_re, want_im = _run_split(fn_full, [re, im], [(n, m), (n, m)],
                                  np.float64)

    # Identical arithmetic in identical order, so identical bits.
    np.testing.assert_array_equal(got_re, want_re)
    np.testing.assert_array_equal(got_im, want_im)

    positions = np.tile(positions_1d, (n, 1))
    ref = _oracle(data, positions, boundary=boundary)
    np.testing.assert_allclose(got_re + 1j * got_im,
                               ref,
                               rtol=1e-11,
                               atol=1e-11)


def _gather2d_module(n, row_shift):
    """A gather2d whose row coordinate is the output row plus a constant
    shift (provably bounded) and whose column coordinate is arbitrary."""
    iota = _dense(np.arange(n, dtype=np.float64))
    return f"""
module {{
  func.func @g2d(%re: tensor<{n}x{n}xf64>, %im: tensor<{n}x{n}xf64>)
      -> (tensor<{n}x{n}xf64>, tensor<{n}x{n}xf64>) {{
    %iota = sar.constant dense<{iota}> : tensor<{n}xf64>
    %rows0 = sar.broadcast %iota {{dim = 0 : i64}}
        : tensor<{n}xf64> -> tensor<{n}x{n}xf64>
    %rows = sar.add_scalar %rows0, {_lit(row_shift)} : tensor<{n}x{n}xf64>
    %cols = sar.broadcast %iota {{dim = 1 : i64}}
        : tensor<{n}xf64> -> tensor<{n}x{n}xf64>
    %or, %oi = sar.gather2d_split %re, %im, %rows, %cols
        : (tensor<{n}x{n}xf64>, tensor<{n}x{n}xf64>, tensor<{n}x{n}xf64>,
           tensor<{n}x{n}xf64>)
        -> (tensor<{n}x{n}xf64>, tensor<{n}x{n}xf64>)
    return %or, %oi : tensor<{n}x{n}xf64>, tensor<{n}x{n}xf64>
  }}
}}
"""


@pytest.mark.parametrize("row_shift", [0.0, 1.5, -2.25])
def test_banded_gather2d_matches_full_plane(tmp_path, row_shift):
    """The row-band gather2d and the full-plane path perform identical
    arithmetic, so the results must agree bit for bit -- including at the
    row edges the staging clamp covers."""
    n = 16
    mlir = _gather2d_module(n, row_shift)

    (tmp_path / "b").mkdir()
    (tmp_path / "f").mkdir()
    _, fn_banded = _run(mlir, "b", tmp_path, banded=True, top="g2d")
    _, fn_full = _run(mlir, "f", tmp_path, banded=False, top="g2d")

    rng = np.random.default_rng(7)
    data = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    re = np.ascontiguousarray(data.real)
    im = np.ascontiguousarray(data.imag)

    got_re, got_im = _run_split(fn_banded, [re, im], [(n, n), (n, n)],
                                np.float64)
    want_re, want_im = _run_split(fn_full, [re, im], [(n, n), (n, n)],
                                  np.float64)
    np.testing.assert_array_equal(got_re, want_re)
    np.testing.assert_array_equal(got_im, want_im)


def _gather2d_runtime_cols_module(n):
    """A banded gather2d whose row coordinate is provably the output row and
    whose column coordinate arrives as a kernel argument, so the test can
    drive values no MLIR literal can spell."""
    iota = _dense(np.arange(n, dtype=np.float64))
    return f"""
module {{
  func.func @g2d(%re: tensor<{n}x{n}xf64>, %im: tensor<{n}x{n}xf64>,
                 %cols: tensor<{n}x{n}xf64>)
      -> (tensor<{n}x{n}xf64>, tensor<{n}x{n}xf64>) {{
    %iota = sar.constant dense<{iota}> : tensor<{n}xf64>
    %rows = sar.broadcast %iota {{dim = 0 : i64}}
        : tensor<{n}xf64> -> tensor<{n}x{n}xf64>
    %or, %oi = sar.gather2d_split %re, %im, %rows, %cols
        {{boundary = "edge"}}
        : (tensor<{n}x{n}xf64>, tensor<{n}x{n}xf64>, tensor<{n}x{n}xf64>,
           tensor<{n}x{n}xf64>)
        -> (tensor<{n}x{n}xf64>, tensor<{n}x{n}xf64>)
    return %or, %oi : tensor<{n}x{n}xf64>, tensor<{n}x{n}xf64>
  }}
}}
"""


def test_banded_gather2d_sanitizes_unindexable_positions(tmp_path):
    """A position that cannot become an index reads as zero, on the banded
    path as much as the full-plane one.

    `edge` clamps rather than masking, so an unsanitized floor hands the
    garbage index to the clamp and returns a real sample -- or a NaN. Only
    the two paths agreeing keeps one operation with one meaning.
    """
    n = 16
    mlir = _gather2d_runtime_cols_module(n)

    (tmp_path / "b").mkdir()
    (tmp_path / "f").mkdir()
    _, fn_banded = _run(mlir, "b", tmp_path, banded=True, top="g2d")
    _, fn_full = _run(mlir, "f", tmp_path, banded=False, top="g2d")

    rng = np.random.default_rng(11)
    data = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    re = np.ascontiguousarray(data.real)
    im = np.ascontiguousarray(data.imag)
    cols = np.tile(np.arange(n, dtype=np.float64), (n, 1))
    specials = [np.nan, np.inf, -np.inf, 1e300]
    cols[0, :len(specials)] = specials
    cols = np.ascontiguousarray(cols)

    outs = [(n, n), (n, n)]
    got_re, got_im = _run_split(fn_banded, [re, im, cols], outs, np.float64)
    want_re, want_im = _run_split(fn_full, [re, im, cols], outs, np.float64)
    np.testing.assert_array_equal(got_re, want_re)
    np.testing.assert_array_equal(got_im, want_im)
    # And the defined answer is zero, not a clamped neighbour.
    assert np.array_equal(got_re[0, :len(specials)], np.zeros(len(specials)))
    assert np.array_equal(got_im[0, :len(specials)], np.zeros(len(specials)))
