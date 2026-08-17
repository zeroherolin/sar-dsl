"""sar.sort: validates full-line sorting against numpy's sort on both
backends, covering rank-1/rank-2 normalization and both axes."""

import numpy as np
import pytest

import sar

from conftest import requires_cpu, requires_hls


@requires_cpu
@pytest.mark.parametrize("dim", [0, 1])
def test_sort_matches_numpy_cpu(dim):
    """Ascending sort along each axis matches numpy exactly."""
    n, m = 16, 32
    rng = np.random.default_rng(41)
    data = rng.uniform(-10.0, 10.0, size=(n, m))

    @sar.func
    def kernel(x: sar.f64[n, m]) -> sar.f64[n, m]:
        return sar.sort(x, axis=dim)

    np.testing.assert_allclose(kernel(data),
                               np.sort(data, axis=dim),
                               rtol=1e-14,
                               atol=1e-14)


@requires_cpu
def test_sort_preserves_interior_order_on_ties():
    """Stable: ties keep their relative order (implementation detail: bubble
    sort is stable)."""
    data = np.array([[3.0, 1.0, 1.0, 2.0]])

    @sar.func
    def kernel(x: sar.f64[1, 4]) -> sar.f64[1, 4]:
        return sar.sort(x, axis=1)

    result = kernel(data)
    # Ascending: [1, 1, 2, 3].
    np.testing.assert_allclose(result, [[1.0, 1.0, 2.0, 3.0]],
                               rtol=1e-14,
                               atol=1e-14)


@requires_cpu
def test_sort_f32_precision():
    """f32 tensors sort correctly without promoting to f64."""
    n, m = 8, 16
    rng = np.random.default_rng(43)
    data = rng.uniform(-5.0, 5.0, size=(n, m)).astype(np.float32)

    @sar.func
    def kernel(x: sar.f32[n, m]) -> sar.f32[n, m]:
        return sar.sort(x, axis=1)

    np.testing.assert_allclose(kernel(data),
                               np.sort(data, axis=1),
                               rtol=1e-6,
                               atol=1e-6)


@requires_hls
@pytest.mark.parametrize("dim", [0, 1])
def test_sort_emits_on_hls(dim):
    """The HLS backend lowers sort (emission only -- the backend symmetry
    gate likewise only compiles; numerical validation is the CPU tests
    above)."""
    n, m = 8, 16

    @sar.func
    def kernel(x: sar.f32[n, m]) -> sar.f32[n, m]:
        return sar.sort(x, axis=dim)

    kernel.name = f"sort_hls_dim{dim}"
    compiled = kernel.compile(backend="hls")
    assert "#pragma HLS" in compiled.source()


def test_sort_rejects_complex():
    """Ordering is undefined for complex, so complex inputs are rejected."""
    from sar.language import TraceError
    with pytest.raises(TraceError, match="expects a float tensor"):

        @sar.func
        def kernel(x: sar.c64[4, 8]) -> sar.c64[4, 8]:
            return sar.sort(x, axis=1)

        kernel.to_mlir()


def test_sort_rejects_bad_axis():
    """dim must be in [0, 1] for rank-2 tensors."""
    from sar.language import TraceError
    with pytest.raises(TraceError, match="dim must be 0 or 1"):

        @sar.func
        def kernel(x: sar.f64[4, 8]) -> sar.f64[4, 8]:
            return sar.sort(x, axis=2)

        kernel.to_mlir()
