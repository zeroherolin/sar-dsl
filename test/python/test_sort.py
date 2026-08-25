"""sar.sort: validates full-line sorting against numpy's sort on both
backends, covering rank-1 and rank-2 inputs and both axes."""

import numpy as np
import pytest

import sar

from conftest import requires_cpu, requires_hls


@requires_cpu
@pytest.mark.parametrize("extent", [1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 33, 64])
def test_sort_is_exact_at_every_extent(extent):
    """The network is a schedule of comparators derived from the extent, so
    the extents that are not powers of two are the ones worth pinning: a
    partner past the end must simply not pair, with no padding value
    reaching the result."""
    rows = 3
    rng = np.random.default_rng(97 + extent)
    data = rng.uniform(-10.0, 10.0, size=(rows, extent))

    @sar.func
    def kernel(x: sar.f64[rows, extent]) -> sar.f64[rows, extent]:
        return sar.sort(x, axis=1)

    # Bit-exact: a compare-exchange network only ever moves values, so
    # anything short of equality would mean the schedule is wrong.
    np.testing.assert_array_equal(kernel(data), np.sort(data, axis=1))


@requires_cpu
def test_sort_places_nans_like_numpy():
    """NaNs sort to the end, as numpy does. The comparator treats a NaN
    partner as the larger element, which is what carries them there."""
    rng = np.random.default_rng(23)
    data = rng.uniform(-5.0, 5.0, size=(2, 13))
    data[0, 0] = np.nan
    data[0, 7] = np.nan
    data[1, 6] = np.nan

    @sar.func
    def kernel(x: sar.f64[2, 13]) -> sar.f64[2, 13]:
        return sar.sort(x, axis=1)

    result = kernel(data)
    expected = np.sort(data, axis=1)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(result[~np.isnan(result)],
                                  expected[~np.isnan(expected)])


@requires_cpu
@pytest.mark.parametrize("dim", [0, 1])
def test_sort_matches_numpy_cpu(dim):
    """Ascending sort along each axis matches numpy exactly."""
    n, m = 16, 32
    rng = np.random.default_rng(41)
    data = rng.uniform(-10.0, 10.0, size=(n, m))
    data[0, 0] = np.nan
    data[3, 7] = np.nan

    @sar.func
    def kernel(x: sar.f64[n, m]) -> sar.f64[n, m]:
        return sar.sort(x, axis=dim)

    np.testing.assert_allclose(kernel(data),
                               np.sort(data, axis=dim),
                               rtol=1e-14,
                               atol=1e-14,
                               equal_nan=True)


@requires_cpu
def test_sort_preserves_interior_order_on_ties():
    """Equal values come out equal: ordering ties is not observable, but
    the sorted line must still be exactly numpy's."""
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


@requires_cpu
def test_sort_rank1():
    """A rank-1 tensor sorts along its only axis, with and without `axis`."""
    rng = np.random.default_rng(44)
    data = rng.uniform(-10.0, 10.0, size=17)

    @sar.func
    def kernel(x: sar.f64[17]) -> sar.f64[17]:
        return sar.sort(x)

    @sar.func
    def kernel_axis0(x: sar.f64[17]) -> sar.f64[17]:
        return sar.sort(x, axis=0)

    expected = np.sort(data)
    np.testing.assert_allclose(kernel(data), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(kernel_axis0(data),
                               expected,
                               rtol=1e-14,
                               atol=1e-14)


@requires_hls
def test_sort_axis0_emits_on_hls():
    """The transposed dim=0 form emits through the HLS backend."""
    n, m = 8, 16

    @sar.func
    def kernel(x: sar.f32[n, m]) -> sar.f32[n, m]:
        return sar.sort(x, axis=0)

    kernel.name = "sort_hls_dim0"
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
