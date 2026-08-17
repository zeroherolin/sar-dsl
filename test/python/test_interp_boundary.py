"""sar.interp1d boundary policies: `zero`, `edge` and `reflect` validated
against an explicit numpy oracle built independently of the DSL."""

import numpy as np
import pytest

import sar

from conftest import requires_cpu

pytestmark = requires_cpu


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


def _window(window, t, beta):
    if window == "rect":
        return 1.0
    if window == "hann":
        return 0.5 + 0.5 * np.cos(np.pi * t)
    if window == "hamming":
        return 0.54 + 0.46 * np.cos(np.pi * t)
    return np.i0(beta * np.sqrt(max(1.0 - t * t, 0.0))) / np.i0(beta)


def _boundary_oracle(data,
                     positions,
                     boundary,
                     kernel="sinc",
                     taps=8,
                     window="hann",
                     beta=2.5):
    """Scalar reference: for every output sample, walk the tap support and
    resolve each tap through the boundary policy before weighting it."""
    n, m = data.shape
    out = np.zeros_like(data)
    for i in range(n):
        for j in range(m):
            pos = positions[i, j]
            if kernel == "nearest":
                resolved = _resolve(int(np.floor(pos + 0.5)), m, boundary)
                out[i, j] = 0 if resolved is None else data[i, resolved]
                continue

            idx0 = int(np.floor(pos))
            frac = pos - idx0
            if kernel == "linear":
                offsets = [0, 1]
                weights = [1.0 - frac, frac]
            elif kernel == "cubic":
                s, s2, s3 = frac, frac**2, frac**3
                offsets = [-1, 0, 1, 2]
                weights = [
                    -0.5 * s3 + s2 - 0.5 * s, 1.5 * s3 - 2.5 * s2 + 1.0,
                    -1.5 * s3 + 2.0 * s2 + 0.5 * s, 0.5 * s3 - 0.5 * s2
                ]
            else:
                half = taps // 2
                offsets = list(range(1 - half, half + 1))
                # The kernel stays centred on the unresolved tap: the policy
                # changes which sample is fetched, not the weight it carries.
                weights = [
                    np.sinc(pos - (idx0 + k)) *
                    _window(window, (pos - (idx0 + k)) / half, beta)
                    for k in offsets
                ]

            acc = 0.0 + 0.0j
            for k, w in zip(offsets, weights):
                resolved = _resolve(idx0 + k, m, boundary)
                if resolved is not None:
                    acc += data[i, resolved] * w
            out[i, j] = acc
    return out


@pytest.mark.parametrize("boundary", ["zero", "edge", "reflect"])
def test_interp1d_boundary_policy_sinc(boundary):
    """Every policy matches the oracle for the default sinc kernel, with
    positions that run off both ends of the row."""
    n, m = 8, 32
    rng = np.random.default_rng(17)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    positions = rng.uniform(-3.0, m + 2.0, size=(n, m))

    @sar.func
    def kernel(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
        return sar.interp1d(d, p, boundary=boundary)

    np.testing.assert_allclose(kernel(data, positions),
                               _boundary_oracle(data, positions, boundary),
                               rtol=1e-11,
                               atol=1e-11)


@pytest.mark.parametrize("boundary", ["zero", "edge", "reflect"])
@pytest.mark.parametrize("kernel", ["nearest", "linear", "cubic"])
def test_interp1d_boundary_policy_all_kernels(boundary, kernel):
    """The policy is orthogonal to the kernel: every combination matches."""
    n, m = 6, 24
    rng = np.random.default_rng(23)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    positions = rng.uniform(-2.0, m + 1.5, size=(n, m))

    @sar.func
    def k(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
        return sar.interp1d(d, p, kernel=kernel, boundary=boundary)

    np.testing.assert_allclose(k(data, positions),
                               _boundary_oracle(data,
                                                positions,
                                                boundary,
                                                kernel=kernel),
                               rtol=1e-11,
                               atol=1e-11)


@pytest.mark.parametrize("boundary", ["zero", "edge", "reflect"])
def test_interp1d_boundary_policy_kaiser(boundary):
    """A non-default sinc taper still honours the policy."""
    n, m = 4, 20
    rng = np.random.default_rng(29)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    positions = rng.uniform(-3.0, m + 2.0, size=(n, m))

    @sar.func
    def k(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
        return sar.interp1d(d,
                            p,
                            taps=16,
                            window="kaiser",
                            beta=4.0,
                            boundary=boundary)

    np.testing.assert_allclose(k(data, positions),
                               _boundary_oracle(data,
                                                positions,
                                                boundary,
                                                taps=16,
                                                window="kaiser",
                                                beta=4.0),
                               rtol=1e-11,
                               atol=1e-11)


def test_interp1d_edge_clamps_to_endpoint():
    """Hand-checked: `edge` resolves every out-of-range tap to the nearest
    end sample."""
    data = np.arange(8, dtype=np.complex128).reshape(1, 8)
    positions = np.array([[-1.0, -0.5, 0.0, 3.0, 7.0, 7.5, 8.0, 9.0]])

    @sar.func
    def k(d: sar.c128[1, 8], p: sar.f64[1, 8]) -> sar.c128[1, 8]:
        return sar.interp1d(d, p, kernel="nearest", boundary="edge")

    # round(pos) = -1, 0, 0, 3, 7, 8, 8, 9 -> clamped into [0, 7].
    np.testing.assert_allclose(k(data, positions), [[0, 0, 0, 3, 7, 7, 7, 7]],
                               rtol=1e-12,
                               atol=1e-12)


def test_interp1d_reflect_mirrors_about_boundary():
    """Hand-checked: `reflect` mirrors an out-of-range tap back inside."""
    data = np.array([[1, 2, 3, 4]], dtype=np.complex128)
    positions = np.array([[-1.0, -0.5, 4.0, 4.5]])

    @sar.func
    def k(d: sar.c128[1, 4], p: sar.f64[1, 4]) -> sar.c128[1, 4]:
        return sar.interp1d(d, p, kernel="nearest", boundary="reflect")

    # round(pos) = -1, 0, 4, 5 -> mirrored to 0, 0, 3, 2.
    np.testing.assert_allclose(k(data, positions), [[1, 1, 4, 3]],
                               rtol=1e-12,
                               atol=1e-12)


def test_interp1d_default_boundary_is_zero():
    """The default is unchanged, so existing kernels keep zero-padding."""
    data = np.ones((1, 4), dtype=np.complex128)
    positions = np.array([[-1.0, -2.0, 4.0, 5.0]])

    @sar.func
    def k(d: sar.c128[1, 4], p: sar.f64[1, 4]) -> sar.c128[1, 4]:
        return sar.interp1d(d, p, kernel="nearest")

    np.testing.assert_allclose(k(data, positions),
                               np.zeros((1, 4)),
                               atol=1e-12)


def test_interp1d_policies_agree_in_the_interior():
    """Away from the edges no tap leaves the row, so the policy cannot
    change the result."""
    n, m = 4, 32
    rng = np.random.default_rng(31)
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    positions = rng.uniform(8.0, m - 8.0, size=(n, m))

    def run(boundary):

        @sar.func
        def k(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
            return sar.interp1d(d, p, boundary=boundary)

        return k(data, positions)

    base = run("zero")
    np.testing.assert_allclose(run("edge"), base, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(run("reflect"), base, rtol=1e-12, atol=1e-12)


def test_interp1d_boundary_validation():
    """An unknown policy is rejected at trace time."""
    n, m = 4, 8

    from sar.language import TraceError
    with pytest.raises(TraceError, match="boundary must be one of"):

        @sar.func
        def k(d: sar.c128[n, m], p: sar.f64[n, m]) -> sar.c128[n, m]:
            return sar.interp1d(d, p, boundary="wrap")

        k.to_mlir()
