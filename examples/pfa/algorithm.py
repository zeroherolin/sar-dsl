"""The Polar Format Algorithm (PFA) with SVA sidelobe suppression.

Every stage is a `@sar.op` operator composed of built-in ops, including
two pieces that go beyond straight op chaining:

- the interpolation positions of both regrid passes are computed inside
  the kernel from the collection axes (`1/cos(theta)` outer products,
  `sar.atan2` to invert the pulse angle);
- spatially variant apodization (`sva_range`) is a data-dependent
  per-pixel weight selection, written with the `sar.where` / comparison
  primitives (the numpy `np.where` idiom).
"""

import sar

from .geometry import Geometry

__all__ = ["build_kernel", "make_inputs", "sva_range", "compress"]


@sar.op
def compress(grid):
    """Rectangular spectrum -> 2x-oversampled *baseband* complex image.

    The aperture is zero-padded symmetrically and ifftshift-ed so the
    band sits at DC: SVA below compares neighboring samples per I/Q
    component, which only makes sense without a carrier. Uniform
    weighting -- SVA replaces amplitude windowing.
    """
    n0, n1 = grid.shape
    padded = sar.pad(grid, ((n0 // 2, n0 // 2), (n1 // 2, n1 // 2)))
    return sar.fftshift(sar.ifft2(sar.ifftshift(padded)))


@sar.op
def _sva_component(center, q):
    """SVA on one I/Q component: choose the cosine-on-pedestal weight
    `a in [0, 0.5]` that minimizes `center + a * q` per pixel."""
    a = -center / sar.where(q == 0.0, 1.0, q)
    inside = sar.where(a <= 0.0, center, 0.0)  # null inside the range
    full = center + 0.5 * q  # clamp at full Hanning
    picked = sar.where(a >= 0.5, full, inside)
    return sar.where(q == 0.0, center, picked)


@sar.op
def sva_range(img):
    """Spatially variant apodization along range (Stankwitz): suppresses
    the sinc sidelobes of a uniformly weighted image without widening
    the mainlobe. On the 2x-oversampled image the neighbors sit one
    resolution cell away (+/- 2 samples); border columns pass through."""
    n = img.shape[1]
    center = img[:, 2:-2]
    q = img[:, :-4] + img[:, 4:]
    interior = sar.make_complex(_sva_component(center.real, q.real),
                                _sva_component(center.imag, q.imag))
    return sar.concatenate((img[:, 0:2], interior, img[:, n - 2:n]), dim=1)


@sar.op
def sva2d(img):
    """Separable 2-D SVA: the range pass followed by the same pass
    across azimuth (via corner turns), Stankwitz's sequential form."""
    return sar.transpose(sva_range(sar.transpose(sva_range(img))))


def build_kernel(n: int,
                 geom: Geometry,
                 name: str = "pfa",
                 dtype=sar.c128) -> sar.Kernel:
    """Builds the `n x n` PFA kernel for the given collection geometry.

    The geometry-dependent operators are defined here so the grid scalars
    (sample spacings, axis origins) bake into the IR at trace time, like
    the radar parameters of the sibling algorithms. `name` becomes the IR
    symbol and the HLS top function. `dtype` selects the data-path
    precision; the collection axes stay f64 either way, since they feed
    `interp1d` positions.
    """
    if dtype not in (sar.c128, sar.c64):
        raise ValueError("dtype must be sar.c128 or sar.c64")
    fd = sar.f64 if dtype is sar.c128 else sar.f32

    N = int(n)
    dk = geom.k[1] - geom.k[0]
    dtheta = geom.theta[1] - geom.theta[0]
    k0, theta0 = float(geom.k[0]), float(geom.theta[0])

    @sar.op
    def polar_positions(theta, u, v):
        """Fractional sample positions of both regrid passes, computed
        in-kernel from the axes.

        Pass 1 resamples each pulse along range: the radial sample where
        `k cos(theta_p) = v_j` sits at index `(v_j / cos(theta_p) - k0)
        / dk`. Pass 2 resamples across pulses: the pulse where
        `v_j tan(theta_p) = u_i` sits at `(atan2(u_i, v_j) - theta0) /
        dtheta`.
        """
        inv_cos = 1.0 / sar.cos(theta)
        k_wanted = (sar.broadcast(v, (N, N), dim=1) *
                    sar.broadcast(inv_cos, (N, N), dim=0))
        pos_range = (k_wanted - k0) * (1.0 / dk)

        theta_wanted = sar.atan2(sar.broadcast(u, (N, N), dim=0),
                                 sar.broadcast(v, (N, N), dim=1))
        pos_cross = (theta_wanted - theta0) * (1.0 / dtheta)
        return pos_range, pos_cross

    @sar.op
    def polar_to_rect(data, theta, u, v):
        """Polar -> rectangular regrid, the heart of PFA: one resampling
        pass along range, one across pulses."""
        pos_range, pos_cross = polar_positions(theta, u, v)
        keystone = sar.interp1d(data, pos_range, axis=1)
        return sar.interp1d(keystone, pos_cross, axis=0)

    def pfa(data: dtype[N, N]) -> (fd[2 * N, 2 * N], fd[2 * N, 2 * N]):
        # The collection axes are fixed by the geometry the kernel was
        # built for, so they bake in as constants (like the radar
        # parameters of the sibling algorithms) rather than being passed
        # as ports: an HLS design then carries 4 I/O planes instead of 7.
        image = compress(polar_to_rect(data, geom.theta, geom.u, geom.v))
        return abs(image), abs(sva2d(image))

    pfa.__name__ = name
    return sar.func(pfa)


def make_inputs(n: int, geom: Geometry):
    """Host-provided inputs: none -- the collection axes bake into the IR
    at trace time, so the kernel takes only the phase history."""
    return ()
