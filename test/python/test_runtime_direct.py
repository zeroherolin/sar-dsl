"""Direct unit tests for libsar_runtime via ctypes.

These call the runtime's C ABI (`_mlir_ciface_sar_rt_*`) without going
through the compiler, so a numerical regression in the FFT or
interpolation kernels is attributed to the runtime rather than to a
lowering. Shape contracts mirror the IR verifiers: `positions` and the
output buffer always have the same shape as the input data.
"""

import ctypes
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from conftest import requires_cpu

pytestmark = requires_cpu


@pytest.fixture(scope="module")
def runtime_lib():
    """The loaded libsar_runtime shared object."""
    from sar.compiler.toolchain import find_runtime_library
    return ctypes.CDLL(find_runtime_library())


@pytest.fixture(scope="module")
def descriptor():
    """Factory building strided memref descriptors for numpy arrays."""
    from sar.runtime import _make_descriptor
    return _make_descriptor


def _call_fft(runtime_lib, descriptor, x, dim, inverse, symbol):
    out = np.empty_like(x)
    fn = getattr(runtime_lib, symbol)
    fn.restype = None
    in_desc = descriptor(x)
    out_desc = descriptor(out)
    fn(ctypes.byref(in_desc), ctypes.byref(out_desc), ctypes.c_int64(dim),
       ctypes.c_bool(inverse))
    return out


def _random_complex(rng, shape, dtype):
    data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return data.astype(dtype)


# --------------------------------------------------------------------------- #
# FFT
# --------------------------------------------------------------------------- #


def test_fft_1d_radix2(runtime_lib, descriptor):
    """Power-of-two 1-D forward FFT matches numpy (unscaled)."""
    rng = np.random.default_rng(0)
    x = _random_complex(rng, (16, ), np.complex128)
    out = _call_fft(runtime_lib, descriptor, x, 0, False,
                    "_mlir_ciface_sar_rt_fft_1d_c128")
    np.testing.assert_allclose(out, np.fft.fft(x), rtol=1e-12, atol=1e-12)


def test_ifft_1d_scaling(runtime_lib, descriptor):
    """Inverse FFT carries the 1/N scaling (numpy convention)."""
    rng = np.random.default_rng(1)
    x = _random_complex(rng, (8, ), np.complex128)
    out = _call_fft(runtime_lib, descriptor, x, 0, True,
                    "_mlir_ciface_sar_rt_fft_1d_c128")
    np.testing.assert_allclose(out, np.fft.ifft(x), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [12, 33, 97, 100])
def test_fft_1d_bluestein(runtime_lib, descriptor, n):
    """Non-power-of-two sizes go through Bluestein's chirp-z reduction."""
    rng = np.random.default_rng(n)
    x = _random_complex(rng, (n, ), np.complex128)
    out = _call_fft(runtime_lib, descriptor, x, 0, False,
                    "_mlir_ciface_sar_rt_fft_1d_c128")
    np.testing.assert_allclose(out, np.fft.fft(x), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dim", [0, 1])
def test_fft_2d_along_dim(runtime_lib, descriptor, dim):
    """2-D FFT transforms only the selected axis."""
    rng = np.random.default_rng(10 + dim)
    x = _random_complex(rng, (4, 8), np.complex128)
    out = _call_fft(runtime_lib, descriptor, x, dim, False,
                    "_mlir_ciface_sar_rt_fft_2d_c128")
    np.testing.assert_allclose(out,
                               np.fft.fft(x, axis=dim),
                               rtol=1e-12,
                               atol=1e-12)


def test_fft_2d_c64_precision(runtime_lib, descriptor):
    """The c64 entry point computes in double and rounds on store."""
    rng = np.random.default_rng(3)
    x = _random_complex(rng, (4, 8), np.complex64)
    out = _call_fft(runtime_lib, descriptor, x, 1, False,
                    "_mlir_ciface_sar_rt_fft_2d_c64")
    expected = np.fft.fft(x.astype(np.complex128), axis=1)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_fft_roundtrip(runtime_lib, descriptor):
    """ifft(fft(x)) == x."""
    rng = np.random.default_rng(4)
    x = _random_complex(rng, (16, ), np.complex128)
    spectrum = _call_fft(runtime_lib, descriptor, x, 0, False,
                         "_mlir_ciface_sar_rt_fft_1d_c128")
    restored = _call_fft(runtime_lib, descriptor, spectrum, 0, True,
                         "_mlir_ciface_sar_rt_fft_1d_c128")
    np.testing.assert_allclose(restored, x, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #

_KERNEL_IDS = {"nearest": 0, "linear": 1, "cubic": 2, "sinc": 3}
_WINDOW_HANN = 1
_BOUNDARY_IDS = {"zero": 0, "edge": 1, "reflect": 2}


def _call_interp(runtime_lib,
                 descriptor,
                 data,
                 positions,
                 kernel,
                 taps,
                 boundary="zero"):
    out = np.empty_like(data)
    symbol = ("_mlir_ciface_sar_rt_interp1d_2d_c64" if data.dtype
              == np.complex64 else "_mlir_ciface_sar_rt_interp1d_2d_c128")
    fn = getattr(runtime_lib, symbol)
    fn.restype = None
    data_desc = descriptor(data)
    pos_desc = descriptor(positions)
    out_desc = descriptor(out)
    fn(ctypes.byref(data_desc), ctypes.byref(pos_desc), ctypes.byref(out_desc),
       ctypes.c_int64(_KERNEL_IDS[kernel]), ctypes.c_int64(taps),
       ctypes.c_int64(_WINDOW_HANN), ctypes.c_double(2.5),
       ctypes.c_int64(_BOUNDARY_IDS[boundary]))
    return out


@pytest.mark.parametrize("kernel,taps", [
    ("nearest", 1),
    ("linear", 2),
    ("cubic", 4),
    ("sinc", 8),
    ("sinc", 16),
])
def test_interp1d_identity_at_integer_positions(runtime_lib, descriptor,
                                                kernel, taps):
    """Every kernel reproduces the input when sampled on the grid."""
    rng = np.random.default_rng(hash(kernel) % 1000 + taps)
    shape = (4, 16)
    data = _random_complex(rng, shape, np.complex128)
    positions = np.tile(np.arange(shape[1], dtype=np.float64), (shape[0], 1))

    out = _call_interp(runtime_lib, descriptor, data, positions, kernel, taps)

    # Interior samples are exact; edges lose taps that fall out of range.
    guard = taps // 2
    inner = slice(guard, shape[1] - guard) if guard else slice(None)
    np.testing.assert_allclose(out[:, inner],
                               data[:, inner],
                               rtol=1e-10,
                               atol=1e-10)


def test_interp1d_linear_halfway(runtime_lib, descriptor):
    """Linear interpolation at half-sample offsets averages neighbours."""
    data = np.arange(8, dtype=np.float64).astype(np.complex128)
    data = data.reshape(1, 8)
    positions = (np.arange(8, dtype=np.float64) + 0.5).reshape(1, 8)

    out = _call_interp(runtime_lib, descriptor, data, positions, "linear", 2)

    # out[j] = (data[j] + data[j+1]) / 2 for j < 7; the last tap runs off the
    # end and contributes zero.
    expected = np.empty((1, 8), dtype=np.complex128)
    expected[0, :7] = (data[0, :7] + data[0, 1:]) / 2.0
    expected[0, 7] = data[0, 7] / 2.0
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_interp1d_out_of_range_taps_are_zero(runtime_lib, descriptor):
    """Positions far outside the row yield zero, not garbage."""
    data = np.ones((1, 8), dtype=np.complex128)
    positions = np.full((1, 8), -50.0)
    out = _call_interp(runtime_lib, descriptor, data, positions, "sinc", 8)
    np.testing.assert_allclose(out, np.zeros_like(out), atol=1e-12)


def test_interp1d_fractional_positions_bounded(runtime_lib, descriptor):
    """Half-sample sinc resampling stays finite and bounded."""
    rng = np.random.default_rng(7)
    shape = (2, 16)
    data = _random_complex(rng, shape, np.complex128)
    positions = np.tile(
        np.arange(shape[1], dtype=np.float64) + 0.5, (shape[0], 1))

    out = _call_interp(runtime_lib, descriptor, data, positions, "sinc", 8)

    interior = out[:, 2:-2]
    assert np.isfinite(interior).all()
    assert np.abs(interior).max() < 10.0 * np.abs(data).max()


def test_interp1d_edge_boundary_at_runtime(runtime_lib, descriptor):
    """Edge boundary policy clamps out-of-range taps to the endpoint."""
    data = np.arange(8, dtype=np.complex128).reshape(1, 8)
    positions = np.array([[-1.0, 0.0, 7.0, 8.0, 3.0, 4.0, 5.0, 6.0]])

    out = _call_interp(runtime_lib,
                       descriptor,
                       data,
                       positions,
                       "nearest",
                       1,
                       boundary="edge")
    # nearest rounds to -1, 0, 7, 8, 3, 4, 5, 6; edge clamps into [0, 7].
    np.testing.assert_allclose(out, [[0, 0, 7, 7, 3, 4, 5, 6]],
                               rtol=1e-12,
                               atol=1e-12)


def test_interp1d_reflect_boundary_at_runtime(runtime_lib, descriptor):
    """Reflect boundary policy mirrors out-of-range taps back inside."""
    data = np.arange(8, dtype=np.complex128).reshape(1, 8)
    positions = np.array([[-1.0, -2.0, 8.0, 9.0, 3.0, 4.0, 5.0, 6.0]])

    out = _call_interp(runtime_lib,
                       descriptor,
                       data,
                       positions,
                       "nearest",
                       1,
                       boundary="reflect")
    # nearest rounds to -1, -2, 8, 9; mirrored to 0, 1, 7, 6.
    np.testing.assert_allclose(out, [[0, 1, 7, 6, 3, 4, 5, 6]],
                               rtol=1e-12,
                               atol=1e-12)


# --------------------------------------------------------------------------- #
# Thread configuration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("env_var", ["SAR_RT_NUM_THREADS", "OMP_NUM_THREADS"])
def test_runtime_thread_env_is_honoured(env_var):
    """The worker count is read once from the environment.

    The value is cached in a function-local static on the C++ side, so this
    has to run in a fresh process to be meaningful.
    """
    script = textwrap.dedent("""
        import ctypes
        import numpy as np
        from sar.compiler.toolchain import find_runtime_library
        from sar.runtime import _make_descriptor

        lib = ctypes.CDLL(find_runtime_library())
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((64, 64))
             + 1j * rng.standard_normal((64, 64))).astype(np.complex128)
        out = np.empty_like(x)
        fn = lib._mlir_ciface_sar_rt_fft_2d_c128
        fn.restype = None
        a, b = _make_descriptor(x), _make_descriptor(out)
        fn(ctypes.byref(a), ctypes.byref(b),
           ctypes.c_int64(1), ctypes.c_bool(False))
        assert np.allclose(out, np.fft.fft(x, axis=1), rtol=1e-12, atol=1e-12)
        print("OK")
    """)
    import os
    env = dict(os.environ, PYTHONPATH="python", **{env_var: "2"})
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True,
                          text=True,
                          env=env)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
