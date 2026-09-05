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

from conftest import REPO_ROOT, requires_cpu

pytestmark = requires_cpu


@pytest.fixture(scope="module")
def runtime_lib():
    """The loaded libsar_runtime shared object."""
    from sar.compiler.toolchain import find_runtime_library
    return ctypes.CDLL(find_runtime_library())


@pytest.fixture(scope="module")
def descriptor():
    """Factory building strided memref descriptors for numpy arrays."""
    from sar.runtime import make_descriptor
    return make_descriptor


def test_compiled_kernel_wraps_library_load_errors(tmp_path):
    from sar.errors import LaunchError
    from sar.runtime import CompiledKernel

    invalid = tmp_path / "not-a-library.so"
    invalid.write_text("not an ELF library")
    with pytest.raises(LaunchError,
                       match="cannot load compiled kernel") as err:
        CompiledKernel(str(invalid), "missing", [], [])
    assert isinstance(err.value.__cause__, OSError)


def test_compiled_kernel_wraps_missing_symbols():
    from sar.compiler.toolchain import find_runtime_library
    from sar.errors import LaunchError
    from sar.runtime import CompiledKernel

    with pytest.raises(LaunchError, match="_mlir_ciface_missing") as err:
        CompiledKernel(find_runtime_library(), "missing", [], [])
    assert isinstance(err.value.__cause__, AttributeError)


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


def test_runtime_fft_accepts_strided_and_negative_stride_memrefs(
        runtime_lib, descriptor):
    rng = np.random.default_rng(55)
    base = _random_complex(rng, (8, 16), np.complex128)
    for view in (base[:, ::2], base[:, ::-2]):
        out = _call_fft(runtime_lib, descriptor, view, 1, False,
                        "_mlir_ciface_sar_rt_fft_2d_c128")
        np.testing.assert_allclose(out,
                                   np.fft.fft(view, axis=1),
                                   rtol=1e-12,
                                   atol=1e-12)


def test_fft_roundtrip(runtime_lib, descriptor):
    """ifft(fft(x)) == x."""
    rng = np.random.default_rng(4)
    x = _random_complex(rng, (16, ), np.complex128)
    spectrum = _call_fft(runtime_lib, descriptor, x, 0, False,
                         "_mlir_ciface_sar_rt_fft_1d_c128")
    restored = _call_fft(runtime_lib, descriptor, spectrum, 0, True,
                         "_mlir_ciface_sar_rt_fft_1d_c128")
    np.testing.assert_allclose(restored, x, rtol=1e-12, atol=1e-12)


def test_fft_plan_cache_reuses_lengths_and_can_be_cleared(
        runtime_lib, descriptor):
    from sar.runtime import clear_fft_plan_cache, fft_plan_cache_size

    clear_fft_plan_cache()
    assert fft_plan_cache_size() == 0
    rng = np.random.default_rng(44)
    for n in (12, 12, 16):
        x = _random_complex(rng, (n, ), np.complex128)
        _call_fft(runtime_lib, descriptor, x, 0, False,
                  "_mlir_ciface_sar_rt_fft_1d_c128")
    assert fft_plan_cache_size() == 2
    clear_fft_plan_cache()
    assert fft_plan_cache_size() == 0


def test_runtime_thread_config_respects_affinity():
    from sar.runtime import thread_config

    config = thread_config()
    assert 1 <= config["runtime_workers"] <= config["available_workers"]


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


@pytest.mark.parametrize("kernel,taps,seed", [
    ("nearest", 1, 101),
    ("linear", 2, 102),
    ("cubic", 4, 103),
    ("sinc", 8, 104),
    ("sinc", 16, 105),
])
def test_interp1d_identity_at_integer_positions(runtime_lib, descriptor,
                                                kernel, taps, seed):
    """Every kernel reproduces the input when sampled on the grid."""
    rng = np.random.default_rng(seed)
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
    """Reflect repeats for positions arbitrarily far beyond either edge."""
    data = np.arange(8, dtype=np.complex128).reshape(1, 8)
    positions = np.array([[-1.0, -2.0, -5.0, -8.0, 8.0, 9.0, 14.0, 16.0]])

    out = _call_interp(runtime_lib,
                       descriptor,
                       data,
                       positions,
                       "nearest",
                       1,
                       boundary="reflect")
    np.testing.assert_allclose(out, [[0, 1, 4, 7, 7, 6, 1, 0]],
                               rtol=1e-12,
                               atol=1e-12)


def test_runtime_rejects_negative_allocations():
    script = textwrap.dedent("""
        import ctypes
        from sar.compiler.toolchain import find_runtime_library

        lib = ctypes.CDLL(find_runtime_library())
        alloc = lib._mlir_memref_to_llvm_alloc
        alloc.argtypes = [ctypes.c_int64]
        alloc.restype = ctypes.c_void_p
        alloc(-1)
    """)
    import os
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "python"))
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True,
                          text=True,
                          env=env,
                          timeout=30)
    assert proc.returncode < 0
    assert "allocation size must be non-negative" in proc.stderr


def test_runtime_rejects_double_free():
    script = textwrap.dedent("""
        import ctypes
        from sar.compiler.toolchain import find_runtime_library

        lib = ctypes.CDLL(find_runtime_library())
        alloc = lib._mlir_memref_to_llvm_alloc
        alloc.argtypes = [ctypes.c_int64]
        alloc.restype = ctypes.c_void_p
        release = lib._mlir_memref_to_llvm_free
        release.argtypes = [ctypes.c_void_p]
        ptr = alloc(1 << 20)
        release(ptr)
        release(ptr)
    """)
    import os
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "python"))
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True,
                          text=True,
                          env=env,
                          timeout=30)
    assert proc.returncode < 0
    assert "allocation released twice" in proc.stderr


def test_runtime_fft_descriptor_errors_are_recoverable(runtime_lib,
                                                       descriptor):
    from sar.compiler.toolchain import find_runtime_library

    lib = ctypes.CDLL(find_runtime_library())
    clear = lib.sar_rt_error_clear
    clear.restype = None
    get = lib.sar_rt_error_get
    get.restype = ctypes.c_char_p
    x = np.ones(8, dtype=np.complex128)
    out = np.empty_like(x)
    a, b = descriptor(x), descriptor(out)
    b.sizes[0] = 7
    fn = lib._mlir_ciface_sar_rt_fft_1d_c128
    fn.restype = None
    clear()
    fn(ctypes.byref(a), ctypes.byref(b), ctypes.c_int64(0),
       ctypes.c_bool(False))
    assert b"output shape must match" in get()
    # Process and runtime remain usable after the rejected call.
    clear()
    got = _call_fft(runtime_lib, descriptor, x, 0, False,
                    "_mlir_ciface_sar_rt_fft_1d_c128")
    np.testing.assert_allclose(got, np.fft.fft(x), rtol=1e-12, atol=1e-12)


def test_runtime_interp_enum_errors_are_recoverable(runtime_lib, descriptor):
    lib = runtime_lib
    clear = lib.sar_rt_error_clear
    clear.restype = None
    get = lib.sar_rt_error_get
    get.restype = ctypes.c_char_p
    data = np.ones((1, 8), dtype=np.complex128)
    positions = np.arange(8, dtype=np.float64).reshape(1, 8)
    out = np.empty_like(data)
    descriptors = [descriptor(x) for x in (data, positions, out)]
    fn = lib._mlir_ciface_sar_rt_interp1d_2d_c128
    fn.restype = None
    clear()
    fn(*[ctypes.byref(x) for x in descriptors], ctypes.c_int64(99),
       ctypes.c_int64(8), ctypes.c_int64(1), ctypes.c_double(2.5),
       ctypes.c_int64(0))
    assert b"unknown kernel" in get()


def test_runtime_preserves_first_error_and_skips_later_calls(
        runtime_lib, descriptor):
    lib = runtime_lib
    clear = lib.sar_rt_error_clear
    clear.restype = None
    get = lib.sar_rt_error_get
    get.restype = ctypes.c_char_p
    x = np.ones(8, dtype=np.complex128)
    first_out = np.full_like(x, 11)
    second_out = np.full_like(x, 22)
    a, first, second = descriptor(x), descriptor(first_out), descriptor(
        second_out)
    first.sizes[0] = 7
    fn = lib._mlir_ciface_sar_rt_fft_1d_c128
    fn.restype = None
    clear()
    fn(ctypes.byref(a), ctypes.byref(first), ctypes.c_int64(0),
       ctypes.c_bool(False))
    initial = get()
    fn(ctypes.byref(a), ctypes.byref(second), ctypes.c_int64(0),
       ctypes.c_bool(False))
    assert get() == initial
    np.testing.assert_array_equal(second_out, np.full_like(x, 22))


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
        from sar.runtime import make_descriptor

        lib = ctypes.CDLL(find_runtime_library())
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((64, 64))
             + 1j * rng.standard_normal((64, 64))).astype(np.complex128)
        out = np.empty_like(x)
        fn = lib._mlir_ciface_sar_rt_fft_2d_c128
        fn.restype = None
        a, b = make_descriptor(x), make_descriptor(out)
        fn(ctypes.byref(a), ctypes.byref(b),
           ctypes.c_int64(1), ctypes.c_bool(False))
        assert np.allclose(out, np.fft.fft(x, axis=1), rtol=1e-12, atol=1e-12)
        print("OK")
    """)
    import os
    env = dict(os.environ,
               PYTHONPATH=str(REPO_ROOT / "python"),
               **{env_var: "2"})
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True,
                          text=True,
                          env=env,
                          timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_openmp_binding_does_not_shrink_runtime_pool():
    """Pool sizing uses load-time process affinity, not a bound OMP thread."""
    script = textwrap.dedent("""
        import os
        # runtimeWorkerCount() caps SAR_RT_NUM_THREADS at the load-time
        # affinity count, so snapshot the mask before the runtime loads.
        expected_workers = min(8, len(os.sched_getaffinity(0)))

        import numpy as np
        import sar
        from sar.runtime import thread_config

        @sar.func
        def scale(x: sar.f64[4096]) -> sar.f64[4096]:
            return x * 2.0

        scale(np.ones(4096))  # initialize libomp and apply close binding
        config = thread_config()
        assert config["available_workers"] > 2, config
        assert config["runtime_workers"] == expected_workers, config
        x = np.ones((64, 64), dtype=np.complex128)
        import ctypes
        from sar.compiler.toolchain import find_runtime_library
        from sar.runtime import make_descriptor
        lib = ctypes.CDLL(find_runtime_library())
        fn = lib._mlir_ciface_sar_rt_fft_2d_c128
        out = np.empty_like(x)
        a, b = make_descriptor(x), make_descriptor(out)
        fn(ctypes.byref(a), ctypes.byref(b), ctypes.c_int64(1),
           ctypes.c_bool(False))
        task_mask_counts = []
        for path in __import__("pathlib").Path("/proc/self/task").iterdir():
            with (path / "status").open() as handle:
                line = next(line for line in handle
                            if line.startswith("Cpus_allowed_list:"))
            count = 0
            for span in line.split(":", 1)[1].strip().split(","):
                bounds = [int(value) for value in span.split("-")]
                count += bounds[-1] - bounds[0] + 1
            task_mask_counts.append(count)
        assert (task_mask_counts.count(config["available_workers"])
                >= expected_workers - 1), task_mask_counts
        print(config)
    """)
    import os
    env = dict(os.environ,
               PYTHONPATH=str(REPO_ROOT / "python"),
               SAR_RT_NUM_THREADS="8",
               OMP_NUM_THREADS="8",
               OMP_PROC_BIND="close",
               OMP_PLACES="cores")
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True,
                          text=True,
                          env=env,
                          timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_runtime_thread_pool_is_reused_and_concurrent_calls_are_safe():
    script = textwrap.dedent("""
        import ctypes
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        from sar.compiler.toolchain import find_runtime_library
        from sar.runtime import make_descriptor

        lib = ctypes.CDLL(find_runtime_library())
        fn = lib._mlir_ciface_sar_rt_fft_2d_c128
        fn.restype = None
        rng = np.random.default_rng(17)
        inputs = [
            (rng.standard_normal((32, 32))
             + 1j * rng.standard_normal((32, 32))).astype(np.complex128)
            for _ in range(2)
        ]

        def threads():
            with open("/proc/self/status") as handle:
                line = next(line for line in handle
                            if line.startswith("Threads:"))
            return int(line.split()[1])

        np.fft.fft(inputs[0], axis=1)
        baseline = threads()
        def run(index):
            for _ in range(20):
                out = np.empty_like(inputs[index])
                a, b = make_descriptor(inputs[index]), make_descriptor(out)
                fn(ctypes.byref(a), ctypes.byref(b),
                   ctypes.c_int64(1), ctypes.c_bool(False))
                assert np.allclose(out, np.fft.fft(inputs[index], axis=1),
                                   rtol=1e-12, atol=1e-12)

        first = np.empty_like(inputs[0])
        a, b = make_descriptor(inputs[0]), make_descriptor(first)
        fn(ctypes.byref(a), ctypes.byref(b),
           ctypes.c_int64(1), ctypes.c_bool(False))
        pooled = threads()
        def concurrent_batch():
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(run, index) for index in range(2)]
                for future in futures:
                    future.result()

        concurrent_batch()
        after_first = threads()
        concurrent_batch()
        assert threads() <= after_first + 1
        assert pooled >= baseline
        print("OK")
    """)
    import os
    env = dict(os.environ,
               PYTHONPATH=str(REPO_ROOT / "python"),
               SAR_RT_NUM_THREADS="4")
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True,
                          text=True,
                          env=env,
                          timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
