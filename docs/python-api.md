# Python API reference

This is a compact map of the public `sar` package. Semantics follow NumPy
unless a note says otherwise; function docstrings carry the details, and
the exact IR-level semantics live in
[dialect.md](dialect.md). `dim=` and `axis=` are interchangeable
everywhere; indexing is 0-based ([MATLAB guide](matlab-users.md)).

## Types

`sar.f32`, `sar.f64`, `sar.i32`, `sar.i64`, `sar.c64`, `sar.c128` --
element dtypes; indexing builds a tensor type annotation
(`sar.c64[512, 512]`). `c64` is a pair of f32 (NumPy `complex64`),
`c128` a pair of f64.

`sar.Tensor` is the traced tensor value used inside kernels.
`sar.__version__` reports the installed package version.

## Defining kernels and operators

| Name | Meaning |
|------|---------|
| `@sar.func` | traces a Python function into a compiled kernel; annotation-free kernels specialize per call signature |
| `@sar.op` | defines an operator as a composition: inlines into kernels, runs eagerly on NumPy arrays ([defining-ops.md](defining-ops.md)) |
| `sar.Kernel` | compiled kernel handle: what an annotated `@sar.func` is, and what specializing an annotation-free one yields |
| `Kernel.compile(backend=, options=)` | compiles for a backend; returns a callable (cpu) or a design handle (hls) |
| `Kernel.to_mlir()` | the traced MLIR module text |
| `GenericKernel.specialize(*types)` | pins an annotation-free `@sar.func` to explicit types, returning a `Kernel` |
| `sar.constant(value, dtype=, shape=)` | materializes a NumPy array or scalar as a tensor constant |

Inside a kernel, tensors carry NumPy-style sugar: operators (`+`, `*`,
comparisons building masks), `abs(x)`, `x ** 2`, `x.T` /
`x.transpose()`, `x.real`, `x.conj()` / `x.conjugate()`, reduction and
scan methods (`x.sum(axis=)`, `x.mean()`, `x.std()`, `x.var()`,
`x.cumsum()`), `x.clip()`, `x.round()`, `x.astype(dtype)`, basic
slicing `x[2:6, ::2]`, and introspection (`x.shape`, `x.dtype`,
`x.ndim`, `x.size`, `len(x)`).

## Element-wise operations

`sqrt`, `cos`, `sin`, `exp`, `log`, `log2`, `log10`, `atan2` (NumPy
argument order), `hypot`, `sinc`, `absolute`/`abs`, `sign`, `floor`,
`ceil`, `round` (half away from zero -- MATLAB, not NumPy), `maximum`,
`minimum`, `clip`, `cast`, `where` (exact selection by a mask).

Complex access: `conj` (alias `conjugate`), `real`, `imag`, `angle`,
`make_complex`, `expj` (`exp(jx)` from a float tensor).

## Reductions, scans, order statistics

`sum`, `max`, `min`, `mean`, `std`, `var` (N-normalized, NumPy
convention), `argmax`, `argmin` (i64 indices, first occurrence on
ties), `cumsum` (inclusive prefix sum), `sort`, `rank_filter`,
`median_filter`.

## Layout and data movement

`transpose`, `broadcast` (rank-1 to rank-2 along `dim`), `concatenate`/
`concat`, `pad`, `flip`, `circshift`, `multilook`, `dynamic_slice` /
`dynamic_update_slice` (static-shape windows at runtime `i64[1]` offsets;
offsets clamp so the window stays in bounds).

## Compiled loops

`iterate(n, body, *carries)` stays one loop in the design; a Python `for`
unrolls at trace time. `index=True` exposes a 0-based `i64[1]` index that
can drive `dynamic_slice` / `dynamic_update_slice`.

## Signal processing

| Name | Meaning |
|------|---------|
| `fft`, `ifft` | DFT along one axis; unscaled forward, 1/N inverse; `norm=` follows NumPy; any size >= 2 on both backends |
| `fft2`, `ifft2` | both axes |
| `fftshift`, `ifftshift` | center / uncenter a spectrum; all axes when `dim` is omitted |
| `interp1d` | resampling at fractional positions; `kernel=` nearest/linear/cubic/sinc, `boundary=` zero/edge/reflect |
| `gather2d` | 2-D gather at data-dependent positions (`out[i,j] = data[rows[i,j], cols[i,j]]`), the backprojection access pattern; `kernel=` nearest/linear, `boundary=` zero/edge |
| `stolt_interp` | omega-K Stolt remapping, composed from `interp1d` and phase ramps |
| `dechirp`, `matched_filter` | pulse-compression idioms |
| `mag2db`, `pow2db`, `db`, `db2mag`, `db2pow` | decibel conversions |
| `window`, `hanning`, `hamming`, `blackman`, `kaiser`, `taylorwin` | window constants; constructors usable in host code too |

## Compilation driver

| Name | Meaning |
|------|---------|
| `sar.compile(kernel, backend=, options=)` | functional spelling of `Kernel.compile` |
| `sar.list_backends()`, `sar.get_backend(name)` | backend discovery ([backends.md](backends.md)) |
| `HLSDesign.source()`, `.cpp_path`, `.config` | the emitted C++ and the resolved configuration with provenance |
| `HLSDesign.write_testbench(inputs, expected, dir)` | self-contained csim package with golden data |
| `HLSDesign.write_synthesis_script(dir)` | design + Vitis HLS csynth script, any interface |

## Errors and diagnostics

| Name | Raised / warned when |
|------|---------------------|
| `SARError` | base class of everything below |
| `TraceError` | the Python being traced is not a valid kernel |
| `ToolchainError` | a compiler tool is missing or fails to run |
| `CompilationError` | a pipeline stage rejects the kernel |
| `LaunchError` | arguments do not match the compiled signature |
| `HLSConfigError` | an HLS option or config-file value is unknown, mistyped or out of range (also a `ValueError`) |
| `DomainWarning` | suspicious spectral-domain usage (double FFT, centered spectrum into `ifft`, mixed-domain arithmetic) |
| `PrecisionWarning` | host data widens the kernel's working precision |

Escalate a warning into an error with
`warnings.simplefilter("error", sar.DomainWarning)`.

Environment variables (tool paths, cache, runtime threads, HLS config)
are tabulated in [backends.md](backends.md#environment-variables).
