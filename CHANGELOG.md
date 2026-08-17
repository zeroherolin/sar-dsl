# Changelog

Notable changes to SAR-DSL. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[semantic versioning](https://semver.org/) with a 0.x caveat: minor
releases may change APIs until 1.0.

## [Unreleased]

## [0.1.0]

First release: a working compiler from Python-authored SAR imaging
kernels to native CPU code and synthesizable Vitis HLS C++.

### Language

- `@sar.func` kernels traced from plain Python; type annotations
  optional (annotation-free kernels specialize per call signature).
- `@sar.op` user-defined operators: compositions that inline into
  kernels on every backend and run eagerly on numpy arrays.
- Orthogonal primitive set: element-wise arithmetic and
  transcendentals, comparisons and exact selection (`where`),
  reductions and `argmax`/`argmin`, scan (`cumsum`) and order
  statistics (`sort`, `rank_filter`/`median_filter`), layout
  (`slice`/`concat`/`pad`/`transpose`/`broadcast`/`flip`/`fftshift`),
  FFTs of any size >= 2, kernel-selectable resampling
  (`interp1d`: nearest/linear/cubic/windowed sinc, three boundary
  policies), 2-D gather at data-dependent positions (`gather2d`, the
  backprojection access pattern), and compiled counted loops with
  tensor-carried state (`iterate`).
- Matlab/scipy-familiar vocabulary (`fft2`, `matched_filter`,
  `dechirp`, dB conversions, window constants incl. Taylor) that
  decomposes into primitives at trace time.
- numpy-style tensor sugar: operators and comparisons, `x.T` /
  `x.transpose()`, `x.conj()` / `x.conjugate()`, reduction and scan
  methods, `x.clip()` / `x.round()` / `x.astype()`, basic slicing,
  introspection (`.shape`/`.ndim`/`.size`/`len()`), `axis=` beside
  `dim=` everywhere, and clear trace-time diagnostics for the things a
  symbolic tensor cannot do (`float(x)`, truth values, numpy ufuncs).
- Trace-time diagnostics: spectral-domain misuse (`DomainWarning`) and
  host-data precision widening (`PrecisionWarning`).

### Backends

- `cpu`: JIT to native code (linalg fusion, OpenMP, `libsar_runtime`
  FFT/interpolation, plane-pooled allocation).
- `hls`: Vitis HLS 2022.2 C++ emission with generated csim testbenches
  and synthesis scripts. Split-complex Stockham FFTs (Bluestein for
  non-powers-of-two), straight-line masked interpolation gathers,
  staged corner turns, banded gathers under a compile-time
  displacement bound, and a fixed top-level signature: the algorithm's
  own I/O planes plus one scratch port, whatever the placement.
- Backend symmetry as policy: every DSL construct compiles on every
  backend (`test_backend_symmetry.py` is the gate).
- HLS configuration split into constraints (device budgets, interface,
  part, clock -- `hls_config.yaml`) and strategy, which the compiler
  derives per kernel (`autotune.py`) and reports with provenance;
  unknown or invalid options raise `sar.HLSConfigError`.

### Algorithms and validation

- Four complete imaging chains -- omega-K, Range-Doppler, Chirp
  Scaling, Polar Format + SVA -- validated against NumPy references,
  focused on real ALOS-1 data, and emitted as single HLS designs that
  pass their generated C-simulation testbenches.
- Every chain's `build_kernel` takes `dtype=sar.c64`/`sar.c128`: single
  precision is a first-class build option, and the precision benchmark
  builds it rather than rewriting source. Cross-backend agreement at f32
  is gated by csim.
- Benchmark suite: performance, focusing quality (IRW/PSLR/ISLR),
  cross-backend accuracy, precision ladders and FPGA resource
  estimates.
