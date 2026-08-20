# Changelog

Notable changes to SAR-DSL. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[semantic versioning](https://semver.org/) with a 0.x caveat: minor
releases may change APIs until 1.0.

## [Unreleased]

A compiler from Python-authored SAR imaging kernels to native CPU code
and synthesizable Vitis HLS C++.

- **Language**: `@sar.func` kernels traced from plain Python, `@sar.op`
  user-defined operators, an orthogonal primitive set (element-wise
  arithmetic, reductions, selection, layout, FFTs of any size >= 2,
  kernel-selectable resampling, data-dependent 2-D gather, compiled
  counted loops, runtime-offset slice/update windows), MATLAB/SciPy-familiar
  vocabulary that decomposes into primitives at trace time, NumPy-style tensor
  sugar, and trace-time diagnostics.
- **Backends**: `cpu` (JIT to native code with linalg fusion, OpenMP and
  `libsar_runtime`) and `hls` (Vitis HLS 2022.2 C++ with generated csim
  testbenches and synthesis scripts). Every DSL construct compiles on
  every backend.
- **HLS resource contract**: `hls_config.yaml` states hard BRAM, URAM and
  LUTRAM caps plus DSP, FF, and LUT synthesis-report budgets. Spilled planes
  are carved into scratch arenas, so the port list is the algorithm's own I/O
  plus the ping-pong its dataflow needs; a design that cannot fit even fully
  streamed fails compilation. Mixed-radix FFT grouping, lane parallelism,
  transfer unrolling, tiling, and banking are derived per kernel and reported
  with provenance in each generated manifest.
- **Algorithms**: four complete imaging chains (omega-K, Range-Doppler,
  Chirp Scaling, Polar Format + SVA) validated against NumPy references. The
  three stripmap chains also run on real ALOS-1 data; PFA uses a synthetic
  spotlight collection. The WKA example includes an independent hand-written
  FP32 HLS implementation as a comparison baseline.

Measured accuracy, performance, and resource figures are in
[benchmarks/](benchmarks/README.md).
