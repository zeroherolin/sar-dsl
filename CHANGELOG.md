# Changelog

Notable changes to SAR-DSL. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[semantic versioning](https://semver.org/) with a 0.x caveat: minor
releases may change APIs until 1.0.

No release has been cut yet. Per-release change entries start with the
first one; until then this file only sketches what that release is.

## [Unreleased]

A compiler from Python-authored SAR imaging kernels to native CPU code
and synthesizable Vitis HLS C++.

- **Language**: `@sar.func` kernels traced from plain Python, `@sar.op`
  user-defined operators, an orthogonal primitive set (element-wise
  arithmetic, reductions, selection, layout, FFTs of any size >= 2,
  kernel-selectable resampling, data-dependent 2-D gather, compiled
  counted loops), MATLAB/SciPy-familiar vocabulary that decomposes into
  primitives at trace time, NumPy-style tensor sugar, and trace-time
  diagnostics.
- **Backends**: `cpu` (JIT to native code with linalg fusion, OpenMP and
  `libsar_runtime`) and `hls` (Vitis HLS 2022.2 C++ with generated csim
  testbenches and synthesis scripts). Every DSL construct compiles on
  every backend; `test_backend_symmetry.py` is the gate.
- **HLS resource contract**: `hls_config.yaml` states hard BRAM, URAM and
  LUTRAM caps plus a DSP synthesis-report budget. Placement never exceeds
  the memory caps; a design that cannot fit even fully streamed fails
  compilation. Strategy is derived per kernel and reported with provenance.
- **Algorithms and validation**: four complete imaging chains (omega-K,
  Range-Doppler, Chirp Scaling, Polar Format + SVA) validated against
  NumPy references; the three stripmap chains also run on real ALOS-1
  data, while PFA uses a synthetic spotlight collection. All pass
  generated C-simulation at both precisions; representative designs
  complete Vitis synthesis at 4 ns. The WKA example also includes an
  independent hand-written FP32 HLS implementation documenting packed
  AXI, radix-4 row parallelism, engine reuse, fused Stolt processing, and
  ping-pong corner turns.
