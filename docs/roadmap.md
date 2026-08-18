# Scope and roadmap

SAR-DSL keeps the language small and backend-neutral: algorithms compose a
shared set of signal-processing primitives, and every public construct must
lower through both the CPU and HLS backends. The
[dialect reference](dialect.md), [Python API](python-api.md), and
[backend guide](backends.md) document the implemented surface; this file only
records its validation boundary and open compiler work.

## Verified scope

- The omega-K, Range-Doppler, Chirp Scaling, and Polar Format examples run
  against NumPy references. Their complete kernels emit through both backends
  at c64 and c128 precision.
- Backend symmetry, operation verification, lowering passes, generated C++,
  runtime numerics, and complete imaging chains are covered by pytest and lit.
- HLS strategy is derived from kernel facts and device constraints. Placement
  enforces hard BRAM, URAM, and LUTRAM caps; designs that cannot fit are
  rejected rather than overcommitted.
- Complete N=32 designs for all four algorithms pass Vitis HLS 2022.2
  C-simulation and synthesis at the 4 ns target.
- A 4096 × 4096 RDA design and a 16384 × 16384 c64 omega-K design complete
  synthesis. The latter fits the resource caps but estimates 5.698 ns against
  the 4 ns target; it is a synthesizability result, not timing closure.
- Shapes are static and specialize per geometry. The CPU backend is validated
  on Linux x86-64; Vitis is optional unless synthesis artifacts are being
  validated.

Measured accuracy, performance, and resource data are in
[benchmarks/](../benchmarks/).

## Open compiler work

- Derive a mixed-radix, multi-line FFT engine and instance-reuse plan from
  bandwidth and DSP constraints.
- Split spilled buffers across a fixed number of AXI scratch masters using
  lifetime and concurrent-access analysis.
- Externalize large axis/window tables and deduplicate size-specialized copy
  loops to reduce extreme-raster source and synthesis time.
- Add a target-legality verifier before HLS emission and extend nested
  fan-out/fan-in, view/subview, and multi-block topology coverage.
- Allow `sar.iterate` indices to drive dynamic slice offsets for sub-aperture
  and block processing.
- Gate reassociating phase rewrites behind an explicit numerical-error budget.

## Non-goals

- Licensed Vitis CI infrastructure, Vivado place-and-route, board drivers, and
  on-device benchmarking. Validation ends at local Vitis HLS C-simulation and
  synthesis.
- Dynamic shapes; one specialization per geometry is the compilation model.
- Fixed-point HLS types.
- Algorithm-specific or ML-flavored convenience operations without independent
  signal-processing semantics.
- Data-dependent control flow inside kernels; autofocus-style iteration remains
  host-orchestrated.
