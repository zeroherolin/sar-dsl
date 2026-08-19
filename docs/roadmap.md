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
- 16384 × 16384 c64 omega-K, Range-Doppler, and Chirp Scaling designs complete
  synthesis. WKA estimates 2.92 ns against the 4 ns target at 10.6G cycles;
  RDA and CSA miss it. Implementation timing remains outside this scope.
- A Polar Format c64 design completes N=8192 synthesis within resource
  budgets; its 6.067 ns estimate remains open timing work.
- Shapes are static and specialize per geometry. The CPU backend is validated
  on Linux x86-64; Vitis is optional unless synthesis artifacts are being
  validated.

Measured accuracy, performance, and resource data are in
[benchmarks/](../benchmarks/).

## Open compiler work

- Vectorize external DRAM ports to wide beats: scalar `m_axi` ports cap
  at one element per cycle whatever the loop shape, and the wide-port
  transfers are the dominant remaining latency gap to the hand-written
  design.
- Reuse one parameterized FFT engine across compatible transform call sites.
- Externalize large axis/window tables and deduplicate size-specialized copy
  loops to reduce extreme-raster source and synthesis time.
- Close remaining external-load and interpolation/gather initiation-interval
  bottlenecks so Range-Doppler, Chirp Scaling, and Polar Format meet 4 ns at
  production rasters.
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
