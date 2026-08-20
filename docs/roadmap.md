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
- Production c64 AXI designs -- the stripmap chains at 16384², PFA at 8192²
  input and 16384² output -- complete synthesis within the 4 ns target and
  the resource budgets.
- Top-level ports are the algorithm's own I/O plus the scratch arenas its
  dataflow needs; the compiler adds none of its own.
- A reduced two-FFT AXI design with compiler-managed scratch passes Verilog RTL
  co-simulation against the same golden output as C simulation.
- Shapes are static and specialize per geometry. The CPU backend is validated
  on Linux x86-64; Vitis is optional unless synthesis artifacts are being
  validated.

Measured accuracy, performance, and resource data are in
[benchmarks/](../benchmarks/).

## Open work

- Generated designs close timing and fit the budgets, but a generated omega-K
  is several times the latency of the hand-written baseline in
  [benchmarks/](../benchmarks/README.md). Arena traffic still moves one scalar
  per beat where the kernel's read-only input ports already move full AXI
  words.
- Floating-point reassociation stays disabled: enabling it needs an
  optimization that can prove an explicit error bound. Canonicalization itself
  is bit-exact.

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
