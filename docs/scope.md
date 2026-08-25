# Project scope

SAR-DSL keeps the language small and backend-neutral: algorithms compose a shared set of signal-processing primitives, and every public construct must lower through both the CPU and HLS backends. The [dialect reference](dialect.md), [Python API](python-api.md), and [backend guide](backends.md) document the implemented surface; this file records the validation boundary, the limits a user can hit, and what the project deliberately does not do.

## Verified scope

- The omega-K, Range-Doppler, Chirp Scaling, and Polar Format examples run against NumPy references. Their complete kernels emit through both backends at c64 and c128 precision.
- Backend symmetry, operation verification, lowering passes, generated C++, runtime numerics, and complete imaging chains are covered by pytest and lit.
- HLS strategy is derived from kernel facts and device constraints; the constraints themselves are either sized from a tabulated `part` or stated outright, never mixed. Placement enforces hard BRAM, URAM, and LUTRAM caps; designs that cannot fit are rejected rather than overcommitted.
- Complete c128 designs at input edge N=512 for all four algorithms pass Vitis HLS 2022.2 C-simulation and synthesis below the 4 ns board period. CSA's 3.627 ns estimate is recorded as a warning against the 3.5 ns uncertainty-adjusted scheduling budget.
- Production c64 AXI designs -- the stripmap chains at 16384 × 16384 input and output, and PFA at 8192 × 8192 input with 16384 × 16384 output -- complete synthesis within the resource budgets and below the 4 ns board period; CSA's 3.627 ns estimate carries the same timing warning rather than failing the design.
- Top-level ports are the algorithm's own I/O plus the scratch arenas its dataflow needs; intermediate planes are carved into those arenas rather than exposed as one master per buffer.
- A 256-point two-FFT AXI design with compiler-managed scratch passes Verilog RTL co-simulation against the same golden output as C simulation.
- Shapes are static and specialize per geometry. The CPU backend is validated on Linux x86-64; Vitis is optional unless synthesis artifacts are being validated.

Measured accuracy, performance, and resource data are in [benchmarks/](../benchmarks/).

## Known limits

These are properties of the current compiler, not of the language; a kernel that hits one still compiles on the other interface or backend.

- **`interface='stream'` needs a row-major sweep.** AXI4-Stream has no addresses, so every public port must be read or written exactly once in row-major order. Element-wise front ends qualify; anything containing a transform or a corner turn does not, and takes `interface='axi'` instead.
- **`ap_memory` designs are bounded by the on-chip caps.** With no off-chip tier the whole working set has to fit, so the raster a validation package can reach depends on how many full planes the chain holds live: 256 × 256 for the three stripmap chains on the reference VU13P budgets, and 64 × 64 for polar format, which resamples onto twice its input on each axis and reaches the bound first. Past that the compiler narrows the transform transfers before it refuses the design; `interface='axi'` streams instead and is not bounded this way.
- **`sar.sort` sorts a whole line, not one statistic.** The lowering is Batcher's odd-even mergesort -- an exact compare-exchange network over the static extent, O(n log^2 n) compare-exchanges in O(log^2 n) parallel sweeps. `sar.rank_filter` settles a single order statistic and is the cheaper answer when that is what is wanted.
- **Stages are dataflow only when the memory contract permits it.** The compiler forks small read-only two-reader inputs and marks that schedule dataflow. Production designs whose full-size planes spill to DRAM still run their stage chain sequentially: reusing one scratch arena across independent processes would make Vitis reject the m_axi bundle, while giving every live plane its own master would change the external ABI and the resource model. The legalization check keeps that trade explicit instead of emitting a design that synthesizes with hidden bus serialization.
- **Floating-point reassociation stays disabled.** Enabling it needs an optimization that can prove an explicit error bound; canonicalization itself is bit-exact.

## Non-goals

- Licensed Vitis CI infrastructure, Vivado place-and-route, board drivers, and on-device benchmarking. Validation ends at local C-sim through Vitis HLS and synthesis.
- Dynamic shapes; one specialization per geometry is the compilation model.
- Fixed-point HLS types.
- Algorithm-specific or ML-flavored convenience operations without independent signal-processing semantics.
- Data-dependent control flow inside kernels; autofocus-style iteration remains host-orchestrated.
