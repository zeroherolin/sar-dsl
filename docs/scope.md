# Project scope

SAR-DSL is a statically specialized compiler for whole-array synthetic aperture radar imaging pipelines. The language is backend-neutral: every public construct must lower through both the CPU and HLS backends.

## Supported and validated

- Static rank-1 and rank-2 tensors with `f32`, `f64`, `i32`, `i64`, `c64`, and `c128` elements.
- Element-wise arithmetic, reductions, selection, layout changes, FFTs, interpolation, data-dependent 2-D gathers, order-statistic operations, and fixed-trip compiled loops.
- Native CPU execution on Linux x86-64.
- Vitis HLS C++ emission for `ap_memory`, AXI4 memory-mapped, and eligible AXI4-Stream interfaces.
- Generated C simulation, representative RTL co-simulation, and synthesis-report validation.
- Complete omega-K, Range-Doppler, Chirp Scaling, and Polar Format examples, each checked against a NumPy reference.
- ALOS-1 PALSAR FBS HH stripmap processing through the externally downloaded ASF DAAC granule documented in the [examples guide](../examples/README.md#alos-1-stripmap-data).
- Resource-constrained HLS planning from either a listed target device or a complete explicit device contract.

The [benchmark report](../benchmarks/README.md) is the source for measured accuracy, image quality, performance, resource use, and synthesis results.

## Language and backend limits

- **Shapes are static.** Each geometry and dtype produces a separate specialization.
- **Kernel control flow is static.** Python loops unroll while tracing; `sar.iterate` preserves a fixed-trip loop. Data-dependent branching is expressed with element-wise masks and `sar.where`.
- **AXI4-Stream requires sequential access.** Every public port must have one complete monotonic row-major sweep. Transforms, transposes, and gathers require addressed storage and use the AXI memory-mapped interface.
- **`ap_memory` is capacity-limited.** All live arrays must fit the configured on-chip memory budget. AXI memory-mapped designs may place full-size intermediates in compiler-managed external scratch arenas.
- **Ordering operations are explicit networks.** `sar.sort` uses an exact static sorting network; use `sar.rank_filter` when only one local order statistic is needed.
- **Floating-point reassociation is disabled.** Canonicalization preserves operation ordering unless a transformation is exact.
- **CPU and HLS leaf implementations may use different internal precision.** Declared tensor dtypes define storage and visible arithmetic; numerical tolerances are documented with the benchmark results.

## Outside the project

- Dynamic-shape execution.
- Fixed-point or arbitrary-precision signal tensors.
- Vivado implementation, board support packages, drivers, and on-device runtime management.
- Post-route timing closure and hardware throughput claims.
- Licensed Vitis infrastructure in public CI.
- Algorithm-specific primitives that can be expressed as compositions of existing language operations.
