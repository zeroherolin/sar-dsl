# SAR-DSL architecture

## Overview

SAR-DSL compiles Python-authored SAR imaging kernels through a small MLIR dialect to multiple hardware targets. The design uses a multi-backend compiler architecture -- a thin language layer, a dialect-centric core, and pluggable per-target backends -- while making deliberately different choices where a signal-processing DSL has different needs than a GPU kernel language.

```
 ┌────────────────────────────────────────────────────────────┐
 │ python/sar                                                 │
 │  language/ tracing, dtype/shape checks   ir/ textual MLIR  │
 │  compiler/ stage driver, cache, tools    runtime/ ctypes   │
 │  backends/ discovery & registry                            │
 └───────────────┬────────────────────────────────────────────┘
                 │  serialized `sar` dialect module
 ┌───────────────▼────────────────────────────────────────────┐
 │ C++ core (sar-opt)                                         │
 │  Dialect/SAR/IR          ops on builtin tensors, verifiers │
 │  Dialect/SAR/Transforms  sar-decomplexify (re/im split)    │
 │  Conversion/             SARToLinalg, SARSignalToRuntime,  │
 │                          SARFFTToAffine, SARInterpToAffine │
 │  Pipelines/              sar-to-{linalg,affine,llvm}       │
 └───────┬───────────────────────────────┬────────────────────┘
         │ cpu                           │ hls
 ┌───────▼───────────────┐   ┌───────────▼────────────────────┐
 │ fusion + OpenMP loops │   │ affine + HLS pipeline          │
 │ mlir-translate        │   │ sar-translate --hls-emit-hlscpp│
 │ clang -O3 -shared     │   │ -> Vitis HLS C++               │
 │ + libsar_runtime      │   │                                │
 │ -> kernel.so (ctypes) │   │                                │
 └───────────────────────┘   └────────────────────────────────┘
```

## Design decisions

### 1. Operations on builtin tensors

A domain dialect could define its own container types (`!sar.matrix` etc.), but such types carry no semantics beyond shape and element type while forcing type conversions, function-signature rewrites and custom constant encodings through the whole stack. Following TOSA and StableHLO, the `sar` dialect operates directly on builtin `tensor<...>` values: verifiers focus on real domain invariants (FFT sizes, interpolation-kernel parameters, plane-precision matching, slice bounds), and all upstream machinery (bufferization, linalg, dense constants with complex elements) works unmodified.

### 2. Textual IR as the frontend boundary

The frontend builds IR as _text_ (generic op syntax) rather than through the MLIR Python bindings:

- the Python package is pure and importable with zero compiled artifacts;
- the frontend is insulated from LLVM C++/Python API churn -- the contract is the serialized IR, which is far more stable;
- kernels are tiny (tens of ops), so builder performance is irrelevant.

Type/shape errors are reported twice: eagerly at trace time with Pythonic messages, and authoritatively by the C++ verifiers when `sar-opt` parses the module. Ops are printed in generic form (`"sar.fft"(%0) <{dim = 1 : i64}>`) because that syntax is stable regardless of custom assembly formats.

### 3. CPU execution model

The `sar-to-llvm-pipeline` (registered in `sar-opt`) does the entire descent:

1. `convert-sar-signal-to-runtime`: `sar.fft`/`sar.ifft`/`sar.interp1d` become calls to `libsar_runtime` using the documented bufferization escape hatch (`to_buffer` / fresh alloc / `to_tensor restrict writable`). Runtime declarations carry `llvm.emit_c_interface`, so calls lower to `_mlir_ciface_*` symbols taking memref descriptors by pointer.
2. `convert-sar-to-linalg`: everything else becomes linalg-on-tensors (complex element types preserved; `fftshift` is a gather-style generic), followed by **linalg elementwise fusion** -- long phase-computation chains collapse into single generics, eliminating whole intermediate tensors (the dominant memory-bandwidth cost at 16384^2).
3. one-shot bufferization with identity layouts, `sar-distinct-return-buffers` (two result positions may share one SSA value, but the destination-passing ABI needs a distinct backing buffer for each), `buffer-results-to-out-params` (destination-passing style: results become trailing out-arguments -- no cross-boundary ownership questions), **buffer sharing** (`sar-reuse-buffers`: a chain of whole-raster passes retires a plane as soon as the next pass has read it, so the ones whose lifetimes do not overlap share an allocation), buffer deallocation, **linalg -> scf.parallel -> OpenMP dialect** (linked against LLVM's libomp), complex->standard, and the usual LLVM conversions with `-O3 -march=native` codegen.

   On this path the sharing runs with `allow-retype`; the mechanism, and why the affine path leaves it off, are described with the pass in [dialect.md](dialect.md#transformations-and-lowering-paths).

   What the sharing can reach is bounded by what escapes. A buffer handed to anything that returns a memref or a tensor may be aliased past the span the scan measures, so it is left alone; on the CPU path that covers the FFT and interpolation staging buffers, which reach the runtime through a `memref.cast`. Those chains therefore save little here, and the sharing that matters for them happens on the affine path, where the same transforms are loop nests rather than calls.

The Python launcher allocates result arrays with NumPy and invokes `_mlir_ciface_<kernel>` via ctypes with strided memref descriptors.

Binding FFT/interpolation to a runtime library mirrors how production compilers bind vendor libraries (cuFFT, FFTW): the _pipeline structure_ is the compiler's job; leaf transforms with decades of optimization behind them are not re-derived from loops. The runtime is one file of dependency-free C++ (radix-2/Bluestein FFT in double precision, kernel-selectable resampling, and an affinity-capped reusable thread pool). This pool parallelizes only FFT/interpolation calls; generated loops use the separate OpenMP runtime, as detailed in [backends.md](backends.md#cpu-backend-sarbackendscpu).

### 4. DSL vocabulary above IR primitives

The IR op set stays small and orthogonal (element-wise arithmetic, reductions, selection, layout, FFT/interpolation leaves); the _language_ surface is deliberately richer. MATLAB/SciPy-familiar vocabulary -- `fft2`, `matched_filter`, `dechirp`, `mag2db`, `sinc`, `circshift`, window constants -- decomposes into those primitives at trace time (`sar/language/signal.py`), and user `@sar.op` definitions extend the vocabulary through exactly the same mechanism. Two granularities, one rule: anything expressible as a composition lives in Python; only computations with irreducible semantics (FFT kernels, gathers, selection) become IR ops with verifiers and per-backend lowerings. The rule is applied without exceptions: even Stolt remapping -- the heart of omega-K -- is a Python composition of a position computation, `interp1d` and phase ramps, which element-wise fusion collapses back into single loop nests.

### 5. Backend plugin model

`sar.backends` discovers `Backend` classes from `sar/backends/<name>` (built-in) or `$SAR_DSL_BACKEND_PATH` (out-of-tree extensions). A backend implements:

```python
class Backend(BaseBackend):
    name = "cpu"
    @classmethod
    def is_available(cls) -> bool: ...
    def add_stages(self, stages, metadata): ...   # ordered artifact refiners
    def make_launcher(self, artifact, metadata): ...
```

Stages communicate through a shared `KernelMetadata` and a content-addressed on-disk cache (`~/.cache/sar-dsl/<sha256>/`). Keys include the IR, options, backend package, and Python driver fingerprint; eviction is a bounded, process-safe LRU. Execution backends return callables; emission backends (`hls`) return self-contained `HLSDesign` artifact handles.

### 6. HLS backend

All kernels follow the split-complex affine path (`sar-to-affine-pipeline`):

1. `sar-decomplexify` (Dialect/SAR/Transforms) rewrites functions so no complex types remain: complex tensors become (re, im) float plane pairs, complex arithmetic expands into real SAR ops (`mul` becomes the 4-multiply form, `abs` becomes `sqrt(re^2+im^2)`), and the signal/interpolation ops map to their split-complex forms (`fft_split`, `interp1d_split`).
2. `convert-sar-fft-to-affine` lowers `fft_split` to mixed radix-4/2 **Stockham autosort** loop nests. Stockham is the deliberate choice over Cooley-Tukey: it needs no bit-reversal permutation, so every access is an affine function of loop indices -- which is what HLS loop analysis, pipelining and array partitioning require. Twiddle factors are precomputed into one constant memref per stage; a radix-4 stage is cyclically banked so its three reads land apart. Per-stage tables prevent a concurrent reader from replicating entries that only another stage uses. Radix-4 halves the stage count where possible. Lines are transformed in blocks: a prefetch sweep copies each block into on-chip line buffers with unit-stride external accesses, the butterfly stages run on chip with a compact DSP-budgeted lane loop innermost (each twiddle fetch shared across lanes, the stage chain drawing scratch blocks from a reusable pool), and a mirrored sweep writes the block back. On a slow axis the transfer block may cover a complete packed word while a narrower compute engine visits it in sub-blocks, preserving burst width without replicating every butterfly buffer. The buffers carry banking hints (`hls.partition_*`) the backend's partition pass applies verbatim.

   Identical affine sweeps are fused before lifetime-based allocation reuse. This ordering is part of the performance contract: reusing a later output as an earlier input would introduce a false alias dependence and prevent real/imaginary phase scans or corner turns from sharing one raster pass.

   Corner turns -- the transposes that carry a raster between the range and azimuth domains -- are staged through an on-chip block (`sar-stage-transposes`). A transposing nest reads one buffer along its rows and writes the other along its columns, so one of the two strides by a whole row whichever loop is innermost; a square block in between makes both sides contiguous. The block is sized from the block RAM cap, split across the kernel's corner turns.

   Sizes that are not powers of two reduce to that same machinery through **Bluestein's chirp-z** algorithm: the DFT is rewritten as a convolution evaluated by two padded Stockham transforms. Because the size is a compile-time constant, the chirp and the kernel spectrum `B = FFT_M(b)` are folded on the host into constant globals, so the device-side work stays two power-of-two transforms plus three element-wise passes -- all affine, nothing size-dependent at runtime.

3. `convert-sar-interp-to-affine` lowers `interp1d_split` to affine loops with the interpolation taps statically unrolled (the tap count and weights follow the op's `kernel` attributes). The gather index is inherently data-dependent, so those loads are plain (non-affine) memref loads with _clamped_ addresses, and out-of-range taps are masked with selects -- the loop bodies stay straight-line, which keeps HLS pipelining applicable. On packed compiler-owned arenas, the HLS pass recognizes the clamp family and reuses two adjacent vector words for the complete tap window. Position and weight arithmetic is f64, mirroring the CPU runtime.
4. `sar.gather2d` lowers like the interpolation -- clamped, select-masked loads in a straight-line body -- and compiled loops (`sar.iterate`) reach this flow as `scf.for`; runtime-offset `dynamic_slice`/`dynamic_update_slice` lower to bounded indexed accesses.

   Both gathers try for a **band** first. A SAR resampling stage moves each sample by an amount acquisition geometry sets, not the scene: Stolt remapping displaces a range bin by the ratio of Doppler to carrier frequency, RCMC by the range curvature over the synthetic aperture, polar regridding by the aperture's angular extent. For any realizable stripmap geometry all three are small fractions of the swath. So when `sar-displacement-range` can pin a compile-time bound `D` on `|positions - j|`, output column `j` only ever reads source columns `[j-D-taps/2, j+D+taps/2]`, and the gather runs against a sliding band that fits on chip instead of the resident plane a data-dependent gather would demand. If the position field comes from runtime inputs and no bound can be proven, the compiler next tries a complete source-row cache. It also prefers that immutable cache over a wider sliding band whose ring-buffer dependence would dominate scheduling. The cache is used only when its replicated split-complex storage fits the resource-derived per-gather allowance; it turns arbitrary tap reads into banked on-chip accesses after one contiguous row fill. Rows that exceed that allowance retain the direct full-plane gather. An imprecise range analysis therefore costs performance only when neither safe staging form fits; it never changes the result. Since a dataflow task may not yield values, `sar-demote-loop-carries` first rewrites the buffer carry into side effects (the body iterates in the init buffer, a per-iteration copy replaces the yield).

5. The bufferized affine IR enters the HLS pipeline (`-hls-pipeline`), which builds the dataflow hierarchy, places buffers on or off chip and shapes the interfaces. Independent sibling sweeps with identical bounds fuse before task formation, so split real/imaginary outputs share phase arithmetic. `sar-translate` then emits the final C++.

With that, every SAR operation lowers in the affine flow and complete imaging chains (omega-K, range-Doppler, chirp scaling, polar format) emit as single HLS designs. Correctness is checked at two levels: the identical affine IR is compiled to native code by `--sar-affine-to-llvm-pipeline` and checked against NumPy (validating the lowering's arithmetic), and the emitted C++ itself C-simulates against a generated testbench -- all four imaging chains pass, matching their NumPy references to rounding distance (`benchmarks/README.md` tabulates the errors).

## The HLS dialect

The `hls` dialect (`include/sar/Dialect/HLS`) models what a synthesis backend needs to say about a design that MLIR's own dialects cannot: dataflow structure (`schedule`, `node`, `buffer`, `stream`), where a buffer lives (`#hls.mem<dram>`, `bram_t2p`, ...), how an array is banked (`#hls.partition`), and what an interface looks like (`axi.bundle`, `axi.port`).

The SAR-DSL build consumes one coherent LLVM/MLIR toolchain: `sar-opt` runs every pass from SAR down to scheduled HLS IR, and `sar-translate` writes the C++. Local development may build the pinned submodule in-tree; CI uses the matching prebuilt LLVM release and only builds SAR-DSL. Memory placement, AXI shaping, and the configuration schema are in [backends.md](backends.md).
