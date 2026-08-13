# SAR-DSL architecture

## Overview

SAR-DSL compiles Python-authored SAR imaging kernels through a small MLIR
dialect to multiple hardware targets. The design borrows the *shape* of
mature multi-backend compilers (Triton / FlagTree) -- a thin language layer,
a dialect-centric core, and pluggable per-target backends -- while making
deliberately different choices where a signal-processing DSL has different
needs than a GPU kernel language.

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
         │ cpu                           │ scalehls
 ┌───────▼───────────────┐   ┌───────────▼────────────────────┐
 │ fusion + OpenMP loops │   │ float:   linalg -> HIDA-pytorch│
 │ mlir-translate        │   │ complex: affine -> HIDA-cpp    │
 │ clang -O3 -shared     │   │ scalehls-translate             │
 │ + libsar_runtime      │   │ -> Vitis HLS C++               │
 │ -> kernel.so (ctypes) │   │                                │
 └───────────────────────┘   └────────────────────────────────┘
```

## Design decisions

### 1. Operations on builtin tensors

A domain dialect could define its own container types (`!sar.matrix`
etc.), but such types carry no semantics beyond shape and element type
while forcing type conversions, function-signature rewrites and custom
constant encodings through the whole stack. Following TOSA and StableHLO,
the `sar` dialect operates directly on builtin `tensor<...>` values:
verifiers focus on real domain invariants (power-of-two FFT sizes,
frequency-axis lengths, precision matching), and all upstream machinery
(bufferization, linalg, dense constants with complex elements) works
unmodified.

### 2. Textual IR as the frontend boundary

The frontend builds IR as *text* (generic op syntax) rather than through the
MLIR Python bindings:

- the Python package is pure and importable with zero compiled artifacts;
- the frontend is insulated from LLVM C++/Python API churn -- the contract
  is the serialized IR, which is far more stable;
- kernels are tiny (tens of ops), so builder performance is irrelevant.

Type/shape errors are reported twice: eagerly at trace time with Pythonic
messages, and authoritatively by the C++ verifiers when `sar-opt` parses the
module. Ops are printed in generic form (`"sar.fft"(%0) <{dim = 1 : i64}>`)
because that syntax is stable regardless of custom assembly formats.

### 3. CPU execution model

The `sar-to-llvm` pipeline (registered in `sar-opt`) does the entire descent:

1. `convert-sar-signal-to-runtime`: `sar.fft`/`sar.ifft`/`sar.stolt_interp`
   /`sar.interp1d` become calls to `libsar_runtime` using the documented
   bufferization escape hatch (`to_buffer` / fresh alloc / `to_tensor
   restrict writable`). Runtime declarations carry `llvm.emit_c_interface`,
   so calls lower to `_mlir_ciface_*` symbols taking memref descriptors by
   pointer.
2. `convert-sar-to-linalg`: everything else becomes linalg-on-tensors
   (complex element types preserved; `fftshift` is a gather-style generic),
   followed by **linalg elementwise fusion** -- long phase-computation
   chains collapse into single generics, eliminating whole intermediate
   tensors (the dominant memory-bandwidth cost at 16384^2).
3. one-shot bufferization with identity layouts,
   `buffer-results-to-out-params` (destination-passing style: results become
   trailing out-arguments -- no cross-boundary ownership questions), buffer
   deallocation, **linalg -> scf.parallel -> OpenMP dialect** (linked
   against LLVM's libomp), complex->standard, and the usual LLVM
   conversions with `-O3 -march=native` codegen.

   Measured effect on the full 16384x16384 ALOS-1 omega-K chain:
   55.4 s -> 3.6 s.

The Python launcher allocates result arrays with numpy and invokes
`_mlir_ciface_<kernel>` via ctypes with strided memref descriptors.

Binding FFT/Stolt to a runtime library mirrors how production compilers bind
vendor libraries (cuFFT, FFTW): the *pipeline structure* is the compiler's
job; leaf transforms with decades of optimization behind them are not
re-derived from loops. The runtime is a few hundred lines of
dependency-free C++ (radix-2 FFT in double precision, windowed-sinc
resampling, simple std::thread parallelism).

### 4. Backend plugin model (FlagTree-style)

`sar.backends` discovers `Backend` classes from `sar/backends/<name>`
(installed) or `third_party/<name>/backend` (source tree). A backend
implements:

```python
class Backend(BaseBackend):
    name = "cpu"
    @classmethod
    def is_available(cls) -> bool: ...
    def add_stages(self, stages, metadata): ...   # ordered artifact refiners
    def make_launcher(self, artifact, metadata): ...
```

Stages communicate through a shared `KernelMetadata` and a content-addressed
on-disk cache (`~/.cache/sar-dsl/<sha256>/`), so recompilation is skipped
for unchanged kernels. Execution backends return callables; emission
backends (ScaleHLS) return artifact handles.

### 5. ScaleHLS-HIDA backend: two flows

Float-only element-wise kernels take the original path: linalg-on-tensors
into HIDA's PyTorch entry point (dataflow decomposition, tiling).

Kernels using complex arithmetic or FFTs take the split-complex affine
path (`sar-to-affine-pipeline`):

1. `sar-decomplexify` (Dialect/SAR/Transforms) rewrites functions so no
   complex types remain: complex tensors become (re, im) float plane
   pairs, complex arithmetic expands into real SAR ops (`mul` becomes the
   4-multiply form, `expj` becomes `cos`/`sin`, `abs` becomes
   `sqrt(re^2+im^2)`), and the signal/interpolation ops map to their
   split-complex forms (`fft_split`, `interp1d_split`,
   `stolt_interp_split`).
2. `convert-sar-fft-to-affine` lowers `fft_split` to radix-2 **Stockham
   autosort** loop nests. Stockham is the deliberate choice over
   Cooley-Tukey: it needs no bit-reversal permutation, so every access is
   an affine function of loop indices -- which is what HLS loop analysis,
   pipelining and array partitioning require. Twiddle factors are
   precomputed into constant memref globals; stages are statically
   unrolled with ping-pong scratch buffers.
3. `convert-sar-interp-to-affine` lowers the split interpolation ops
   (`interp1d_split`, `stolt_interp_split`) to affine loops with the eight
   windowed-sinc taps statically unrolled. The gather index is inherently
   data-dependent, so those loads are plain (non-affine) memref loads with
   *clamped* addresses, and out-of-range taps are masked with selects --
   the loop bodies stay straight-line, which keeps HLS pipelining
   applicable. Position and weight arithmetic is f64, mirroring the CPU
   runtime.
4. The bufferized affine IR enters HIDA's C++ entry point
   (`-hida-cpp-pipeline`), then `scalehls-translate` emits Vitis HLS C++.

With that, every SAR operation lowers in the affine flow and complete
imaging chains (omega-K, range-Doppler, chirp scaling) emit as single
HLS designs. The identical affine IR is compiled to native code by
`--sar-affine-to-llvm-pipeline` and checked against numpy, so the HLS
lowering's arithmetic is validated without an HLS simulator (the full
omega-K chain matches the NumPy reference bit-for-bit at f32 output
precision).

## Two LLVM trees, on purpose

`externals/llvm-project` (current LLVM) serves the SAR core; ScaleHLS-HIDA
pins its own older LLVM through `polygeist`. They never link into the same
binary -- the boundary between the two worlds is serialized linalg IR piped
into `scalehls-opt`, which is exactly the versioned, stable surface this
architecture is built around.
