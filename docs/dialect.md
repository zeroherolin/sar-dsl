# The `sar` dialect

Whole-array operations for SAR imaging. All operations work on builtin
ranked tensors with **static shapes**; supported element types are `f32`,
`f64`, `i32`, `i64`, `complex<f32>` and `complex<f64>`.

## Element-wise arithmetic

| Op | Signature | Notes |
|----|-----------|-------|
| `sar.add/sub/mul/div` | `(T, T) -> T` | float, int or complex elements |
| `sar.add_scalar` | `(T) -> T`, `scalar: f64` | complex: added to real part |
| `sar.mul_scalar` | `(T) -> T`, `scalar: f64` | |
| `sar.sqrt` | `(T) -> T` | float only |
| `sar.cos` / `sar.sin` / `sar.exp` / `sar.log` | `(T) -> T` | float only |
| `sar.atan2` | `(T, T) -> T` | float only; NumPy argument order `(y, x)` |
| `sar.abs` | `(complex<t>|t) -> t` | complex magnitude / float abs |
| `sar.cmp` | `(t, t) -> t`, `predicate` | 0.0/1.0 mask (frontend: `x > y`, `x == 0.0`, ...) |
| `sar.where` | `(t, T, T) -> T` | exact per-element selection by a mask (NumPy `where`) |
| `sar.conj` | `(complex<t>) -> complex<t>` | complex conjugate |
| `sar.real` / `sar.imag` | `(complex<t>) -> t` | plane extraction |
| `sar.complex` | `(t, t) -> complex<t>` | assemble from (re, im) planes |
| `sar.cast` | `(T) -> U` | float<->float, complex<->complex, float->complex, int<->int, int->float, float->int (truncation) |
| `sar.constant` | `() -> T` | dense elements attribute |

## Reductions

| Op | Semantics |
|----|-----------|
| `sar.reduce {kind, dim}` | rank-2 -> rank-1 along `dim`; `kind` is `sum` (float/complex) or `max`/`min` (float); rank-1 inputs are normalized to `1 x n` by the frontend |
| `sar.argmax {dim}` | rank-2 float -> rank-1 i64 indices (NumPy `argmax`: first occurrence on ties) |

## Scan and order-statistics

| Op | Semantics |
|----|-----------|
| `sar.cumsum {dim}` | rank-1 or rank-2 float or complex -> same shape; inclusive prefix sum along `dim` (NumPy `cumsum`). The scan is sequential -- it is not decomposed into a parallel tree. For complex tensors, `sar-decomplexify` splits it into two independent float scans |
| `sar.rank_filter {window, rank, dim}` | rank-1 or rank-2 float -> same shape; windowed order-statistic filter along `dim`. `window` must be a positive odd integer; `rank` (0-based) selects the element of the sorted window (0 = min, `window//2` = median, `window-1` = max). Boundary: clamp |
| `sar.sort {dim}` | rank-1 or rank-2 float -> same shape; sorts each line along `dim` into ascending order (NumPy `sort`). The sort is a compare-exchange network over the static extent, so it carries no data-dependent control flow |

## Data movement

| Op | Semantics |
|----|-----------|
| `sar.transpose` | rank-2 corner turn |
| `sar.reverse {dim}` | element order reversed along one axis (frontend: `sar.flip`) |
| `sar.broadcast {dim}` | 1-D -> 2-D; the vector lies along axis `dim` |
| `sar.slice {offsets, sizes, strides}` | statically strided sub-tensor (NumPy basic slicing; frontend: `x[2:6, ::2]`) |
| `sar.dynamic_slice {sizes, strides}` | static-shape sub-tensor at runtime `i64[1]` offsets; offsets clamp so the complete slice remains in bounds |
| `sar.dynamic_update_slice` | copies a static-shape update into a tensor at clamped runtime offsets |
| `sar.concat {dim}` | concatenation of two tensors along `dim` |
| `sar.pad {low, high, value}` | constant padding per axis (NumPy `pad`) |
| `sar.fftshift {dim, inverse?}` | NumPy `fftshift`/`ifftshift` along one axis |

## Signal processing

| Op | Semantics |
|----|-----------|
| `sar.fft {dim}` | unscaled forward DFT along `dim` (NumPy convention); any size >= 2 on both backends (mixed radix-4/2 for powers of two, Bluestein's chirp-z reduction otherwise) |
| `sar.ifft {dim}` | inverse DFT scaled by `1/N` |
| `sar.fft_split {dim, inverse?}` | split-complex FFT on (re, im) float planes; produced by `sar-decomplexify`, not by the frontend |
| `sar.interp1d {dim?, kernel?, taps?, window?, beta?, boundary?}` | resampling along `dim` (default 1) at fractional `positions` (f64 tensor) with a selectable kernel: `nearest`, `linear`, `cubic` (Keys) or `sinc` (default: 8 taps); sinc taper: `rect`/`hann`/`hamming`/`kaiser(beta)`. `boundary` controls out-of-range taps: `zero` (default), `edge` (clamp), or `reflect` (mirror repeating the edge sample, i.e. NumPy `symmetric`). The orthogonal primitive behind Stolt remapping and RCMC |
| `sar.interp1d_split` | split-complex form of `sar.interp1d`; produced by `sar-decomplexify` |
| `sar.gather2d {kernel?, boundary?}` | 2-D gather at data-dependent positions: `out[i,j] = data[rows[i,j], cols[i,j]]` with both coordinates arbitrary functions of the output position (the access pattern of time-domain backprojection). `kernel`: `nearest` or `linear` (bilinear); `boundary`: `zero` (default) or `edge`. The output takes the position shape, independent of the data shape. On the HLS path, a row coordinate provably the output row plus a bounded displacement gathers through a sliding band of source rows (`DisplacementRange`), falling back to the resident plane otherwise |
| `sar.gather2d_split` | split-complex form of `sar.gather2d`; produced by `sar-decomplexify` |

## Compiled loops

| Op | Semantics |
|----|-----------|
| `sar.iterate {trips}` | counted loop with tensor-carried state: applies its region `trips` times, feeding each iteration's `sar.yield` to the next as block arguments. Stays a single loop in the design (a Python `for` unrolls at trace time). Carry types match position by position. With the `index` attribute the body's first block argument is the 0-based iteration index as `tensor<1xi64>`; it can directly drive `dynamic_slice` and `dynamic_update_slice`. Lowers to `scf.for` over tensors (`convert-sar-to-linalg`); the HLS path then demotes carries to side effects. Frontend: `sar.iterate(n, body, *carries, index=False)` |
| `sar.yield` | terminates one `sar.iterate` step with the next iteration's carried values |

Exact formulas are documented on the ops themselves
(`include/sar/Dialect/SAR/IR/SAROps.td`).

Anything expressible as a composition of these primitives lives in the
Python DSL layer instead of the IR: `sar.expj`, `sar.angle`, `sar.maximum`,
negation, `sar.multilook` and `sar.stolt_interp` all trace to primitive
graphs and rely on element-wise fusion for performance (see
[architecture.md](architecture.md)).

## Example

```mlir
func.func @range_compress(%raw: tensor<512x512xcomplex<f32>>,
                          %ref: tensor<512x512xcomplex<f32>>)
    -> tensor<512x512xcomplex<f32>> {
  %0 = sar.fft %raw {dim = 1 : i64} : tensor<512x512xcomplex<f32>>
  %1 = sar.fftshift %0 {dim = 1 : i64} : tensor<512x512xcomplex<f32>>
  %2 = sar.mul %1, %ref : tensor<512x512xcomplex<f32>>
  %3 = sar.fftshift %2 {dim = 1 : i64, inverse} : tensor<512x512xcomplex<f32>>
  %4 = sar.ifft %3 {dim = 1 : i64} : tensor<512x512xcomplex<f32>>
  return %4 : tensor<512x512xcomplex<f32>>
}
```

## Transformations and lowering paths

Passes:

- `--sar-decomplexify` (Transforms): complex tensors become (re, im) float
  plane pairs; complex arithmetic expands to real SAR ops; `fft`/`ifft`
  become `fft_split`. Enables targets without complex support.
- `--sar-verify-precision="precision=<native|f32|f64>"` (Transforms):
  checks function, operand, result, and block-argument types throughout the
  module. `f32` and `f64` require that exact floating-point width; `native`
  leaves declared types unchanged.
- `--convert-sar-to-linalg`: element-wise/structural ops to
  linalg-on-tensors. Signal ops are illegal here.
- `--convert-sar-signal-to-runtime`: `fft`/`ifft`/`interp1d` to
  `libsar_runtime` calls (`_mlir_ciface_sar_rt_*`).
- `--convert-sar-fft-to-affine`: `fft_split` to mixed radix-4/2 Stockham
  affine loop nests working on prefetched on-chip line blocks, with the
  lane loop innermost and banking hints on the buffers; non-power-of-two
  sizes go through Bluestein's chirp-z reduction. Why those two
  algorithms: [architecture.md](architecture.md#6-hls-backend).
- `--convert-sar-interp-to-affine`: `interp1d_split` to affine loops
  with the interpolation taps statically unrolled (the Kaiser window's
  Bessel `I0` expands into a straight-line power series); gathers use
  clamped
  data-dependent loads masked by selects (straight-line bodies, no control
  flow).
- `--sar-emit-c-interface`: attaches `llvm.emit_c_interface` to public
  functions (kernel entry points) so the ctypes launcher finds the
  `_mlir_ciface_*` wrappers; private declarations keep plain C symbols.
- `--sar-reuse-buffers` (Transforms): lets a buffer whose last use has
  passed carry a later one. `min-elements` selects which buffers take part
  -- below it a buffer is a dataflow channel, at and above it it is memory.
  `allow-retype` drops the requirement that the two agree on element type:
  footprints are compared in bytes, a retired buffer backing a newer one
  that fits inside it, and the group allocates `memref<Nxi8>` that each
  member reinterprets with `memref.view`. Only the CPU path enables it; the
  HLS emitter has no reinterpretation to emit. Buffers with a live alias
  (anything returning a memref or a tensor) and functions with branches are
  left alone either way.
- `--sar-fuse-elementwise` (Transforms): fuses an element-wise producer
  into every consumer above `min-elements`, recomputing rather than
  materialising a whole raster.
- `--sar-stage-transposes` (Transforms): stages a transposing loop nest
  through an on-chip block so both sides sweep contiguously. `block-bytes`
  bounds the block.
- `--sar-lower-copy` (Transforms): expands `memref.copy`, which lowers to
  a runtime-library call neither backend links, into an affine sweep.
- `--sar-demote-loop-carries` (Transforms): rewrites the buffer-carried
  `scf.for` a compiled loop bufferizes into as side effects -- the body
  iterates in the init buffer and a per-iteration copy replaces the yield
  -- because the HLS dataflow model forbids tasks with results. The affine
  pipeline runs it; the CPU path keeps the carried form.
- `--sar-privatize-out-params` (Transforms): moves computation that
  bufferization routed through a result out-parameter into a local
  buffer, writing the port once at the end. A top-level port with
  several writing stages would forfeit `#pragma HLS dataflow` for the
  design. The affine pipeline runs it; the CPU path keeps the in-place
  form.

Registered pipelines (see `lib/Pipelines/`):

- `--sar-to-llvm-pipeline`: CPU execution path (runtime calls + linalg
  fusion + OpenMP parallel loops + LLVM dialect).
- `--sar-to-linalg-pipeline`: linalg-on-tensors hand-off, for backends
  that run their own bufferization and loop transformations. The in-tree
  HLS backend does not use it -- it takes the affine hand-off below.
- `--sar-to-affine-pipeline`: split-complex affine/memref hand-off; this
  is what the HLS backend compiles from. Options: `fft-stage-group`
  (Stockham stages per scratch slot, 0 = full unroll),
  `fft-parallel-rows` (lanes per prefetched FFT line block),
  `fft-io-unroll` (elements per external access in the FFT transfer
  sweeps), `reuse-buffer-min-elements` and `recompute-min-elements` (the
  sharing and recompute thresholds of the passes above),
  `transpose-block-bytes` (staged corner-turn block size; 0 leaves
  transposes unstaged) and `interp-enable-banded-gather` (allow the
  banded interpolation gather).
- `--sar-affine-to-llvm-pipeline`: the affine path continued to LLVM, used
  to validate the Stockham lowering numerically on CPU.
