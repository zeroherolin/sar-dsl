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
| `sar.atan2` | `(T, T) -> T` | float only; numpy argument order `(y, x)` |
| `sar.abs` | `(complex<t>|t) -> t` | complex magnitude / float abs |
| `sar.cmp` | `(t, t) -> t`, `predicate` | 0.0/1.0 mask (frontend: `x > y`, `x == 0.0`, ...) |
| `sar.where` | `(t, T, T) -> T` | exact per-element selection by a mask (numpy `where`) |
| `sar.conj` | `(complex<t>) -> complex<t>` | complex conjugate |
| `sar.real` / `sar.imag` | `(complex<t>) -> t` | plane extraction |
| `sar.complex` | `(t, t) -> complex<t>` | assemble from (re, im) planes |
| `sar.cast` | `(T) -> U` | float<->float, complex<->complex, float->complex, int<->int, int->float, float->int (truncation) |
| `sar.constant` | `() -> T` | dense elements attribute |

## Reductions

| Op | Semantics |
|----|-----------|
| `sar.reduce {kind, dim}` | rank-2 -> rank-1 along `dim`; `kind` is `sum` (float/complex) or `max`/`min` (float); rank-1 inputs are normalized to `1 x n` by the frontend |
| `sar.argmax {dim}` | rank-2 float -> rank-1 i64 indices (numpy `argmax`: first occurrence on ties) |

## Data movement

| Op | Semantics |
|----|-----------|
| `sar.transpose` | rank-2 corner turn |
| `sar.reverse {dim}` | element order reversed along one axis (frontend: `sar.flip`) |
| `sar.broadcast {dim}` | 1-D -> 2-D; the vector lies along axis `dim` |
| `sar.slice {offsets, sizes, strides}` | statically strided sub-tensor (numpy basic slicing; frontend: `x[2:6, ::2]`) |
| `sar.concat {dim}` | concatenation of two tensors along `dim` |
| `sar.pad {low, high, value}` | constant padding per axis (numpy `pad`) |
| `sar.fftshift {dim, inverse?}` | numpy `fftshift`/`ifftshift` along one axis |

## Signal processing

| Op | Semantics |
|----|-----------|
| `sar.fft {dim}` | unscaled forward DFT along `dim` (numpy convention); any size >= 2 on both backends (radix-2 where the size allows, Bluestein's chirp-z reduction otherwise) |
| `sar.ifft {dim}` | inverse DFT scaled by `1/N` |
| `sar.fft_split {dim, inverse?}` | split-complex FFT on (re, im) float planes; produced by `sar-decomplexify`, not by the frontend |
| `sar.interp1d {dim?, kernel?, taps?, window?, beta?}` | resampling along `dim` (default 1) at fractional `positions` (f64 tensor) with a selectable kernel: `nearest`, `linear`, `cubic` (Keys) or `sinc` (default: 8 taps); sinc taper: `rect`/`hann`/`hamming`/`kaiser(beta)`. The orthogonal primitive behind Stolt remapping and RCMC |
| `sar.interp1d_split` | split-complex form of `sar.interp1d`; produced by `sar-decomplexify` |

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
- `--convert-sar-to-linalg`: element-wise/structural ops to
  linalg-on-tensors. Signal ops are illegal here.
- `--convert-sar-signal-to-runtime`: `fft`/`ifft`/`interp1d` to
  `libsar_runtime` calls (`_mlir_ciface_sar_rt_*`).
- `--convert-sar-fft-to-affine`: `fft_split` to radix-2 Stockham affine
  loop nests; non-power-of-two sizes go through Bluestein's chirp-z
  reduction. Why those two algorithms:
  [architecture.md](architecture.md#6-hls-backend).
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
  passed carry a later one of the same type. `min-elements` selects which
  buffers take part -- below it a buffer is a dataflow channel, at and
  above it it is memory.
- `--sar-fuse-elementwise` (Transforms): fuses an element-wise producer
  into every consumer above `min-elements`, recomputing rather than
  materialising a whole raster.
- `--sar-stage-transposes` (Transforms): stages a transposing loop nest
  through an on-chip block so both sides sweep contiguously. `block-bytes`
  bounds the block.
- `--sar-lower-copy` (Transforms): expands `memref.copy`, which lowers to
  a runtime-library call neither backend links, into an affine sweep.

Registered pipelines (see `lib/Pipelines/`):

- `--sar-to-llvm-pipeline`: CPU execution path (runtime calls + linalg
  fusion + OpenMP parallel loops + LLVM dialect).
- `--sar-to-linalg-pipeline`: linalg-on-tensors hand-off, for backends
  that run their own bufferization and loop transformations. The in-tree
  HLS backend does not use it -- it takes the affine hand-off below.
- `--sar-to-affine-pipeline`: split-complex affine/memref hand-off; this
  is what the HLS backend compiles from.
- `--sar-affine-to-llvm-pipeline`: the affine path continued to LLVM, used
  to validate the Stockham lowering numerically on CPU.
