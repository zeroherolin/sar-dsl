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
| `sar.max_scalar` | `(T) -> T`, `scalar: f64` | float only (`np.maximum(x, s)`) |
| `sar.neg` | `(T) -> T` | float/complex |
| `sar.sqrt` | `(T) -> T` | float only |
| `sar.cos` / `sar.sin` | `(T) -> T` | float only |
| `sar.abs` | `(complex<t>|t) -> t` | complex magnitude / float abs |
| `sar.expj` | `(t) -> complex<t>` | `cos(x) + j sin(x)` |
| `sar.cast` | `(T) -> U` | float<->float, complex<->complex, float->complex |
| `sar.constant` | `() -> T` | dense elements attribute |

## Data movement

| Op | Semantics |
|----|-----------|
| `sar.transpose` | rank-2 corner turn |
| `sar.broadcast {dim}` | 1-D -> 2-D; the vector lies along axis `dim` |
| `sar.fftshift {dim, inverse?}` | numpy `fftshift`/`ifftshift` along one axis |

## Signal processing

| Op | Semantics |
|----|-----------|
| `sar.fft {dim}` | unscaled forward DFT along `dim` (numpy convention); size must be a power of two |
| `sar.ifft {dim}` | inverse DFT scaled by `1/N` |
| `sar.fft_split {dim, inverse?}` | split-complex FFT on (re, im) float planes; produced by `sar-decomplexify`, not by the frontend |
| `sar.interp1d` | windowed-sinc resampling of each row at fractional `positions` (f64 tensor); the orthogonal primitive behind Stolt remapping and RCMC |
| `sar.interp1d_split` / `sar.stolt_interp_split` | split-complex forms of the interpolation ops; produced by `sar-decomplexify` |
| `sar.stolt_interp {c, fc, vr, t_shift}` | omega-K Stolt remapping (a fused special case of position computation + `sar.interp1d`); operands: 2-D complex spectrum, `fa` (azimuth axis, f64), `fr` (range axis, f64) |

Exact formulas are documented on the ops themselves
(`include/sar/Dialect/SAR/IR/SAROps.td`).

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
- `--convert-sar-signal-to-runtime`: `fft`/`ifft`/`stolt_interp`/`interp1d`
  to `libsar_runtime` calls (`_mlir_ciface_sar_rt_*`).
- `--convert-sar-fft-to-affine`: `fft_split` to radix-2 **Stockham** affine
  loop nests with constant twiddle globals (no bit-reversal, fully affine
  accesses -- HLS-friendly).
- `--convert-sar-interp-to-affine`: the split interpolation ops to affine
  loops with statically unrolled windowed-sinc taps; gathers use clamped
  data-dependent loads masked by selects (straight-line bodies, no control
  flow).

Registered pipelines (see `lib/Pipelines/`):

- `--sar-to-llvm-pipeline`: CPU execution path (runtime calls + linalg
  fusion + OpenMP parallel loops + LLVM dialect).
- `--sar-to-linalg-pipeline`: linalg-on-tensors hand-off (HLS, float
  element-wise kernels).
- `--sar-to-affine-pipeline`: split-complex affine/memref hand-off (HLS
  kernels with complex arithmetic and FFTs).
- `--sar-affine-to-llvm-pipeline`: the affine path continued to LLVM, used
  to validate the Stockham lowering numerically on CPU.
