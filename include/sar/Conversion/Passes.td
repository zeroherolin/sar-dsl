//===- Passes.td - SAR conversion passes -------------------*- tablegen -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_CONVERSION_PASSES_TD
#define SAR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertSARToLinalg : Pass<"convert-sar-to-linalg", "ModuleOp"> {
  let summary = "Lower SAR structural and element-wise ops to linalg";
  let description = [{
    Converts element-wise arithmetic, casts, transposes, broadcasts and
    fftshift operations of the `sar` dialect into linalg-on-tensors IR.
    Signal-processing operations (`sar.fft`, `sar.ifft`, `sar.interp1d`)
    are not handled here; lower them first through
    `convert-sar-signal-to-runtime` (execution targets) or the affine
    FFT/interpolation conversions (HLS targets).
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::complex::ComplexDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def ConvertSARFFTToAffine : Pass<"convert-sar-fft-to-affine", "ModuleOp"> {
  let summary = "Lower split-complex FFTs to affine Stockham loop nests";
  let description = [{
    Converts `sar.fft_split` (produced by `sar-decomplexify`) into affine
    loop nests implementing a radix-4/2 Stockham autosort FFT with precomputed
    twiddle-factor globals. Stockham requires no bit-reversal, so all accesses
    are affine -- suitable for HLS flows.

    Sizes that are not powers of two are handled by Bluestein's chirp-z
    reduction: the DFT becomes a convolution carried by two padded
    Stockham transforms, with the chirp and the kernel spectrum folded
    into constant globals at compile time.

    `fft-stage-group` controls how many consecutive Stockham stages share
    a set of scratch buffers:
      0 (default): every intermediate stage writes its own scratch line,
        so the log2(N) stages form a chain a dataflow backend can overlap
        across lines. This is the maximum-throughput, maximum-area point.
      k > 0: stages are packed into ceil(log2(N)/k) groups drawing from
        ceil(log2(N)/k)-1 scratch slots, which is what reduces on-chip
        area. Reusing a slot puts a write-after-read edge between the
        stages that share it, so the backend can no longer overlap them:
        area is traded for pipeline depth, not for arithmetic.

    Each stage keeps its own affine loop nest whatever the grouping. Stages
    cannot be fused into one nest: stage t+1 reads elements that many
    different iterations of stage t produce, so a shared iteration space
    would violate the dependence.

    Two slots are the floor whenever more than one stage writes scratch --
    a Stockham butterfly is not an in-place operation -- so k at or above
    log2(N) saturates at that floor rather than degenerating to one buffer.
    k=1 gives every stage its own slot, which is the same buffer count as
    the full unroll.
  }];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::memref::MemRefDialect",
  ];
  let options = [
    Option<"fftStageGroup", "fft-stage-group", "unsigned", /*default=*/"0",
           "Group Stockham stages: 0=full unroll, k>0=k stages per scratch "
           "buffer group">,
    Option<"fftParallelRows", "fft-parallel-rows", "unsigned",
           /*default=*/"0",
           "Power-of-two row unroll factor; 0 keeps one row in flight">
  ];
}

def ConvertSARInterpToAffine
    : Pass<"convert-sar-interp-to-affine", "ModuleOp"> {
  let summary = "Lower split-complex interpolation ops to affine loops";
  let description = [{
    Converts `sar.interp1d_split` (produced by `sar-decomplexify`)
    into affine loop nests with statically unrolled
    windowed-sinc taps. Gather indices are data-dependent memref loads with
    clamped addresses; out-of-range taps are masked with selects so the
    loop bodies remain straight-line and pipelineable by HLS flows.

    When `--enable-banded-gather` is set (the default) and displacement-range
    analysis can prove that `|positions[i,j] - j| <= D` for compile-time D,
    each output row reads only a width-W = ceil(D_hi)-floor(D_lo)+taps band
    of the source row (staged into a small on-chip buffer). This eliminates
    one full-raster AXI port on the HLS path. Set `--enable-banded-gather=0`
    to disable the optimisation and always emit the full-plane gather.
  }];
  let options = [
    Option<"enableBandedGather", "enable-banded-gather", "bool",
           /*default=*/"true",
           "Use banded gather when displacement range analysis succeeds">,
    Option<"bandedProfitThreshold", "banded-profit-threshold", "int64_t",
           /*default=*/"2",
           "Minimum ratio N/W required before emitting the banded path">,
  ];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::math::MathDialect",
    "::mlir::memref::MemRefDialect",
  ];
}

def ConvertSARSignalToRuntime
    : Pass<"convert-sar-signal-to-runtime", "ModuleOp"> {
  let summary = "Lower SAR signal-processing ops to SAR runtime library calls";
  let description = [{
    Converts `sar.fft`, `sar.ifft` and `sar.interp1d` into calls to the
    SAR runtime library (`libsar_runtime`). The generated calls use the
    MLIR C interface convention (`_mlir_ciface_*` symbols with memref
    descriptors passed by pointer).
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
  ];
}

#endif // SAR_CONVERSION_PASSES_TD
