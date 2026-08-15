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
    are not handled here; run `convert-sar-signal-to-runtime` first for
    execution targets, or keep kernels free of them for HLS targets.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::complex::ComplexDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::math::MathDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def ConvertSARFFTToAffine : Pass<"convert-sar-fft-to-affine", "ModuleOp"> {
  let summary = "Lower split-complex FFTs to affine Stockham loop nests";
  let description = [{
    Converts `sar.fft_split` (produced by `sar-decomplexify`) into affine
    loop nests implementing the radix-2 Stockham autosort FFT with
    precomputed twiddle-factor globals. Stockham requires no bit-reversal,
    so all accesses are affine -- suitable for HLS flows.

    Sizes that are not powers of two are handled by Bluestein's chirp-z
    reduction: the DFT becomes a convolution carried by two padded
    Stockham transforms, with the chirp and the kernel spectrum folded
    into constant globals at compile time.
  }];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::memref::MemRefDialect",
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
  }];
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
