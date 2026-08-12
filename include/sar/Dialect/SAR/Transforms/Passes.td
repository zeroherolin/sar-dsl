//===- Passes.td - SAR dialect transformation passes -------*- tablegen -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_SAR_TRANSFORMS_PASSES_TD
#define SAR_DIALECT_SAR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SARDecomplexify : Pass<"sar-decomplexify", "ModuleOp"> {
  let summary = "Split complex tensors into real/imaginary float planes";
  let description = [{
    Rewrites kernels so that no complex element types remain: every complex
    tensor value becomes a pair of float tensors and complex arithmetic is
    expanded into real arithmetic. Function signatures change accordingly
    (each complex argument/result is replaced by adjacent `re, im` float
    tensors, in order).

    This enables targets without complex-number support -- notably HLS
    C++ emission through ScaleHLS-HIDA. Signal and interpolation ops map
    to their split-complex forms: `sar.fft`/`sar.ifft` become
    `sar.fft_split`, `sar.interp1d` becomes `sar.interp1d_split` and
    `sar.stolt_interp` becomes `sar.stolt_interp_split`.
  }];
  let dependentDialects = ["::mlir::func::FuncDialect"];
}

#endif // SAR_DIALECT_SAR_TRANSFORMS_PASSES_TD
