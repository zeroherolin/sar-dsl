//===- SARDialect.td - SAR dialect definition --------------*- tablegen -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_SAR_IR_SARDIALECT_TD
#define SAR_DIALECT_SAR_IR_SARDIALECT_TD

include "mlir/IR/OpBase.td"

def SAR_Dialect : Dialect {
  let name = "sar";
  let summary = "High-level operations for SAR imaging pipelines";
  let description = [{
    The `sar` dialect models synthetic aperture radar (SAR) imaging
    computations at the granularity of whole-array operations: element-wise
    arithmetic, axis-wise FFT/IFFT, spectrum shifts, corner turns and
    domain-specific kernels such as Stolt interpolation.

    Design notes:
    - Operations work directly on builtin ranked tensors with static shapes
      (like TOSA/StableHLO); the dialect intentionally does not define its
      own container types.
    - Complex data uses the builtin `complex<f32>` / `complex<f64>` element
      types.
    - The dialect stays backend-agnostic. Lowering pipelines decide whether
      an operation becomes linalg IR, loops, or a call into the SAR runtime
      library.
  }];
  let cppNamespace = "::mlir::sar";
  let hasConstantMaterializer = 1;
}

#endif // SAR_DIALECT_SAR_IR_SARDIALECT_TD
