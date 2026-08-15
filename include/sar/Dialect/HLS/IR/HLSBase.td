//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_IR_HLSBASE_TD
#define SAR_DIALECT_HLS_IR_HLSBASE_TD

include "mlir/IR/OpBase.td"

def HLSDialect : Dialect {
  let name = "hls";
  let summary = "An HLS out-of-tree MLIR dialect";
  let description = [{
    This dialect contains dedicated operations, types, and attributes designed
    for representing HLS-specific structures and components in MLIR, while
    enabling comprehensive optimization for both performance and area.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::sar::hls";
}

#endif // SAR_DIALECT_HLS_IR_HLSBASE_TD
