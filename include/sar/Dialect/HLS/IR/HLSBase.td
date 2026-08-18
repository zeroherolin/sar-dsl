//===- HLSBase.td - HLS dialect base definitions --------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_IR_HLSBASE_TD
#define SAR_DIALECT_HLS_IR_HLSBASE_TD

include "mlir/IR/OpBase.td"

def HLSDialect : Dialect {
  let name = "hls";
  let summary = "The in-tree SAR-DSL HLS dialect";
  let description = [{
    Operations, types, and attributes for dataflow structure, memory placement,
    interfaces, and synthesis directives.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::sar::hls";
}

#endif // SAR_DIALECT_HLS_IR_HLSBASE_TD
