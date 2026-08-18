//===- EmitHLSCpp.h - scheduled HLS IR to Vitis HLS C++ ---------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_TARGET_HLS_EMITHLSCPP_H
#define SAR_TARGET_HLS_EMITHLSCPP_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace sar {

LogicalResult emitHLSCpp(ModuleOp module, llvm::raw_ostream &os);
void registerEmitHLSCppTranslation();

} // namespace sar
} // namespace mlir

#endif // SAR_TARGET_HLS_EMITHLSCPP_H
