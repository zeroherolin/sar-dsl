//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#ifndef SCALEHLS_TRANSLATION_EMITHLSCPP_H
#define SCALEHLS_TRANSLATION_EMITHLSCPP_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace sar {

LogicalResult emitHLSCpp(ModuleOp module, llvm::raw_ostream &os);
void registerEmitHLSCppTranslation();

} // namespace sar
} // namespace mlir

#endif // SCALEHLS_TRANSLATION_EMITHLSCPP_H
