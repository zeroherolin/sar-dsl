//===- EmitCInterface.cpp - C wrappers for public functions ---------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "sar/Dialect/SAR/Transforms/Passes.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SAREMITCINTERFACE
#include "sar/Dialect/SAR/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;

namespace {

struct SAREmitCInterfacePass
    : sar::impl::SAREmitCInterfaceBase<SAREmitCInterfacePass> {
  void runOnOperation() override {
    for (auto func : getOperation().getOps<func::FuncOp>())
      if (func.isPublic())
        func->setAttr("llvm.emit_c_interface",
                      UnitAttr::get(func.getContext()));
  }
};

} // namespace
