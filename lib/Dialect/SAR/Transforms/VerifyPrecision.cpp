//===- VerifyPrecision.cpp - Check backend precision contracts -----------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/STLExtras.h"

#include "sar/Dialect/SAR/Transforms/Passes.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SARVERIFYPRECISION
#include "sar/Dialect/SAR/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;

namespace {

static bool conflictsWith(Type type, unsigned width) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth() != width;
  if (auto complexType = dyn_cast<ComplexType>(type))
    return conflictsWith(complexType.getElementType(), width);
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return conflictsWith(shapedType.getElementType(), width);
  if (auto functionType = dyn_cast<FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(),
                        [width](Type input) {
                          return conflictsWith(input, width);
                        }) ||
           llvm::any_of(functionType.getResults(), [width](Type result) {
             return conflictsWith(result, width);
           });
  }
  return false;
}

struct VerifyPrecision
    : public sar::impl::SARVerifyPrecisionBase<VerifyPrecision> {
  using SARVerifyPrecisionBase::SARVerifyPrecisionBase;

  void runOnOperation() override {
    StringRef policy = precision;
    if (policy == "native")
      return;
    unsigned width = policy == "f32" ? 32 : policy == "f64" ? 64 : 0;
    if (!width) {
      getOperation().emitError()
          << "precision must be one of native, f32 or f64; got " << policy;
      return signalPassFailure();
    }

    WalkResult result = getOperation().walk([&](Operation *op) {
      auto reject = [&](Type type, StringRef role) {
        if (!conflictsWith(type, width))
          return false;
        op->emitOpError() << role << " type " << type
                          << " violates precision=" << policy;
        return true;
      };

      if (auto function = dyn_cast<FunctionOpInterface>(op))
        if (reject(function.getFunctionType(), "function"))
          return WalkResult::interrupt();
      for (Type type : op->getOperandTypes())
        if (reject(type, "operand"))
          return WalkResult::interrupt();
      for (Type type : op->getResultTypes())
        if (reject(type, "result"))
          return WalkResult::interrupt();
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            if (reject(argument.getType(), "block argument"))
              return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
