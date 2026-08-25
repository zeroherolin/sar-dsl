//===- DistinctReturnBuffers.cpp - separate repeated memref results -------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "sar/Dialect/SAR/Transforms/Passes.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SARDISTINCTRETURNBUFFERS
#include "sar/Dialect/SAR/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;

namespace {
struct SARDistinctReturnBuffers
    : public sar::impl::SARDistinctReturnBuffersBase<SARDistinctReturnBuffers> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    func.walk([&](func::ReturnOp returnOp) {
      DenseSet<Value> seen;
      OpBuilder builder(returnOp);
      for (OpOperand &operand : returnOp->getOpOperands()) {
        Value value = operand.get();
        auto type = dyn_cast<MemRefType>(value.getType());
        if (!type || seen.insert(value).second)
          continue;

        SmallVector<Value> dynamicSizes;
        for (auto [dimension, extent] : llvm::enumerate(type.getShape()))
          if (ShapedType::isDynamic(extent))
            dynamicSizes.push_back(memref::DimOp::create(
                builder, returnOp.getLoc(), value, dimension));
        Value distinct = memref::AllocOp::create(builder, returnOp.getLoc(),
                                                 type, dynamicSizes);
        memref::CopyOp::create(builder, returnOp.getLoc(), value, distinct);
        operand.set(distinct);
        seen.insert(distinct);
      }
    });
  }
};
} // namespace
