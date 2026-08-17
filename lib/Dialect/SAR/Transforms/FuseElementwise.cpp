//===- FuseElementwise.cpp - Recompute-aware element-wise fusion ----------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "sar/Dialect/SAR/Transforms/Passes.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SARFUSEELEMENTWISE
#include "sar/Dialect/SAR/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;

namespace {

struct SARFuseElementwise
    : public sar::impl::SARFuseElementwiseBase<SARFuseElementwise> {
  using SARFuseElementwiseBase::SARFuseElementwiseBase;

  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);

    auto worthRecomputing = [&](OpOperand *fusedOperand) {
      Operation *producer = fusedOperand->get().getDefiningOp();
      if (!producer || !isa<linalg::GenericOp>(producer))
        return false;
      if (producer->getResult(0).hasOneUse())
        return true;
      auto type = dyn_cast<RankedTensorType>(fusedOperand->get().getType());
      return type && type.hasStaticShape() &&
             (uint64_t)type.getNumElements() >= minElements;
    };

    linalg::populateElementwiseOpsFusionPatterns(patterns, worthRecomputing);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
