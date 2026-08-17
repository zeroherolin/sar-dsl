//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_LOWERCOPYTOAFFINE
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

namespace {
struct LowerCopy : public OpRewritePattern<memref::CopyOp> {
  LowerCopy(MLIRContext *context, bool internalCopyOnly = true)
      : OpRewritePattern(context), internalCopyOnly(internalCopyOnly) {}

  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    // Check whether the copy op communicates with inputs or outputs.
    auto isExternalCopy =
        isExtBuffer(copy.getSource()) || isExtBuffer(copy.getTarget());

    // Return failure if we don't need to lower external copies.
    if (internalCopyOnly && isExternalCopy)
      return failure();

    rewriter.setInsertionPoint(copy);
    auto loc = copy.getLoc();
    auto memrefType = cast<MemRefType>(copy.getSource().getType());

    // Create explicit memory copy using an affine loop nest.
    SmallVector<Value, 4> ivs;
    auto constantZero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (auto dimSize : memrefType.getShape()) {
      if (dimSize == 1) {
        ivs.push_back(constantZero);
        continue;
      }
      auto loop = mlir::affine::AffineForOp::create(rewriter, loc, 0, dimSize);
      setParallelAttr(loop);
      // If the copy op is not external, we consider the loop as point loop
      // that needs to be optimized later.
      if (!isExternalCopy)
        setPointAttr(loop);
      rewriter.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
    }

    // Create affine load/store operations.
    auto value = mlir::affine::AffineLoadOp::create(rewriter, loc,
                                                    copy.getSource(), ivs);
    mlir::affine::AffineStoreOp::create(rewriter, loc, value, copy.getTarget(),
                                        ivs);

    rewriter.eraseOp(copy);
    return success();
  }

private:
  bool internalCopyOnly = true;
};
} // namespace

namespace {
struct LowerCopyToAffine
    : public sar::impl::LowerCopyToAffineBase<LowerCopyToAffine> {
  LowerCopyToAffine() = default;
  LowerCopyToAffine(bool argInternalCopyOnly) {
    internalCopyOnly = argInternalCopyOnly;
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();
    auto DT = DominanceInfo(module);

    // Lower copy operation.
    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerCopy>(context, internalCopyOnly);
    (void)applyPatternsGreedily(module, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> sar::createLowerCopyToAffinePass(bool internalCopyOnly) {
  return std::make_unique<LowerCopyToAffine>(internalCopyOnly);
}
