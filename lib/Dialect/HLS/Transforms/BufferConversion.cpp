//===- BufferConversion.cpp - memref allocations to dataflow buffers ------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Rewrites `memref.alloc`/`alloca`/`get_global` into the dialect's own
// buffer ops, which carry the memory kind and initial value a synthesis
// backend needs. Used by the passes that build the dataflow hierarchy.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"


using namespace mlir;
using namespace sar;
using namespace hls;

namespace {
// If an alloc is filled before any other uses, the alloc can be converted to a
// buffer with initial value.
template <typename OpType>
struct ConvertAllocToBufferWithInitValue : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    DominanceInfo DT;
    SmallVector<Operation *> users(op->user_begin(), op->user_end());
    if (users.empty())
      return failure();
    // `dominates` is reflexive, so it is not a strict weak ordering and
    // would make llvm::sort undefined. What this needs is the earliest
    // user, so pick it directly instead of sorting.
    Operation *first = users.front();
    for (Operation *user : llvm::drop_begin(users))
      if (DT.properlyDominates(user, first))
        first = user;

    if (auto fill = dyn_cast<linalg::FillOp>(first))
      if (auto constant = fill.value().getDefiningOp<arith::ConstantOp>()) {
        rewriter.replaceOpWithNewOp<BufferOp>(op, op.getType(),
                                              /*depth=*/1, constant.getValue());
        rewriter.eraseOp(fill);
        return success();
      }
    return failure();
  }
};
} // namespace

namespace {
template <typename OpType>
struct ConvertAllocToBuffer : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<BufferOp>(op, op.getType());
    return success();
  }
};
} // namespace

namespace {
struct ConvertGetGlobalToConstBuffer
    : public OpRewritePattern<memref::GetGlobalOp> {
  using OpRewritePattern<memref::GetGlobalOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::GetGlobalOp op,
                                PatternRewriter &rewriter) const override {
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        op, op.getNameAttr());
    rewriter.replaceOpWithNewOp<ConstBufferOp>(op, global.getType(),
                                               global.getConstantInitValue());
    return success();
  }
};
} // namespace

void sar::populateBufferConversionPatterns(RewritePatternSet &patterns) {
  auto context = patterns.getContext();
  patterns.add<ConvertAllocToBufferWithInitValue<memref::AllocOp>>(context);
  patterns.add<ConvertAllocToBufferWithInitValue<memref::AllocaOp>>(context);
  patterns.add<ConvertAllocToBuffer<memref::AllocOp>>(context);
  patterns.add<ConvertAllocToBuffer<memref::AllocaOp>>(context);
  patterns.add<ConvertGetGlobalToConstBuffer>(context);
}
