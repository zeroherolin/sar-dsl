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
template <typename OpType>
struct BufferizeDispatchOrTask : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    bool hasChanged = false;

    for (auto result : op->getResults()) {
      if (auto tensorType = dyn_cast<TensorType>(result.getType())) {
        auto memrefType =
            MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        result.setType(memrefType);

        auto loc = rewriter.getUnknownLoc();
        rewriter.setInsertionPointAfter(op);
        auto tensor = rewriter.template create<bufferization::ToTensorOp>(
            loc, tensorType, result);
        result.replaceAllUsesExcept(tensor, tensor);

        rewriter.setInsertionPoint(op.getYieldOp());
        auto output = op.getYieldOp().getOperand(result.getResultNumber());
        auto memref = rewriter.template create<bufferization::ToBufferOp>(
            loc, memrefType, output);
        op.getYieldOp()->getOpOperand(result.getResultNumber()).set(memref);
        hasChanged = true;
      }
    }
    return success(hasChanged);
  }
};
} // namespace

namespace {
struct BufferizeTensorEmpty : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorType = op.getType();
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    rewriter.setInsertionPoint(op);
    auto buffer = rewriter.create<BufferOp>(op.getLoc(), memrefType);
    rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, tensorType,
                                                           buffer);
    return success();
  }
};
} // namespace

namespace {
template <typename OpType>
struct HoistBuffer : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    for (auto &result : op.getYieldOp()->getOpOperands())
      if (auto buffer =
              result.get().template getDefiningOp<BufferLikeInterface>())
        if (op == buffer->getParentOp()) {
          buffer->moveBefore(op);
          op.getResult(result.getOperandNumber())
              .replaceAllUsesWith(buffer.getMemref());
          return success();
        }
    return failure();
  }
};
} // namespace

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
    llvm::sort(users, [&](auto a, auto b) { return DT.dominates(a, b); });

    if (auto fill = dyn_cast<linalg::FillOp>(users.front()))
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
