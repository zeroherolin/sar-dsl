//===- LowerCopy.cpp - Expand memref.copy into an affine sweep ------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// `memref.copy` lowers to a call into the MLIR runtime support library. A
// kernel this project emits links against neither that library (the CPU
// backend loads a bare shared object) nor anything like it (an HLS target
// has no runtime at all), so the call would only surface as an unresolved
// symbol at load time.
//
// Bufferization introduces the op wherever a value has to change layout --
// a strided view reaching a consumer that wants its own buffer. Expanding
// it into a loop nest keeps that case working and, on the HLS path, leaves
// a sweep the backend can pipeline rather than an opaque call.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "sar/Dialect/SAR/Transforms/Passes.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SARLOWERCOPY
#include "sar/Dialect/SAR/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;

namespace {

struct ExpandCopy : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<MemRefType>(copy.getSource().getType());
    if (!type || !type.hasStaticShape())
      return failure();

    Location loc = copy.getLoc();
    SmallVector<Value> indices;
    OpBuilder::InsertionGuard guard(rewriter);
    for (int64_t extent : type.getShape()) {
      auto loop = affine::AffineForOp::create(rewriter, loc, 0, extent);
      indices.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
    }
    Value element =
        affine::AffineLoadOp::create(rewriter, loc, copy.getSource(), indices);
    affine::AffineStoreOp::create(rewriter, loc, element, copy.getTarget(),
                                  indices);
    rewriter.eraseOp(copy);
    return success();
  }
};

struct SARLowerCopy : public sar::impl::SARLowerCopyBase<SARLowerCopy> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandCopy>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
