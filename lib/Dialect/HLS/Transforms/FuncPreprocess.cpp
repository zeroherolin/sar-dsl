//===- FuncPreprocess.cpp - func preprocess -------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_FUNCPREPROCESS
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

namespace {
/// Simple memref load to affine load raising.
struct MemrefLoadRaisePattern : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp load,
                                PatternRewriter &rewriter) const override {
    if (llvm::all_of(load.getIndices(), [&](Value operand) {
          return isValidDim(operand) || isValidSymbol(operand);
        })) {
      rewriter.replaceOpWithNewOp<AffineLoadOp>(load, load.getMemref(),
                                                load.getIndices());
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
/// Simple memref store to affine store raising.
struct MemrefStoreRaisePattern : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp store,
                                PatternRewriter &rewriter) const override {
    if (llvm::all_of(store.getIndices(), [&](Value operand) {
          return isValidDim(operand) || isValidSymbol(operand);
        })) {
      rewriter.replaceOpWithNewOp<AffineStoreOp>(
          store, store.getValue(), store.getMemref(), store.getIndices());
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
struct AffineStoreUndefFoldPattern
    : public OpRewritePattern<mlir::affine::AffineStoreOp> {
  using OpRewritePattern<mlir::affine::AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::affine::AffineStoreOp store,
                                PatternRewriter &rewriter) const override {
    if (store.getValueToStore().getDefiningOp<LLVM::UndefOp>()) {
      store.emitWarning("undef memory store is folded");
      rewriter.eraseOp(store);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
struct AllocaDemotePattern : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp alloca,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(alloca);
    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        alloca, alloca.getType(), alloca.getDynamicSizes(),
        alloca.getSymbolOperands(), alloca.getAlignmentAttr());
    return success();
  }
};
} // namespace

namespace {
/// Simple arith.addi to affine.apply raising that only supports dim + dim or
/// dim + constant.
struct AddIRaisePattern : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp add,
                                PatternRewriter &r) const override {
    r.setInsertionPoint(add);

    if (isValidDim(add.getLhs()) && isValidDim(add.getRhs())) {
      r.replaceOpWithNewOp<mlir::affine::AffineApplyOp>(
          add, r.getAffineDimExpr(0) + r.getAffineDimExpr(1),
          ValueRange({add.getLhs(), add.getRhs()}));
      return success();
    }

    if (auto rhs = add.getRhs().getDefiningOp<arith::ConstantIndexOp>();
        rhs && isValidDim(add.getLhs())) {
      r.replaceOpWithNewOp<mlir::affine::AffineApplyOp>(
          add, r.getAffineDimExpr(0) + rhs.value(), add.getLhs());
      return success();
    }

    if (auto lhs = add.getLhs().getDefiningOp<arith::ConstantIndexOp>();
        lhs && isValidDim(add.getRhs())) {
      r.replaceOpWithNewOp<mlir::affine::AffineApplyOp>(
          add, lhs.value() + r.getAffineDimExpr(0), add.getRhs());
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
/// Simple arith.muli to affine.apply raising that only supports dim * constant.
struct MulIRaisePattern : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern<arith::MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp mul,
                                PatternRewriter &r) const override {
    r.setInsertionPoint(mul);

    if (auto rhs = mul.getRhs().getDefiningOp<arith::ConstantIndexOp>();
        rhs && isValidDim(mul.getLhs())) {
      r.replaceOpWithNewOp<mlir::affine::AffineApplyOp>(
          mul, r.getAffineDimExpr(0) * rhs.value(), mul.getLhs());
      return success();
    }

    if (auto lhs = mul.getLhs().getDefiningOp<arith::ConstantIndexOp>();
        lhs && isValidDim(mul.getRhs())) {
      r.replaceOpWithNewOp<mlir::affine::AffineApplyOp>(
          mul, lhs.value() * r.getAffineDimExpr(0), mul.getRhs());
      return success();
    }
    return failure();
  }
};
} // namespace

LogicalResult sar::applyFuncPreprocess(func::FuncOp func, bool isTopFunc) {
  auto context = func.getContext();

  // HLS preprocessing requires a single-block function.
  if (!llvm::hasSingleElement(func)) {
    func.emitError("requires a single basic block");
    return failure();
  }

  // Set top function attribute.
  if (isTopFunc)
    setTopFuncAttr(func);

  // Set parallel attribute to each loop that is applicable. Meanwhile, strip
  // all loop directives.
  func.walk([&](AffineForOp loop) {
    loop->removeAttr("loop_directive");
    if (isLoopParallel(loop))
      setParallelAttr(loop);
  });

  mlir::RewritePatternSet patterns(context);
  patterns.add<MemrefLoadRaisePattern>(context);
  patterns.add<MemrefStoreRaisePattern>(context);
  patterns.add<AddIRaisePattern>(context);
  patterns.add<MulIRaisePattern>(context);
  patterns.add<AffineStoreUndefFoldPattern>(context);
  patterns.add<AllocaDemotePattern>(context);
  sar::populateBufferConversionPatterns(patterns);
  vector::populateVectorTransferLoweringPatterns(patterns);
  return applyPatternsGreedily(func, std::move(patterns));
}

namespace {
struct FuncPreprocess : public sar::impl::FuncPreprocessBase<FuncPreprocess> {
  FuncPreprocess() = default;
  FuncPreprocess(std::string hlsTopFunc) { topFunc = hlsTopFunc; }

  void runOnOperation() override {
    auto func = getOperation();
    auto isTop = func.getName() == topFunc;
    if (failed(applyFuncPreprocess(func, isTop)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> sar::createFuncPreprocessPass(std::string hlsTopFunc) {
  return std::make_unique<FuncPreprocess>(hlsTopFunc);
}
