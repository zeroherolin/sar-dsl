//===- ReduceInitiationInterval.cpp - shorten loop-carried chains ---------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/IR/Utils.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_REDUCEINITIATIONINTERVAL
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

#define DEBUG_TYPE "hls-reduce-initiation-interval"

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace sar::hls;

/// Find a chain of commutative operators starting from "headOps" and ended with
/// the "store". "headOps" is a list of single-use operations starting with "op"
/// that can be moved together during the transformation.
static bool findCommutativeChain(Operation *op, AffineWriteOpInterface store,
                                 SmallVectorImpl<Operation *> &headOps,
                                 SmallVectorImpl<Operation *> &chainOps) {
  assert((chainOps.empty() || op == chainOps.back()) && "invalid chaining");

  if (llvm::any_of(op->getUsers(), [&](auto user) { return user == store; }))
    return true;

  for (auto user : op->getUsers()) {
    // Reordering floating-point operations changes rounding, signed-zero and
    // NaN behavior. Only exact integer/index arithmetic is legal here; an
    // explicitly budgeted floating-point reassociation belongs in a separate
    // numerical optimization.
    bool exactType = llvm::all_of(user->getResultTypes(), [](Type type) {
      return isa<IntegerType, IndexType>(type);
    });
    if (exactType && user->hasTrait<OpTrait::IsCommutative>() &&
        (chainOps.empty() || op->getName() == user->getName())) {
      LLVM_DEBUG(llvm::dbgs() << "Chain op: " << *user << "\n");

      chainOps.push_back(user);
      return findCommutativeChain(user, store, headOps, chainOps);
    }
  }

  if (op->hasOneUse()) {
    auto user = *op->user_begin();
    if (user->hasOneUse() && chainOps.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Head op: " << *user << "\n");

      headOps.push_back(user);
      return findCommutativeChain(user, store, headOps, chainOps);
    }
  }
  return false;
}

/// Moves loop-invariant heads ahead of the carried value in an associative
/// integer/index chain. This shortens the recurrence without reordering the
/// heads or changing floating-point evaluation.
static bool optimizeCommutativeChain(SmallVectorImpl<Operation *> &headOps,
                                     SmallVectorImpl<Operation *> &chainOps,
                                     PatternRewriter &rewriter) {
  if (headOps.empty() || chainOps.empty())
    return false;

  // Get the first operator of the chain and the target operand to operate on.
  auto firstOp = *chainOps.begin();
  auto &targetOperand =
      firstOp->getOperand(0) == headOps.back()->getUses().begin()->get()
          ? firstOp->getOpOperand(1)
          : firstOp->getOpOperand(0);

  // The operator the chain can be rebalanced ahead of.
  Operation *targetOp = chainOps.back();
  for (auto op : chainOps)
    if (!op->getResult(0).hasOneUse()) {
      targetOp = op;
      break;
    }
  if (targetOp == firstOp)
    return false;

  // Move the head ops and first commutative operator after the target operator.
  rewriter.setInsertionPointAfter(targetOp);
  for (auto op : headOps) {
    op->remove();
    rewriter.insert(op);
  }
  firstOp->remove();
  rewriter.insert(firstOp);

  // Reconnect the chain.
  firstOp->getResult(0).replaceAllUsesWith(targetOperand.get());
  targetOp->getResult(0).replaceAllUsesWith(firstOp->getResult(0));
  targetOperand.set(targetOp->getResult(0));
  return true;
}

namespace {
struct ReduceInitiationIntervalPattern : public OpRewritePattern<AffineForOp> {
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineForOp loop,
                                PatternRewriter &rewriter) const override {
    bool hasChanged = false;
    MemAccessesMap map;
    for (auto &op : *loop.getBody()) {
      if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
        map[MemRefAccess(&op).memref].push_back(&op);
    }

    // Traverse all buffer accesses in the loop body.
    for (auto pair : map) {
      auto accesses = pair.second;

      // Only if a load depends on a dominated store (a back dependence), the
      // associated II constraint is possible to be optimized.
      for (unsigned i = 0, e = accesses.size(); i < e; ++i) {
        auto dstLoad = dyn_cast<AffineReadOpInterface>(accesses[i]);
        if (!dstLoad)
          continue;
        // Moving the load rests on the conservative assumption that the
        // load op only has one use.
        if (!dstLoad->hasOneUse())
          break;
        LLVM_DEBUG(llvm::dbgs() << "\n==========Load: " << dstLoad << "\n");

        for (unsigned j = i + 1, e = accesses.size(); j < e; ++j) {
          auto srcStore = dyn_cast<AffineWriteOpInterface>(accesses[j]);
          if (!srcStore || MemRefAccess(srcStore) != MemRefAccess(dstLoad))
            continue;
          LLVM_DEBUG(llvm::dbgs() << "Store: " << srcStore << "\n");

          SmallVector<Operation *, 32> chainOps;
          SmallVector<Operation *, 4> headOps({dstLoad});
          if (findCommutativeChain(dstLoad, srcStore, headOps, chainOps))
            if (optimizeCommutativeChain(headOps, chainOps, rewriter)) {
              LLVM_DEBUG(llvm::dbgs() << "Optimization succeeded\n");
              hasChanged = true;
            }

          // Only the first dominated store can start this chain.
          break;
        }
      }
    }
    return success(hasChanged);
  }
};
} // namespace

namespace {
struct ReduceInitiationInterval
    : public sar::impl::ReduceInitiationIntervalBase<ReduceInitiationInterval> {
  void runOnOperation() override {
    auto func = getOperation();
    mlir::RewritePatternSet patterns(func.getContext());
    patterns.add<ReduceInitiationIntervalPattern>(func.getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns),
                                     GreedyRewriteConfig()
                                         .setUseTopDownTraversal(false)
                                         .setMaxIterations(2))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> sar::createReduceInitiationIntervalPass() {
  return std::make_unique<ReduceInitiationInterval>();
}
