//===- SimplifyCopy.cpp - simplify copy -----------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SIMPLIFYCOPY
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

#define DEBUG_TYPE "hls-simplify-copy"

using namespace mlir;
using namespace sar;
using namespace hls;

namespace {
struct SplitElementwiseGenericOp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (isElementwiseGenericOp(op) && op.getNumDpsInputs() == 1 &&
        op.getNumDpsInits() == 1) {
      auto &input = op->getOpOperand(0);
      auto &output = op->getOpOperand(1);
      if (input.get() == output.get())
        return failure();

      memref::CopyOp::create(rewriter, op.getLoc(), input.get(), output.get());
      input.set(output.get());
      return success();
    }
    return failure();
  }
};
} // namespace

static void findBufferUsers(Value memref, SmallVector<Operation *> &users) {
  for (auto user : memref.getUsers()) {
    if (auto viewOp = dyn_cast<ViewLikeOpInterface>(user))
      findBufferUsers(viewOp->getResult(0), users);
    else
      users.push_back(user);
  }
}

namespace {
struct SimplifyBufferCopy : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "\nCurrent copy: " << copy << "\n";);

    // If the source and target buffers are allocated in different memory space,
    // return failure.
    auto sourceType = cast<MemRefType>(copy.getSource().getType());
    auto targetType = cast<MemRefType>(copy.getTarget().getType());
    if (sourceType.getMemorySpaceAsInt() != targetType.getMemorySpaceAsInt())
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Located at the same memory space\n";);

    // Both the source and target buffers should be block arguments or defined
    // by BufferOp, otherwise return failure.
    auto source = findBuffer(copy.getSource());
    auto target = findBuffer(copy.getTarget());
    if (!source || !target)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Defined by block argument or BufferOp\n";);

    // If both the source and target buffers are block arguments, return failure
    // as either of them can be eliminated.
    auto sourceBuf = source.getDefiningOp<BufferOp>();
    auto targetBuf = target.getDefiningOp<BufferOp>();
    if (!sourceBuf && !targetBuf)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "At least one buffer is replaceable\n";);

    // Collect all users of the source and target buffer.
    SmallVector<Operation *> sourceUsers;
    SmallVector<Operation *> targetUsers;
    findBufferUsers(source, sourceUsers);
    findBufferUsers(target, targetUsers);

    // Collect the dominating and dominated buffer users.
    SmallVector<Operation *> sourceDomUsers;
    SmallVector<Operation *> sourcePostDomUsers;
    SmallVector<Operation *> targetDomUsers;
    SmallVector<Operation *> targetPostDomUsers;

    for (auto user : sourceUsers) {
      if (user == copy)
        continue;
      else if (crossRegionDominates(user, copy))
        sourceDomUsers.push_back(user);
      else
        sourcePostDomUsers.push_back(user);
    }

    for (auto user : targetUsers) {
      if (user == copy)
        continue;
      else if (crossRegionDominates(user, copy))
        targetDomUsers.push_back(user);
      else
        targetPostDomUsers.push_back(user);
    }

    // A helper to check whether any user has write effect.
    auto hasWriteUsers = [](SmallVector<Operation *> users) {
      return llvm::any_of(users, [](Operation *user) {
        return hasEffect<MemoryEffects::Write>(user) ||
               isa<StreamWriteOp>(user);
      });
    };

    // If the source buffer has write users dominated by the copy and the target
    // buffer has users dominated by the copy, or vice versa, the copy cannot be
    // eliminated.
    if ((hasWriteUsers(sourcePostDomUsers) && !targetPostDomUsers.empty()) ||
        (hasWriteUsers(targetPostDomUsers) && !sourcePostDomUsers.empty()))
      return failure();

    // If the source buffer has writers dominating the copy and the target
    // buffer has users dominating the copy, the copy cannot be eliminated.
    // Meanwhile, as long as the target buffer has users dominating the copy,
    // return failure.
    if ((hasWriteUsers(sourceDomUsers) && !targetDomUsers.empty()) ||
        hasWriteUsers(targetDomUsers))
      return failure();

    // If both the source and target buffer have users dominating the copy
    // (which should both be read only), the init value must be the same.
    if (!sourceDomUsers.empty() && !targetDomUsers.empty())
      if ((!sourceBuf || !targetBuf) ||
          sourceBuf.getInitValue() != targetBuf.getInitValue())
        return failure();

    LLVM_DEBUG(llvm::dbgs() << "Dominance is valid\n");

    auto sourceView = copy.getSource().getDefiningOp();
    auto targetView = copy.getTarget().getDefiningOp();
    DominanceInfo domInfo;

    // To replace the target buffer, the buffer must be directly defined by a
    // BufferOp without view. Meanwhile, the source view should either be a
    // block argument or dominate all users of the target buffer -- a
    // conservative dominance screen; moving the source view down to the
    // target buffer could legalize more cases at the cost of a placement
    // analysis.
    if (targetBuf && targetBuf == targetView &&
        (!sourceView || llvm::all_of(targetUsers, [&](Operation *user) {
          return domInfo.dominates(sourceView, user);
        }))) {
      LLVM_DEBUG(llvm::dbgs() << "Target and copy are erased\n");

      rewriter.replaceOp(targetBuf, copy.getSource());
      rewriter.eraseOp(copy);
      return success();
    }

    // Similarly, we need the same conditions to replace the source buffer.
    // One more screen first: when the target is a top-function port,
    // replacing a source that several operations write would hand the
    // port several writing processes -- exactly the multi-writer port
    // that forfeits `#pragma HLS dataflow` for the whole design (and what
    // sar-privatize-out-params exists to prevent). A single-writer source
    // may still merge; the port stays a one-writer channel.
    if (isa<BlockArgument>(target)) {
      unsigned writers = 0;
      for (auto user : sourceUsers)
        if (user != copy && hasEffect<MemoryEffects::Write>(user))
          ++writers;
      if (writers >= 2)
        return failure();
    }
    if (sourceBuf && sourceBuf == sourceView &&
        (!targetView || llvm::all_of(sourceUsers, [&](Operation *user) {
          return domInfo.dominates(targetView, user);
        }))) {
      // If the source buffer has initial value, the value must be pertained
      // by the target buffer after the replacement. Therefore, we have some
      // additional conditions here to check.
      if (sourceBuf.getInitValue()) {
        if (!targetBuf || (targetBuf.getInitValue() && targetBuf != targetView))
          return failure();
        targetBuf.setInitValueAttr(sourceBuf.getInitValue().value());
      }
      LLVM_DEBUG(llvm::dbgs() << "Source and copy are erased\n");

      rewriter.replaceOp(sourceBuf, copy.getTarget());
      rewriter.eraseOp(copy);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
struct SimplifyCopy : public sar::impl::SimplifyCopyBase<SimplifyCopy> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<SplitElementwiseGenericOp>(context);
    patterns.add<SimplifyBufferCopy>(context);
    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> sar::createSimplifyCopyPass() {
  return std::make_unique<SimplifyCopy>();
}
