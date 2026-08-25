//===- CreateDataflowFromAffine.cpp - build the dataflow hierarchy --------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"
#include "sar/Support/HLSHints.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_CREATEDATAFLOWFROMAFFINE
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

namespace {
struct TaskPartition : public OpRewritePattern<DispatchOp> {
  using OpRewritePattern<DispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DispatchOp dispatch,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(dispatch.getOps(), [](Operation &op) {
          return isa<bufferization::BufferizationDialect, tensor::TensorDialect,
                     linalg::LinalgDialect>(op.getDialect()) ||
                 isa<func::CallOp, DispatchOp, TaskOp, ScheduleOp, NodeOp>(op);
        }))
      return failure();
    auto &block = dispatch.getRegion().front();
    auto isTaskSeed = [](Operation &op) {
      // Compact unrolled lane loops are body, not tasks (see the band
      // filter in the pass entry).
      return isa<AffineForOp, scf::ForOp>(op) &&
             !op.hasAttr(kUnrollFactorAttr) && !op.hasAttr(kTaskBodyAttr);
    };
    if (llvm::none_of(block, isTaskSeed))
      return failure();

    // Fuse operations into dataflow tasks: each loop seeds a task and takes
    // the non-loop operations collected before it (anything else, including
    // a block-level `affine.if`, rides along into the adjacent task rather
    // than forming one of its own).
    SmallVector<Operation *, 4> opsToFuse;
    unsigned taskIdx = 0;
    for (auto &op : llvm::make_early_inc_range(block)) {
      if (hasEffect<MemoryEffects::Allocate>(&op)) {
        // Allocations stay outside tasks at the beginning of the block.
        op.moveBefore(&block, block.begin());

      } else if (isTaskSeed(op)) {
        // Each loop roots a task together with the operations collected before
        // it.
        opsToFuse.push_back(&op);
        fuseOpsIntoTask(opsToFuse, rewriter);
        opsToFuse.clear();
        taskIdx++;

      } else if (&op == block.getTerminator()) {
        // If the block will only generate one task, stop it.
        if (opsToFuse.empty() || taskIdx == 0)
          continue;
        fuseOpsIntoTask(opsToFuse, rewriter);
        opsToFuse.clear();
        taskIdx++;

      } else {
        opsToFuse.push_back(&op);
      }
    }
    return success();
  }
};
} // namespace

namespace {
struct CreateDataflowFromAffine
    : public sar::impl::CreateDataflowFromAffineBase<CreateDataflowFromAffine> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    dispatchBlock(&func.front());
    AffineLoopBands targetBands;
    getLoopBands(func.front(), targetBands, /*allowHavingChilds=*/true);
    for (auto &band : llvm::reverse(targetBands)) {
      // A body whose only loops are compact unrolled lanes is a leaf
      // compute body: the lane loop stands in for parallel copies of the
      // body, not for a task, and carving it out would put a call where
      // the enclosing loop needs a pipeline.
      auto *body = band.back().getBody();
      bool hasTaskableLoop = llvm::any_of(*body, [](Operation &op) {
        return isa<AffineForOp, scf::ForOp>(op) &&
               !op.hasAttr(kUnrollFactorAttr) && !op.hasAttr(kTaskBodyAttr);
      });
      if (hasTaskableLoop)
        dispatchBlock(body);
    }

    mlir::RewritePatternSet patterns(context);
    patterns.add<TaskPartition>(context);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> sar::createCreateDataflowFromAffinePass() {
  return std::make_unique<CreateDataflowFromAffine>();
}
