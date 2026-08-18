//===- ScheduleDataflowNode.cpp - schedule dataflow node ------------------===//
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
#define GEN_PASS_DEF_SCHEDULEDATAFLOWNODE
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace sar;
using namespace hls;

namespace {
struct ALAPScheduleNode : public OpRewritePattern<NodeOp> {
  ALAPScheduleNode(MLIRContext *context, bool ignoreViolations)
      : OpRewritePattern<NodeOp>(context), ignoreViolations(ignoreViolations) {}

  LogicalResult matchAndRewrite(NodeOp node,
                                PatternRewriter &rewriter) const override {
    if (node.getLevel())
      return failure();

    DominanceInfo domInfo;
    unsigned level = 0;
    for (auto output : node.getOutputs()) {
      // Stop to schedule the node if an internal buffer has multi-producer or
      // multi-consumer violation. DRAM buffer is not considered - the
      // dependencies associated with them are handled later by tokens.
      if (!isExtBuffer(output) && !ignoreViolations)
        if (getDependentConsumers(output, node).size() > 1 ||
            getProducers(output).size() > 1)
          return failure();

      for (auto consumer : getDependentConsumers(output, node)) {
        if (!consumer.getLevel())
          return failure();
        level = std::max(level, consumer.getLevel().value() + 1);
      }
    }
    node.setLevelAttr(rewriter.getI32IntegerAttr(level));
    return success();
  }

private:
  bool ignoreViolations;
};
} // namespace

namespace {
struct ScheduleDataflowNode
    : public sar::impl::ScheduleDataflowNodeBase<ScheduleDataflowNode> {
  ScheduleDataflowNode() = default;
  explicit ScheduleDataflowNode(bool argIgnoreViolations) {
    ignoreViolations = argIgnoreViolations;
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<ALAPScheduleNode>(context, ignoreViolations.getValue());
    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass>
sar::createScheduleDataflowNodePass(bool ignoreViolations) {
  return std::make_unique<ScheduleDataflowNode>(ignoreViolations);
}
