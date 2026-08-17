//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_BALANCEDATAFLOWNODE
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace sar;
using namespace hls;

namespace {
struct InsertCopyNode : public OpRewritePattern<NodeOp> {
  using OpRewritePattern<NodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NodeOp node,
                                PatternRewriter &rewriter) const override {
    if (!node.getLevel())
      return failure();

    for (auto output : node.getOutputs()) {
      if (isa<BlockArgument>(output) && node.getScheduleOp().isDependenceFree())
        continue;

      // An external buffer needs no balancing. The copy chain below exists to
      // hold on-chip data alive across pipeline levels, but a DRAM buffer is
      // already persistent -- a consumer several levels down simply reads it
      // later. Building the chain anyway would clone a full-size plane per
      // level, spending both the bandwidth of the copy and one more AXI
      // interface on each clone.
      if (isExtBuffer(output))
        continue;

      SmallVector<std::pair<unsigned, NodeOp>, 4> worklist;
      for (auto consumer : getDependentConsumers(output, node)) {
        // Unscheduled nodes carry no level; unsigned wrap on a reversed
        // pair would fabricate a huge diff, so compute signed.
        if (!node.getLevel() || !consumer.getLevel())
          continue;
        int64_t diff =
            (int64_t)*node.getLevel() - (int64_t)*consumer.getLevel();
        if (diff > 1)
          worklist.push_back({(unsigned)diff, consumer});
      }
      if (worklist.empty())
        continue;

      // Sort all consumers in a descending order of level difference.
      llvm::sort(worklist, [](auto a, auto b) { return a.first > b.first; });
      auto maxDiff = worklist.front().first;

      // A DRAM buffer could carry the pipeline balancing itself: giving it a
      // depth and letting each consumer read through a tap implements the
      // ping-pong in DRAM, saving interfaces and logic. That path is left
      // unfinished upstream -- ConvertDataflowToFunc drops the taps when it
      // turns nodes into functions, and the C++ emitter rejects any buffer
      // with a depth above one -- so balancing always goes through the
      // explicit buffer chain below, which is correct for on-chip and
      // external buffers alike.
      //
      // Reviving it means preserving the taps across the function conversion
      // and emitting `buf[depth][...]` with the tap folded into the index.

      // Construct a chain of buffers to hold data at each level and
      // construct explicit copies to pass data between different dataflow
      // levels.
      auto currentBuf = output;
      auto currentNode = node;
      for (unsigned i = 2; i <= maxDiff; ++i) {
        // Create a new buffer.
        auto loc = rewriter.getUnknownLoc();
        rewriter.setInsertionPoint(currentNode);
        auto newBuf =
            BufferOp::create(rewriter, loc, output.getType()).getMemref();

        // Construct a new node for data copy.
        rewriter.setInsertionPointAfter(currentNode);
        auto newNode = NodeOp::create(rewriter, loc, ValueRange(currentBuf),
                                      ValueRange(newBuf));
        newNode.setLevelAttr(
            rewriter.getI32IntegerAttr(node.getLevel().value() + 1 - i));
        auto block = rewriter.createBlock(&newNode.getBody());
        block->addArguments(TypeRange({currentBuf.getType(), newBuf.getType()}),
                            {currentBuf.getLoc(), newBuf.getLoc()});

        // Create an explicit copy operation.
        rewriter.setInsertionPointToStart(block);
        memref::CopyOp::create(rewriter, loc, block->getArgument(0),
                               block->getArgument(1));

        // Replace all uses at the current level.
        llvm::SmallDenseSet<Operation *, 4> consumers;
        while (!worklist.empty() && (worklist.back().first == i))
          consumers.insert(worklist.pop_back_val().second);
        output.replaceUsesWithIf(newBuf, [&](OpOperand &use) {
          return consumers.count(use.getOwner());
        });

        // Finally, we can update current buffer and current node.
        currentBuf = newBuf;
        currentNode = newNode;
      }
    }
    return success();
  }
};
} // namespace

namespace {
struct BalanceDataflowNode
    : public sar::impl::BalanceDataflowNodeBase<BalanceDataflowNode> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<InsertCopyNode>(context);
    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> sar::createBalanceDataflowNodePass() {
  return std::make_unique<BalanceDataflowNode>();
}
