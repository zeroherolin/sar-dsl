//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_ELIMINATEMULTICONSUMER
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir


using namespace mlir;
using namespace sar;
using namespace hls;

namespace {
struct InsertForkNode : public OpRewritePattern<NodeOp> {
  using OpRewritePattern<NodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NodeOp node,
                                PatternRewriter &rewriter) const override {
    auto loc = rewriter.getUnknownLoc();
    DominanceInfo domInfo;

    auto hasChanged = false;
    for (auto output : node.getOutputs()) {
      // DRAM buffer is not considered - the dependencies associated with them
      // are handled later by tokens.
      if (isExtBuffer(output))
        continue;

      // A buffer may be reused as scratch storage by several producers in
      // turn. Only consumers of the value written by THIS node - located
      // after the node and before any subsequent redefinition - may be
      // redirected to a fork; consumers of other generations must keep
      // reading the original buffer.
      auto producers = getProducersExcept(output, node);
      SmallVector<NodeOp> consumers;
      for (auto consumer : getDependentConsumers(output, node)) {
        if (!domInfo.properlyDominates(node, consumer))
          continue;
        auto overwritten = llvm::any_of(producers, [&](NodeOp producer) {
          return domInfo.properlyDominates(node, producer) &&
                 domInfo.properlyDominates(producer, consumer);
        });
        if (!overwritten)
          consumers.push_back(consumer);
      }
      if (consumers.size() < 2)
        continue;

      hasChanged = true;
      rewriter.setInsertionPointAfter(node);
      SmallVector<Value> buffers;
      SmallVector<Location> bufferLocs;

      // Insert a buffer for each consumer.
      for (auto consumer : consumers) {
        auto buffer = rewriter.create<BufferOp>(loc, output.getType());
        output.replaceUsesWithIf(
            buffer, [&](OpOperand &use) { return use.getOwner() == consumer; });
        buffers.push_back(buffer);
        bufferLocs.push_back(loc);
      }

      // Create a new fork node.
      auto fork = rewriter.create<NodeOp>(loc, output, buffers);
      auto block = rewriter.createBlock(&fork.getBody());
      auto outputArg = block->addArgument(output.getType(), output.getLoc());
      auto bufferArgs = block->addArguments(ValueRange(buffers), bufferLocs);

      // Create explicit copy from the original output to the buffers.
      rewriter.setInsertionPointToStart(block);
      for (auto bufferArg : bufferArgs)
        rewriter.create<memref::CopyOp>(loc, outputArg, bufferArg);
    }
    return success(hasChanged);
  }
};
} // namespace

namespace {
struct EliminateMultiConsumer
    : public sar::impl::EliminateMultiConsumerBase<EliminateMultiConsumer> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<InsertForkNode>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> sar::createEliminateMultiConsumerPass() {
  return std::make_unique<EliminateMultiConsumer>();
}
