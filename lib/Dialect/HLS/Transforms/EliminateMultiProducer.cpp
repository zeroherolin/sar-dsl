//===- EliminateMultiProducer.cpp - privatize multi-writer buffers --------===//
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
#define GEN_PASS_DEF_ELIMINATEMULTIPRODUCER
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

namespace {
static Operation *topLevelInNode(Operation *operation, NodeOp node) {
  Block *body = &node.getBody().front();
  while (operation->getBlock() != body)
    operation = operation->getParentOp();
  return operation;
}

static bool isCompleteWriteWithinNode(Operation *operation, Value buffer,
                                      NodeOp node) {
  auto store = dyn_cast<AffineStoreOp>(operation);
  auto type = dyn_cast<MemRefType>(buffer.getType());
  if (!store || store.getMemRef() != buffer || !type ||
      !type.hasStaticShape() || !store.getAffineMap().isIdentity() ||
      store.getMapOperands().size() != (unsigned)type.getRank())
    return false;

  SmallVector<AffineForOp> loops;
  for (Operation *parent = operation->getParentOp(); parent != node;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast_or_null<AffineForOp>(parent);
    if (!loop)
      return false;
    loops.push_back(loop);
  }
  std::reverse(loops.begin(), loops.end());
  if (loops.size() != (unsigned)type.getRank())
    return false;
  for (auto [dimension, loop] : llvm::enumerate(loops))
    if (store.getMapOperands()[dimension] != loop.getInductionVar() ||
        !loop.hasConstantBounds() || loop.getConstantLowerBound() != 0 ||
        loop.getConstantUpperBound() != type.getDimSize(dimension) ||
        loop.getStep() != 1)
      return false;
  return true;
}

/// Whether a complete write of `buffer` finishes before every read.
///
/// A privatized in-place output only needs the old buffer when some element
/// can be read before the node defines it. A full row-major sweep in an
/// earlier top-level loop initializes every element, so copying the old
/// contents would be both redundant and, if placed near a later read,
/// destructive.
static bool fullyWrittenBeforeReads(NodeOp node, BlockArgument buffer,
                                    ArrayRef<OpOperand *> reads) {
  SmallVector<Operation *> completeWriters;
  for (OpOperand &use : buffer.getUses()) {
    if (!isWritten(use))
      continue;
    Operation *writer = use.getOwner();
    if (isCompleteWriteWithinNode(writer, buffer, node))
      completeWriters.push_back(topLevelInNode(writer, node));
  }
  return llvm::any_of(completeWriters, [&](Operation *writer) {
    return llvm::all_of(reads, [&](OpOperand *read) {
      Operation *reader = topLevelInNode(read->getOwner(), node);
      return writer != reader && writer->isBeforeInBlock(reader);
    });
  });
}

struct BufferMultiProducer : public OpRewritePattern<ScheduleOp> {
  using OpRewritePattern<ScheduleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScheduleOp schedule,
                                PatternRewriter &rewriter) const override {
    DominanceInfo DT(schedule);
    auto loc = rewriter.getUnknownLoc();
    bool hasChanged = false;

    SmallVector<Value> buffers;
    for (auto bufferOp : schedule.getOps<BufferOp>())
      buffers.push_back(bufferOp);

    for (auto buffer : buffers) {
      // An external buffer is memory, not a dataflow channel: several nodes
      // writing it in turn is how a chain reuses DRAM instead of holding one
      // full-size plane per intermediate, and the ordering between them is
      // carried by tokens rather than by the channel. Privatizing it would
      // undo that sharing and cost one plane -- and one interface -- per
      // producer. Scheduling exempts external buffers for the same reason.
      if (isExtBuffer(buffer))
        continue;

      SmallVector<NodeOp, 4> producers(getProducers(buffer));
      if (producers.size() <= 1)
        continue;
      hasChanged = true;

      // Drop the dominating/leading producer, which doesn't need to be
      // transformed.
      llvm::sort(producers, [&](NodeOp a, NodeOp b) {
        return DT.properlyDominates(b, a);
      });
      producers.pop_back();

      for (auto node : producers) {
        unsigned originalNumInputs = node.getNumInputs();
        rewriter.setInsertionPoint(node);

        // Create a new buffer and write to it instead of the original one.
        auto newBuffer = BufferOp::create(rewriter, loc, buffer.getType());
        auto bufferIdx = llvm::find(node.getOutputs(), buffer) -
                         node.getOutputs().begin() + originalNumInputs;
        node.setOperand(bufferIdx, newBuffer);

        buffer.replaceUsesWithIf(newBuffer, [&](OpOperand &use) {
          if (auto user = dyn_cast<NodeOp>(use.getOwner()))
            return DT.properlyDominates(node, user);
          return false;
        });

        // Create a new node and erase the original one.
        auto newNode = NodeOp::create(
            rewriter, node.getLoc(), node.getInputs(), node.getOutputs(),
            node.getParams(), node.getInputTapsAttr(), node.getLevelAttr());
        rewriter.inlineRegionBefore(node.getBody(), newNode.getBody(),
                                    newNode.getBody().end());
        rewriter.eraseOp(node);

        auto newBufferArg = newNode.getBody().getArgument(bufferIdx);
        SmallVector<OpOperand *> readUses;
        for (OpOperand &use : newBufferArg.getUses())
          if (isRead(use))
            readUses.push_back(&use);
        if (readUses.empty() ||
            fullyWrittenBeforeReads(newNode, newBufferArg, readUses))
          continue;

        // This producer needs the prior contents. Rebuild it with the original
        // buffer appended to the input segment and add the matching block
        // argument immediately before the output arguments.
        SmallVector<Value> inputs(newNode.getInputs());
        inputs.push_back(buffer);
        SmallVector<unsigned> inputTaps(newNode.getInputTapsAsInt());
        inputTaps.push_back(0);
        rewriter.setInsertionPoint(newNode);
        auto initializedNode = NodeOp::create(
            rewriter, newNode.getLoc(), inputs, newNode.getOutputs(),
            newNode.getParams(), inputTaps, newNode.getLevelAttr());
        auto bufferArg = newNode.getBody().insertArgument(
            originalNumInputs, newBufferArg.getType(), newBufferArg.getLoc());
        rewriter.inlineRegionBefore(newNode.getBody(),
                                    initializedNode.getBody(),
                                    initializedNode.getBody().end());
        rewriter.eraseOp(newNode);

        // Preserve elements that the node may read before defining them.
        // The copy belongs at the entry, before any producer writes; placing
        // it at a later read can overwrite a value computed earlier in the
        // same node.
        rewriter.setInsertionPointToStart(&initializedNode.getBody().front());
        memref::CopyOp::create(rewriter, loc, bufferArg, newBufferArg);
      }
    }
    return success(hasChanged);
  }
};
} // namespace

namespace {
struct MergeMultiProducer : public OpRewritePattern<ScheduleOp> {
  using OpRewritePattern<ScheduleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScheduleOp schedule,
                                PatternRewriter &rewriter) const override {
    DominanceInfo DT(schedule);
    bool hasChanged = false;

    SmallVector<Value> externalBuffers;
    for (auto arg : schedule.getBody().getArguments())
      externalBuffers.push_back(arg);

    for (auto buffer : externalBuffers) {
      SmallVector<NodeOp> producers(getProducers(buffer));
      if (producers.size() <= 1)
        continue;

      llvm::sort(producers, [&](NodeOp a, NodeOp b) {
        return DT.properlyDominates(a, b);
      });

      auto allNodes = SmallVector<NodeOp>(schedule.getOps<NodeOp>().begin(),
                                          schedule.getOps<NodeOp>().end());
      auto ptr = llvm::find(allNodes, producers.front());
      if (llvm::any_of(llvm::enumerate(producers), [&](auto node) {
            return node.value() != *std::next(ptr, node.index());
          }))
        continue;

      fuseNodeOps(producers, rewriter);
      hasChanged = true;
    }
    return success(hasChanged);
  }
};
} // namespace

namespace {
struct EliminateMultiProducer
    : public sar::impl::EliminateMultiProducerBase<EliminateMultiProducer> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<BufferMultiProducer>(context);
    patterns.add<MergeMultiProducer>(context);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> sar::createEliminateMultiProducerPass() {
  return std::make_unique<EliminateMultiProducer>();
}
