//===- EliminateMultiConsumer.cpp - eliminate multi consumer --------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
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
        auto buffer = BufferOp::create(rewriter, loc, output.getType());
        output.replaceUsesWithIf(
            buffer, [&](OpOperand &use) { return use.getOwner() == consumer; });
        buffers.push_back(buffer);
        bufferLocs.push_back(loc);
      }

      // Create a new fork node.
      auto fork = NodeOp::create(rewriter, loc, output, buffers);
      auto block = rewriter.createBlock(&fork.getBody());
      auto outputArg = block->addArgument(output.getType(), output.getLoc());
      auto bufferArgs = block->addArguments(ValueRange(buffers), bufferLocs);

      // Create explicit copy from the original output to the buffers.
      rewriter.setInsertionPointToStart(block);
      for (auto bufferArg : bufferArgs)
        memref::CopyOp::create(rewriter, loc, outputArg, bufferArg);
    }
    return success(hasChanged);
  }
};
} // namespace

/// A top-level input enters its schedule without a producer node, so the
/// pattern above never forks it. Vitis binds a dual-port memory to two
/// reader processes on its own, but a third reader has no port left and
/// dataflow checking rejects the design (HLS 200-779). A wider fan-out
/// therefore gets an explicit fork copying the input into per-consumer
/// buffers, leaving the port itself with the fork as its only reader.
///
/// Constant tables are exempt: the emitter hoists them to file scope,
/// where they are ROMs the tool replicates per reader by itself
/// (measured: one instance per reading process, e.g. the omega-K
/// twiddle tables). Forking one here would turn a single ROM into
/// per-consumer ping-pong RAM copies plus the copy logic -- at a large
/// FFT size that alone overruns the device. The table may reach the
/// block through any depth of nested schedules, nodes and tasks, so
/// the walk follows block arguments up through whatever region op
/// forwarded them (schedules and nodes map operands to body arguments
/// one to one; tasks carry no operands, so the walk stops there).
static bool isConstTable(Value value) {
  while (auto arg = dyn_cast<BlockArgument>(value)) {
    auto parent = arg.getOwner()->getParentOp();
    if (isa<ScheduleOp, NodeOp>(parent))
      value = parent->getOperand(arg.getArgNumber());
    else
      return false;
  }
  return value.getDefiningOp<ConstBufferOp>() != nullptr;
}

static void forkWideArgFanOut(Block &block) {
  for (auto arg : block.getArguments()) {
    if (!isa<MemRefType>(arg.getType()) || isExtBuffer(arg) ||
        isConstTable(arg))
      continue;

    SmallVector<NodeOp> consumers;
    bool onlyNodeReads = true;
    for (auto &use : arg.getUses()) {
      auto node = dyn_cast<NodeOp>(use.getOwner());
      if (!node || node.getOperandKind(use) != OperandKind::INPUT) {
        onlyNodeReads = false;
        break;
      }
      if (!llvm::is_contained(consumers, node))
        consumers.push_back(node);
    }
    if (!onlyNodeReads || consumers.size() <= 2)
      continue;

    Operation *firstConsumer = consumers.front();
    for (NodeOp consumer : consumers)
      if (consumer->isBeforeInBlock(firstConsumer))
        firstConsumer = consumer;

    auto loc = arg.getLoc();
    OpBuilder builder(firstConsumer);
    SmallVector<Value> buffers;
    SmallVector<Location> bufferLocs;
    for (auto consumer : consumers) {
      auto buffer = BufferOp::create(builder, loc, arg.getType());
      arg.replaceUsesWithIf(
          buffer, [&](OpOperand &use) { return use.getOwner() == consumer; });
      buffers.push_back(buffer);
      bufferLocs.push_back(loc);
    }

    auto fork = NodeOp::create(builder, loc, arg, buffers);
    auto forkBlock = builder.createBlock(&fork.getBody());
    auto inputArg = forkBlock->addArgument(arg.getType(), loc);
    auto bufferArgs = forkBlock->addArguments(ValueRange(buffers), bufferLocs);
    builder.setInsertionPointToStart(forkBlock);
    for (auto bufferArg : bufferArgs)
      memref::CopyOp::create(builder, loc, inputArg, bufferArg);
  }
}

namespace {
struct EliminateMultiConsumer
    : public sar::impl::EliminateMultiConsumerBase<EliminateMultiConsumer> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<InsertForkNode>(context);
    (void)applyPatternsGreedily(func, std::move(patterns));

    func.walk([&](ScheduleOp schedule) {
      forkWideArgFanOut(schedule.getBody().front());
    });
  }
};
} // namespace

std::unique_ptr<Pass> sar::createEliminateMultiConsumerPass() {
  return std::make_unique<EliminateMultiConsumer>();
}
