//===- EliminateMultiConsumer.cpp - eliminate multiple consumers ----------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

#include <limits>

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
static std::optional<uint64_t> getStreamTokenCount(BlockArgument argument,
                                                   bool writer) {
  Operation *endpoint = nullptr;
  for (OpOperand &use : argument.getUses()) {
    Operation *owner = use.getOwner();
    bool expected =
        writer ? isa<StreamWriteOp>(owner) && use.getOperandNumber() == 0
               : isa<StreamReadOp>(owner) && use.getOperandNumber() == 0;
    if (!expected || endpoint)
      return std::nullopt;
    endpoint = owner;
  }
  if (!endpoint)
    return std::nullopt;

  uint64_t count = 1;
  Operation *node = argument.getOwner()->getParentOp();
  for (Operation *parent = endpoint->getParentOp(); parent != node;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast_or_null<affine::AffineForOp>(parent);
    auto tripCount = loop ? affine::getConstantTripCount(loop) : std::nullopt;
    if (!tripCount || *tripCount > std::numeric_limits<uint64_t>::max() / count)
      return std::nullopt;
    count *= *tripCount;
  }
  return count;
}

struct InsertForkNode : public OpRewritePattern<NodeOp> {
  using OpRewritePattern<NodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NodeOp node,
                                PatternRewriter &rewriter) const override {
    auto loc = rewriter.getUnknownLoc();
    DominanceInfo domInfo;

    auto hasChanged = false;
    for (auto output : node.getOutputs()) {
      // DRAM buffers are not considered - the dependencies associated with
      // them are handled later by tokens.
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

      std::optional<uint64_t> streamTokens;
      if (isa<StreamType>(output.getType())) {
        unsigned outputIndex =
            llvm::find(node.getOutputs(), output) - node.getOutputs().begin();
        streamTokens = getStreamTokenCount(
            node.getBody().getArgument(node.getNumInputs() + outputIndex),
            /*writer=*/true);
        for (NodeOp consumer : consumers) {
          auto input = llvm::find(consumer.getInputs(), output);
          if (input == consumer.getInputs().end()) {
            streamTokens.reset();
            break;
          }
          unsigned inputIndex = input - consumer.getInputs().begin();
          auto count = getStreamTokenCount(
              consumer.getBody().getArgument(inputIndex), /*writer=*/false);
          if (!streamTokens || count != streamTokens) {
            streamTokens.reset();
            break;
          }
        }
        // Without a static, identical token count a fork cannot know when to
        // stop. Leave the violation for the scheduler to diagnose rather than
        // silently duplicating only the first element.
        if (!streamTokens)
          continue;
      }

      hasChanged = true;
      rewriter.setInsertionPointAfter(node);
      SmallVector<Value> buffers;
      SmallVector<Location> bufferLocs;

      // Insert a private channel for each consumer. Scalar streams cannot use
      // the memref-copy fork below: a fork consumes one token and writes that
      // same token to every output stream.
      for (auto consumer : consumers) {
        Value buffer;
        if (auto streamType = dyn_cast<StreamType>(output.getType()))
          buffer = StreamOp::create(rewriter, loc, streamType,
                                    streamType.getDepth());
        else
          buffer = BufferOp::create(rewriter, loc, output.getType());
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
      if (auto streamType = dyn_cast<StreamType>(output.getType())) {
        if (*streamTokens > 1) {
          auto loop =
              affine::AffineForOp::create(rewriter, loc, 0, *streamTokens);
          rewriter.setInsertionPointToStart(loop.getBody());
        }
        auto value = StreamReadOp::create(
            rewriter, loc, streamType.getElementType(), outputArg);
        for (auto bufferArg : bufferArgs)
          StreamWriteOp::create(rewriter, loc, bufferArg, value.getResult());
      } else {
        for (auto bufferArg : bufferArgs)
          memref::CopyOp::create(rewriter, loc, outputArg, bufferArg);
      }
    }
    return success(hasChanged);
  }
};
} // namespace

/// Forks top-level inputs with more readers than a dual-port memory can serve.
/// Constant tables remain shared because Vitis replicates their ROM readers;
/// the walk follows forwarded block arguments to identify them.
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
    // Legalization rejects every external argument with more than one
    // consumer: a single pointer cannot be read by multiple Vitis dataflow
    // processes. Fork a read-only on-chip argument even when it has only two
    // consumers; a `> 2` cutoff would leave the common two-phase case
    // serialized (RDA's `tau` input) and make a simpler algorithm slower than
    // the more heavily transformed WKA chain. DRAM arguments stay excluded
    // above because copying a full spilled plane merely to recover overlap
    // would exceed the placement budget; those designs must remain ordered.
    if (!onlyNodeReads || consumers.size() <= 1)
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
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    func.walk([&](ScheduleOp schedule) {
      forkWideArgFanOut(schedule.getBody().front());
    });
  }
};
} // namespace

std::unique_ptr<Pass> sar::createEliminateMultiConsumerPass() {
  return std::make_unique<EliminateMultiConsumer>();
}
