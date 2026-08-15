//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_LEGALIZEDATAFLOW
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir


using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

/// Sorts `nodes` into program order and reports whether they can be fused.
///
/// Fusion collapses the group into the position of its last member, so any
/// node sitting between the members has to move across the fused body. That
/// is only safe when there are none: otherwise a node that consumes an early
/// member's output would end up running before the value is written.
static bool sortAndCheckFusable(SmallVectorImpl<NodeOp> &nodes,
                                ScheduleOp schedule,
                                const DominanceInfo &domInfo) {
  llvm::sort(nodes, [&](NodeOp a, NodeOp b) {
    return domInfo.properlyDominates(a, b);
  });

  auto allNodes = llvm::to_vector(schedule.getOps<NodeOp>());
  auto first = llvm::find(allNodes, nodes.front());
  if (first == allNodes.end() ||
      std::distance(first, allNodes.end()) < (ptrdiff_t)nodes.size())
    return false;
  return std::equal(nodes.begin(), nodes.end(), first);
}

static void collectNodes(llvm::SmallDenseSet<NodeOp> const &allNodes,
                         llvm::SmallDenseSet<NodeOp> &visitedNodes,
                         SmallVector<NodeOp> &nodesToMerge, NodeOp node) {
  if (!visitedNodes.insert(node).second)
    return;
  nodesToMerge.push_back(node);
  for (auto input : node.getInputs())
    for (auto consumer : getConsumersExcept(input, node))
      if (allNodes.count(consumer))
        collectNodes(allNodes, visitedNodes, nodesToMerge, consumer);
}

namespace {
struct FuseMultiConsumer : public OpRewritePattern<ScheduleOp> {
  using OpRewritePattern<ScheduleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScheduleOp schedule,
                                PatternRewriter &rewriter) const override {
    // Collect nodes that are scheduled to the same level.
    llvm::SmallDenseMap<unsigned, llvm::SmallDenseSet<NodeOp>> levelToNodesMap;
    for (auto node : schedule.getOps<NodeOp>()) {
      if (auto level = node.getLevel())
        levelToNodesMap[level.value()].insert(node);
      else
        return failure();
    }

    // Merge nodes at the same level if they share the same input (to remove
    // multi-consumer violation).
    DominanceInfo domInfo;
    bool hasChanged = false;
    for (const auto &p : levelToNodesMap) {
      // llvm::outs() << p.first << "\n";
      llvm::SmallDenseSet<NodeOp> visitedNodes;
      SmallVector<SmallVector<NodeOp>> worklist;

      for (auto node : p.second) {
        if (visitedNodes.count(node))
          continue;
        SmallVector<NodeOp> nodesToMerge;
        collectNodes(p.second, visitedNodes, nodesToMerge, node);
        if (nodesToMerge.size() > 1)
          worklist.push_back(nodesToMerge);
      }

      for (auto nodesToMerge : worklist) {
        if (!sortAndCheckFusable(nodesToMerge, schedule, domInfo))
          continue;
        auto newNode = fuseNodeOps(nodesToMerge, rewriter);
        newNode.setLevelAttr(rewriter.getI32IntegerAttr(p.first));
        hasChanged = true;
      }
    }
    // schedule.setIsLegalAttr(rewriter.getUnitAttr());
    return success(hasChanged);
  }
};
} // namespace

static void collectBypassNodes(
    llvm::SmallDenseMap<unsigned, llvm::SmallDenseSet<NodeOp>> const &map,
    llvm::SmallDenseSet<unsigned> &mergedLevels,
    SmallVector<NodeOp> &nodesToMerge, unsigned targetLevel) {
  // Find the maximum difference between the nodes in the current level and
  // all consumer nodes.
  unsigned maxDiff = 1;
  for (auto node : map.lookup(targetLevel)) {
    for (auto output : node.getOutputs()) {
      if (isa<BlockArgument>(output) &&
          node.getScheduleOp().isDependenceFree())
        continue;

      // DRAM buffer is not considered - the dependencies associated with them
      // are handled later by tokens.
      if (isExtBuffer(output))
        continue;

      SmallVector<std::pair<unsigned, NodeOp>, 4> bypassNodes;
      for (auto consumer : getDependentConsumers(output, node)) {
        auto diff = node.getLevel().value() - consumer.getLevel().value();
        if (diff > 1)
          bypassNodes.push_back({diff, consumer});
      }
      if (bypassNodes.empty())
        continue;

      // Sort all consumers in a descending order of level difference.
      llvm::sort(bypassNodes, [](auto a, auto b) { return a.first > b.first; });
      maxDiff = std::max(maxDiff, bypassNodes.front().first);
    }
  }
  if (maxDiff == 1)
    return;

  for (auto level = targetLevel - 1; level >= targetLevel - maxDiff; --level) {
    if (!mergedLevels.insert(level).second)
      continue;
    // llvm::outs() << "---------- " << level << "\n";
    for (auto node : map.lookup(level))
      nodesToMerge.push_back(node);
    collectBypassNodes(map, mergedLevels, nodesToMerge, level);
  }
}

namespace {
struct FuseBypassPath : public OpRewritePattern<ScheduleOp> {
  using OpRewritePattern<ScheduleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScheduleOp schedule,
                                PatternRewriter &rewriter) const override {
    // Collect nodes that are scheduled to the same level.
    unsigned maxLevel = 0;
    llvm::SmallDenseMap<unsigned, llvm::SmallDenseSet<NodeOp>> levelToNodesMap;
    for (auto node : schedule.getOps<NodeOp>()) {
      if (auto level = node.getLevel()) {
        levelToNodesMap[level.value()].insert(node);
        maxLevel = std::max(maxLevel, level.value());
      } else
        return failure();
    }

    // Traverse all dataflow node levels.
    llvm::SmallDenseSet<unsigned> mergedLevels;
    SmallVector<SmallVector<NodeOp>> worklist;

    for (auto level = maxLevel; level > 0; --level) {
      if (mergedLevels.count(level))
        continue;
      // llvm::outs() << "\n========== " << level << "\n";
      SmallVector<NodeOp> nodesToMerge;
      collectBypassNodes(levelToNodesMap, mergedLevels, nodesToMerge, level);
      if (nodesToMerge.size() > 1)
        worklist.push_back(nodesToMerge);
    }

    bool hasChanged = false;
    DominanceInfo domInfo;
    for (auto nodesToMerge : worklist) {
      if (!sortAndCheckFusable(nodesToMerge, schedule, domInfo))
        continue;
      auto newNode = fuseNodeOps(nodesToMerge, rewriter);
      newNode.setLevelAttr(
          rewriter.getI32IntegerAttr(nodesToMerge.front().getLevel().value()));
      hasChanged = true;
    }
    return success(hasChanged);
  }
};
} // namespace

// namespace {
// struct AllocateInternalBuffer : public OpRewritePattern<BufferOp> {
//   using OpRewritePattern<BufferOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(BufferOp buffer,
//                                 PatternRewriter &rewriter) const override {
//     if (isExtBuffer(buffer) && llvm::hasSingleElement(buffer->getUsers()))
//       if (auto node = dyn_cast<NodeOp>(*buffer->user_begin())) {
//         auto bufferType = buffer.getType();
//         auto newType = MemRefType::get(
//             bufferType.getShape(), bufferType.getElementType(), AffineMap(),
//             MemoryKindAttr::get(buffer.getContext(), MemoryKind::BRAM_T2P));
//         buffer.getMemref().setType(newType);
//         node.updateSignatureRecursively();
//         return success();
//       }
//     return failure();
//   }
// };
// } // namespace

/// Whether any external buffer in `schedule` is written by more than one node.
///
/// That is the signature of a DRAM buffer being reused across lifetimes that
/// do not overlap -- a sequence of whole-raster passes handing one plane back
/// and forth. Reuse and concurrency exclude each other: tokens order the
/// read-after-write edges between producers and consumers, but nothing orders
/// the write-after-read edge that reuse introduces, so a dataflow region here
/// would let a later pass overwrite a plane an earlier one is still reading.
/// Such a schedule stays sequential; the loops nested inside it, whose
/// buffers are on chip and unshared, are still free to become dataflow.
static bool hasSharedExternalBuffer(ScheduleOp schedule) {
  auto isShared = [](Value buffer) {
    return isExtBuffer(buffer) && getProducers(buffer).size() > 1;
  };
  return llvm::any_of(schedule.getBody().getArguments(),
                      [&](Value arg) { return isShared(arg); }) ||
         llvm::any_of(schedule.getOps<BufferOp>(),
                      [&](BufferOp buffer) { return isShared(buffer); });
}

namespace {
struct LegalizeDataflow : public sar::impl::LegalizeDataflowBase<LegalizeDataflow> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    // Fuse multi consumer and bypass path dataflow nodes.
    mlir::RewritePatternSet patterns(context);
    patterns.add<FuseMultiConsumer>(context);
    patterns.add<FuseBypassPath>(context);
    auto frozenPatterns = FrozenRewritePatternSet(std::move(patterns));

    func.walk([&](ScheduleOp schedule) {
      (void)applyOpPatternsGreedily(
          ArrayRef<Operation *>{schedule.getOperation()},
          frozenPatterns);

      if (llvm::all_of(schedule.getOps<NodeOp>(),
                       [](NodeOp node) { return node.getLevel(); }) &&
          !hasSharedExternalBuffer(schedule))
        schedule.setIsLegalAttr(UnitAttr::get(context));
    });

    // // Reallocate internal buffers.
    // patterns.clear();
    // patterns.add<AllocateInternalBuffer>(context);
    // (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> sar::createLegalizeDataflowPass() {
  return std::make_unique<LegalizeDataflow>();
}
