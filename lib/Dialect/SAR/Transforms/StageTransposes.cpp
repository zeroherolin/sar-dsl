//===- StageTransposes.cpp - Stage transposing loop nests on chip ---------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// A transposing nest strides one of its two buffers by a whole row.
// Staging a square block in between makes both sides contiguous: the
// block is filled row-wise and drained column-wise, and it is the only
// thing left striding.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "sar/Dialect/SAR/Transforms/Passes.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SARSTAGETRANSPOSES
#include "sar/Dialect/SAR/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;

namespace {

/// A perfectly nested loop band that writes one buffer while reading another
/// along a different axis, with whatever computation in between.
struct TransposeNest {
  SmallVector<affine::AffineForOp> band;
  /// The reads whose buffers are walked along an axis the write does not.
  /// Each gets its own block: staging only some of them would leave the
  /// rest striding just the same.
  SmallVector<affine::AffineLoadOp> transposingLoads;
  /// Band level the transposing reads sweep. Staging swaps this level with
  /// `destLevel` for the drain half, which is what makes both contiguous.
  unsigned sourceLevel;
  /// Band level the write sweeps.
  unsigned destLevel;
};

/// The band level `map`'s last result varies fastest along, if that is a
/// bare induction variable of the band.
///
/// The writes are held to this stricter form. Staging reorders iterations,
/// which is only sound when each writes its own element, and a bare
/// induction variable is what makes the sweep a permutation of the axis --
/// an expression like `d floordiv 2` reads one level but sends two
/// iterations to the same address.
static std::optional<unsigned> fastestLevel(AffineMap map, ValueRange operands,
                                            ValueRange ivs) {
  if (map.getNumResults() == 0)
    return std::nullopt;
  auto dim = dyn_cast<AffineDimExpr>(map.getResult(map.getNumResults() - 1));
  if (!dim || dim.getPosition() >= operands.size())
    return std::nullopt;
  auto iv = llvm::find(ivs, operands[dim.getPosition()]);
  if (iv == ivs.end())
    return std::nullopt;
  return (unsigned)std::distance(ivs.begin(), iv);
}

/// The band level a *read*'s last result sweeps, allowing the index to be
/// any expression of a single induction variable.
///
/// A shift folded into the map -- what an `fftshift` fused into a corner
/// turn leaves behind -- makes the last result an expression such as
/// `(d0 + 64) mod n`. It still sweeps exactly one level, and staging only
/// needs to know which: both halves of the rewrite re-express the map
/// through the same substitution, so a block element and the address it
/// came from stay paired whatever shape the expression has. Reads need no
/// permutation guarantee -- fetching a value twice is harmless -- which is
/// why this is the read-side rule only.
static std::optional<unsigned> sweptLevel(AffineMap map, ValueRange operands,
                                          ValueRange ivs) {
  if (map.getNumResults() == 0)
    return std::nullopt;

  // Exactly one dimension makes the sweep unambiguous; none is a broadcast,
  // and several would mean the level cannot be swapped on its own.
  llvm::SmallDenseSet<unsigned> positions;
  map.getResult(map.getNumResults() - 1).walk([&](AffineExpr expr) {
    if (auto dim = dyn_cast<AffineDimExpr>(expr))
      positions.insert(dim.getPosition());
  });
  if (positions.size() != 1)
    return std::nullopt;

  unsigned position = *positions.begin();
  if (position >= operands.size())
    return std::nullopt;
  auto iv = llvm::find(ivs, operands[position]);
  if (iv == ivs.end())
    return std::nullopt;
  return (unsigned)std::distance(ivs.begin(), iv);
}

/// Collects the perfectly nested band rooted at `outer`, or nothing if the
/// nest is not perfect or a loop is not a plain zero-based counter.
static SmallVector<affine::AffineForOp> collectBand(affine::AffineForOp outer) {
  SmallVector<affine::AffineForOp> band;
  affine::AffineForOp loop = outer;
  while (true) {
    if (!loop.hasConstantLowerBound() || !loop.hasConstantUpperBound() ||
        loop.getConstantLowerBound() != 0 || loop.getStep() != 1)
      return {};
    band.push_back(loop);
    auto &ops = loop.getBody()->getOperations();
    if (!llvm::hasSingleElement(loop.getBody()->without_terminator()))
      break;
    auto next = dyn_cast<affine::AffineForOp>(&ops.front());
    if (!next)
      break;
    loop = next;
  }
  return band;
}

/// Recognizes a band that writes one buffer while reading another along an
/// axis the write does not walk. Any computation may sit in between: a
/// corner turn is usually fused with the window or phase multiply after it.
static std::optional<TransposeNest> matchTranspose(affine::AffineForOp outer) {
  auto band = collectBand(outer);
  if (band.size() < 2)
    return std::nullopt;

  SmallVector<Value> ivs;
  for (auto loop : band)
    ivs.push_back(loop.getInductionVar());

  // Every access has to be expressible over the tile and point loops the
  // rewrite introduces. A body may write more than one plane -- splitting
  // complex data into real and imaginary halves is the common case -- so
  // long as the writes agree on which level sweeps them.
  SmallVector<affine::AffineStoreOp> stores;
  SmallVector<affine::AffineLoadOp> loads;
  bool retargetable = true;
  band.back().getBody()->walk([&](Operation *op) {
    if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      loads.push_back(load);
    } else if (auto write = dyn_cast<affine::AffineStoreOp>(op)) {
      stores.push_back(write);
    } else if (op->getNumRegions() != 0) {
      // A nested region would carry induction variables -- and accesses --
      // the rewrite does not know how to re-express. This covers non-affine
      // control flow too: the replay clones such an op whole, so an access
      // inside it would keep naming the erased band.
      retargetable = false;
    } else if (!isMemoryEffectFree(op) && !isa<affine::AffineYieldOp>(op)) {
      // Reordering an unknown side effect across tile fill/drain changes the
      // program even when none of its operands are induction variables.
      retargetable = false;
    }
  });
  if (!retargetable || stores.empty())
    return std::nullopt;

  // Every use of a band induction variable must be an index the rewrite
  // re-expresses: a load or store *map* operand. An IV flowing anywhere
  // else -- into arithmetic, or a store's value -- would survive the
  // rewrite as a reference to the erased band.
  for (Value iv : ivs)
    for (OpOperand &use : iv.getUses()) {
      Operation *owner = use.getOwner();
      bool isLoadIndex =
          isa<affine::AffineLoadOp>(owner) && use.getOperandNumber() >= 1;
      bool isStoreIndex =
          isa<affine::AffineStoreOp>(owner) && use.getOperandNumber() >= 2;
      if (!isLoadIndex && !isStoreIndex)
        return std::nullopt;
    }

  auto accessesOnlyIVs = [&](ValueRange operands) {
    return llvm::all_of(operands,
                        [&](Value v) { return llvm::is_contained(ivs, v); });
  };
  auto staged = [](MemRefType type) {
    // A block only helps a buffer whose last axis is not the one being
    // swept; a vector has no other axis to stage against. The element has
    // to have a measurable width for the block to be budgeted.
    return type.getRank() >= 2 && type.hasStaticShape() &&
           type.getElementType().isIntOrFloat();
  };

  std::optional<unsigned> destLevel;
  for (auto store : stores) {
    if (!accessesOnlyIVs(store.getMapOperands()) ||
        !staged(store.getMemRefType()))
      return std::nullopt;
    // Reordering iterations is only sound when each writes its own element,
    // so a write has to name every level of the band. A write that drops a
    // level revisits the same address, and the block would decide the
    // winner by a different order than the original nest.
    if (llvm::any_of(ivs, [&](Value iv) {
          return !llvm::is_contained(store.getMapOperands(), iv);
        }))
      return std::nullopt;
    auto level =
        fastestLevel(store.getAffineMap(), store.getMapOperands(), ivs);
    if (!level || (destLevel && *destLevel != *level))
      return std::nullopt;
    destLevel = level;
  }

  SmallVector<affine::AffineLoadOp> transposing;
  std::optional<unsigned> sourceLevel;
  for (auto load : loads) {
    if (!accessesOnlyIVs(load.getMapOperands()))
      return std::nullopt;
    if (!staged(load.getMemRefType()))
      continue;
    auto level = sweptLevel(load.getAffineMap(), load.getMapOperands(), ivs);
    if (!level || *level == *destLevel)
      continue;
    // One read nest fills every block, so the reads have to agree on which
    // level sweeps them.
    if (sourceLevel && *sourceLevel != *level)
      return std::nullopt;
    sourceLevel = level;
    transposing.push_back(load);
  }
  if (transposing.empty())
    return std::nullopt;
  if (llvm::any_of(loads, [&](affine::AffineLoadOp load) {
        return llvm::any_of(stores, [&](affine::AffineStoreOp store) {
          return load.getMemRef() == store.getMemRef();
        });
      }))
    return std::nullopt;

  return TransposeNest{band, transposing, *sourceLevel, *destLevel};
}

/// Largest power-of-two tile edge that divides both extents and whose square
/// block, one per staged read, fits `budgetBytes`.
static int64_t getTileEdge(int64_t sourceTrip, int64_t destTrip,
                           uint64_t blockBytes, unsigned budgetBytes) {
  int64_t edge = 1;
  while (true) {
    int64_t next = edge * 2;
    if (sourceTrip % next || destTrip % next ||
        (uint64_t)next * next * blockBytes > budgetBytes)
      return edge;
    edge = next;
  }
}

/// Rewrites `nest` to stage a block on chip. Returns failure when no edge is
/// worth using, leaving the nest untouched.
static LogicalResult stageTranspose(TransposeNest nest, unsigned budgetBytes) {
  unsigned depth = nest.band.size();
  int64_t sourceTrip = nest.band[nest.sourceLevel].getConstantUpperBound();
  int64_t destTrip = nest.band[nest.destLevel].getConstantUpperBound();

  uint64_t blockBytes = 0;
  for (auto load : nest.transposingLoads)
    blockBytes +=
        std::max(1u, load.getMemRefType().getElementTypeBitWidth() / 8);
  int64_t edge = getTileEdge(sourceTrip, destTrip, blockBytes, budgetBytes);
  // An edge of one is a plain element-at-a-time copy: the block would carry
  // a single value and nothing becomes contiguous.
  if (edge < 2)
    return failure();

  OpBuilder builder(nest.band.front());
  Location loc = nest.band.front().getLoc();
  auto *ctx = builder.getContext();

  DenseMap<Operation *, Value> blocks;
  for (auto load : nest.transposingLoads)
    blocks[load] = memref::AllocOp::create(
        builder, loc,
        MemRefType::get({edge, edge}, load.getMemRefType().getElementType()));

  // Rebuild the band, splitting the two levels the staging swaps. Every
  // other level keeps its extent and its place.
  SmallVector<Value> outerIVs(depth);
  for (unsigned level = 0; level < depth; ++level) {
    int64_t trip = nest.band[level].getConstantUpperBound();
    if (level == nest.sourceLevel || level == nest.destLevel)
      trip /= edge;
    auto loop = affine::AffineForOp::create(builder, loc, 0, trip);
    outerIVs[level] = loop.getInductionVar();
    builder.setInsertionPointToStart(loop.getBody());
  }

  // Dimensions of the rewritten maps: the rebuilt band, then the two point
  // loops (source first, then destination).
  unsigned pointBase = depth;
  auto blockMap = AffineMap::get(
      depth + 2, 0,
      {getAffineDimExpr(pointBase, ctx), getAffineDimExpr(pointBase + 1, ctx)},
      ctx);

  // Re-expresses an access of the original band: a split level walks
  // tile * edge + point, every other level is unchanged.
  SmallVector<Value> ivs;
  for (auto loop : nest.band)
    ivs.push_back(loop.getInductionVar());
  auto retarget = [&](AffineMap map, ValueRange mapOperands) {
    SmallVector<AffineExpr> replacements(map.getNumDims(),
                                         getAffineConstantExpr(0, ctx));
    for (auto [position, operand] : llvm::enumerate(mapOperands)) {
      auto iv = llvm::find(ivs, operand);
      unsigned level = (unsigned)std::distance(ivs.begin(), iv);
      AffineExpr expr = getAffineDimExpr(level, ctx);
      if (level == nest.sourceLevel)
        expr = expr * edge + getAffineDimExpr(pointBase, ctx);
      else if (level == nest.destLevel)
        expr = expr * edge + getAffineDimExpr(pointBase + 1, ctx);
      replacements[position] = expr;
    }
    return map.replaceDimsAndSymbols(replacements, {}, depth + 2, 0);
  };

  // Emits one half with `fastPoint` innermost, so the buffer that point
  // indexes is swept contiguously.
  auto emitHalf = [&](bool sourceFast, auto emitBody) {
    OpBuilder::InsertionGuard guard(builder);
    auto slow = affine::AffineForOp::create(builder, loc, 0, edge);
    builder.setInsertionPointToStart(slow.getBody());
    auto fast = affine::AffineForOp::create(builder, loc, 0, edge);
    builder.setInsertionPointToStart(fast.getBody());

    SmallVector<Value> operands(outerIVs);
    operands.push_back(sourceFast ? fast.getInductionVar()
                                  : slow.getInductionVar());
    operands.push_back(sourceFast ? slow.getInductionVar()
                                  : fast.getInductionVar());
    emitBody(operands);
  };

  // Fill the blocks with the sources contiguous.
  emitHalf(/*sourceFast=*/true, [&](ValueRange operands) {
    for (auto load : nest.transposingLoads) {
      Value v = affine::AffineLoadOp::create(
          builder, loc, load.getMemRef(),
          retarget(load.getAffineMap(), load.getMapOperands()), operands);
      affine::AffineStoreOp::create(builder, loc, v, blocks[load], blockMap,
                                    operands);
    }
  });

  // Replay the body with the destination contiguous, reading the blocks in
  // place of the strided planes.
  emitHalf(/*sourceFast=*/false, [&](ValueRange operands) {
    IRMapping mapping;
    for (auto &op : nest.band.back().getBody()->without_terminator()) {
      if (auto load = dyn_cast<affine::AffineLoadOp>(&op)) {
        auto staged = blocks.find(load);
        bool isStaged = staged != blocks.end();
        Value v = affine::AffineLoadOp::create(
            builder, loc, isStaged ? staged->second : load.getMemRef(),
            isStaged ? blockMap
                     : retarget(load.getAffineMap(), load.getMapOperands()),
            operands);
        mapping.map(load.getResult(), v);
      } else if (auto write = dyn_cast<affine::AffineStoreOp>(&op)) {
        affine::AffineStoreOp::create(
            builder, loc, mapping.lookupOrDefault(write.getValueToStore()),
            write.getMemRef(),
            retarget(write.getAffineMap(), write.getMapOperands()), operands);
      } else {
        builder.clone(op, mapping);
      }
    }
  });

  nest.band.front().erase();
  return success();
}

struct SARStageTransposes
    : public sar::impl::SARStageTransposesBase<SARStageTransposes> {
  using SARStageTransposesBase::SARStageTransposesBase;

  void runOnOperation() override {
    // Collect in pre-order so that an outer match always precedes any inner
    // match it contains. (The default walk is post-order, which would visit
    // inner loops first and break the containment check below.)
    SmallVector<TransposeNest> nests;
    getOperation().walk<WalkOrder::PreOrder>([&](affine::AffineForOp loop) {
      if (auto nest = matchTranspose(loop))
        nests.push_back(*nest);
    });

    // When an outer loop's induction variable does not appear in any access
    // map, the inner band also matches on its own. Staging rewrites a nest
    // by erasing its entire band, so attempting to stage the inner nest
    // afterward would be a use-after-free. Pre-order guarantees a container
    // precedes what it holds, so tracking which outer loops were kept
    // and skip any nest whose outer loop is a descendant of one of them.
    DenseSet<Operation *> outermost;
    SmallVector<TransposeNest> toStage;
    for (auto &nest : nests) {
      bool contained = false;
      for (Operation *p = nest.band.front()->getParentOp(); p && !contained;
           p = p->getParentOp())
        contained = outermost.contains(p);
      if (contained)
        continue;
      outermost.insert(nest.band.front().getOperation());
      toStage.push_back(nest);
    }

    for (auto &nest : toStage)
      (void)stageTranspose(nest, blockBytes);
  }
};

} // namespace
