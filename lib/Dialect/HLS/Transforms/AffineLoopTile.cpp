//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_AFFINELOOPTILE
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir


using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

/// Coefficient of dimension `pos` in `expr`, if `expr` is linear in it.
static std::optional<int64_t> dimCoefficient(AffineExpr expr, unsigned pos) {
  if (!expr.isFunctionOfDim(pos))
    return 0;
  if (isa<AffineDimExpr>(expr))
    return 1;
  auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binary)
    return std::nullopt;

  auto lhs = dimCoefficient(binary.getLHS(), pos);
  auto rhs = dimCoefficient(binary.getRHS(), pos);
  if (!lhs || !rhs)
    return std::nullopt;
  if (binary.getKind() == AffineExprKind::Add)
    return *lhs + *rhs;
  if (binary.getKind() == AffineExprKind::Mul) {
    if (auto factor = dyn_cast<AffineConstantExpr>(binary.getRHS()))
      return *lhs * factor.getValue();
    if (auto factor = dyn_cast<AffineConstantExpr>(binary.getLHS()))
      return factor.getValue() * *rhs;
  }
  // floordiv, ceildiv and mod are not linear in the dimension.
  return std::nullopt;
}

/// The loop in `band` whose induction variable `access` walks contiguously,
/// if any. External buffers only; on-chip ones cost nothing to stride.
static std::optional<unsigned> contiguousLoop(ArrayRef<AffineForOp> band,
                                              Value memref, AffineMap map,
                                              ValueRange operands) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || getMemoryKind(type) != MemoryKind::DRAM ||
      map.getNumResults() == 0)
    return std::nullopt;

  // The last index is the contiguous one; the loop that sweeps it is the one
  // whose induction variable enters it with a unit coefficient.
  auto last = map.getResult(map.getNumResults() - 1);
  for (unsigned level = 0; level < band.size(); ++level) {
    auto loop = band[level];
    auto operand = llvm::find(operands, loop.getInductionVar());
    if (operand == operands.end())
      continue;
    auto position = (unsigned)std::distance(operands.begin(), operand);
    if (position >= map.getNumDims())
      continue;
    auto stride = dimCoefficient(last, position);
    if (stride && std::abs(*stride) == 1)
      return level;
  }
  return std::nullopt;
}

/// How the external accesses of a band lay out against its loops.
struct StreamingLayout {
  /// The loop every external access sweeps contiguously, if they agree.
  std::optional<unsigned> agreedLevel;
  /// Set when two accesses want different loops innermost -- a transpose.
  bool conflict = false;
};

/// Classifies a band by the layout its external accesses ask for.
///
/// When they agree on a dimension, that dimension is the one that turns into
/// an AXI burst, and burst length is not recoverable later: splitting it
/// multiplies the number of DRAM transactions by the number of tiles. Any
/// reuse it happens to carry is a loop-invariant scalar, which pipelining
/// keeps in a register without needing a tile to hold it.
///
/// They disagree when the band transposes: one buffer is contiguous along
/// the row, the other along the column, and no choice of innermost loop
/// serves both. Tiling is exactly the answer there, trading full-length
/// bursts for a square block that gives both sides a run of the tile width.
static StreamingLayout getStreamingLayout(ArrayRef<AffineForOp> band) {
  StreamingLayout layout;
  if (band.empty())
    return layout;

  AffineForOp outermost = band.front();
  outermost.getBody()->walk([&](Operation *op) {
    std::optional<unsigned> level;
    if (auto read = dyn_cast<AffineReadOpInterface>(op))
      level = contiguousLoop(band, read.getMemRef(), read.getAffineMap(),
                             read.getMapOperands());
    else if (auto write = dyn_cast<AffineWriteOpInterface>(op))
      level = contiguousLoop(band, write.getMemRef(), write.getAffineMap(),
                             write.getMapOperands());
    else
      return WalkResult::advance();

    if (!level)
      return WalkResult::advance();
    if (layout.agreedLevel && *layout.agreedLevel != *level) {
      layout.conflict = true;
      return WalkResult::interrupt();
    }
    layout.agreedLevel = level;
    return WalkResult::advance();
  });
  if (layout.conflict)
    layout.agreedLevel.reset();
  return layout;
}

/// Tile edge for a band whose accesses disagree on layout.
///
/// A transposing band has to stage a block on chip, so the edge is what the
/// budget affords: a square of `dims` sides holding `elementBytes` each. The
/// bigger the block the longer each side's burst runs, which is the whole
/// point, so this takes the largest power of two that still fits.
static unsigned getTransposeTileSize(unsigned dims, unsigned elementBytes,
                                     unsigned budgetBytes) {
  unsigned edge = 1;
  while (true) {
    uint64_t bytes = elementBytes;
    for (unsigned i = 0; i < dims; ++i)
      bytes *= 2 * edge;
    if (bytes > budgetBytes)
      return edge;
    edge *= 2;
  }
}

/// Widest element any external access of `band` touches, in bytes.
static unsigned getExternalElementBytes(ArrayRef<AffineForOp> band) {
  unsigned bytes = 1;
  if (band.empty())
    return bytes;
  auto note = [&](Value memref) {
    if (auto type = dyn_cast<MemRefType>(memref.getType()))
      if (getMemoryKind(type) == MemoryKind::DRAM)
        bytes = std::max(bytes, type.getElementTypeBitWidth() / 8);
  };
  AffineForOp outermost = band.front();
  outermost.getBody()->walk([&](Operation *op) {
    if (auto read = dyn_cast<AffineReadOpInterface>(op))
      note(read.getMemRef());
    else if (auto write = dyn_cast<AffineWriteOpInterface>(op))
      note(write.getMemRef());
  });
  return bytes;
}

/// Whether tiling `loop` can capture any reuse.
///
/// Tiling exists to shorten the distance between two uses of the same datum,
/// so it pays exactly when some access in the body does not move with the
/// loop -- an invariant access is one that a tile would let a buffer hold on
/// to. When every access walks the dimension, each datum is touched once and
/// splitting the loop buys nothing.
static bool tilingCapturesReuse(AffineForOp loop) {
  auto iv = loop.getInductionVar();
  bool sawAccess = false;
  auto invariant = loop.getBody()->walk([&](Operation *op) {
    SmallVector<Value> operands;
    if (auto read = dyn_cast<AffineReadOpInterface>(op))
      operands = llvm::to_vector(read.getMapOperands());
    else if (auto write = dyn_cast<AffineWriteOpInterface>(op))
      operands = llvm::to_vector(write.getMapOperands());
    else
      return WalkResult::advance();
    sawAccess = true;
    if (!llvm::is_contained(operands, iv))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  // With no accesses at all there is nothing to reason about; leave the
  // decision to the caller's default.
  return !sawAccess || invariant.wasInterrupted();
}

/// Apply loop tiling to the input loop band and sink all intra-tile loops to
/// the innermost loop with the original loop order.
bool sar::applyLoopTiling(AffineLoopBand &band, FactorList tileList,
                               bool loopNormalize, bool annotatePointLoop) {
  assert(!band.empty() && "no loops provided");
  if (!isPerfectlyNested(band))
    return false;

  // If all tile sizes are one, we don't need to do anything but annotating all
  // loops as point loop.
  if (llvm::all_of(tileList, [](unsigned size) { return size == 1; })) {
    for (auto loop : band)
      if (annotatePointLoop)
        setPointAttr(loop);
    return true;
  }

  // Record the original band size and attributes to make use of later.
  auto originalBandSize = band.size();
  SmallVector<std::pair<bool, bool>, 6> flags;
  for (auto loop : band)
    flags.push_back({hasParallelAttr(loop), hasPointAttr(loop)});

  // Apply loop tiling.
  AffineLoopBand tiledBand;
  if (failed(tilePerfectlyNested(band, tileList, &tiledBand)))
    return false;

  // Get the tile loop band and point loop band.
  AffineLoopBand pointBand(std::next(tiledBand.begin(), originalBandSize),
                           tiledBand.end());
  tiledBand.resize(originalBandSize);

  // Annotate the required attributes.
  for (auto zip : llvm::zip(tiledBand, pointBand, flags)) {
    auto tileLoop = std::get<0>(zip);
    auto pointLoop = std::get<1>(zip);
    auto flag = std::get<2>(zip);

    // If a tile loop is parallel, the corresponding point loop should also be
    // a parallel loop.
    if (flag.first) {
      setParallelAttr(tileLoop);
      setParallelAttr(pointLoop);
    }

    // Re-annotate the point attribute to the tile loop if required.
    if (flag.second)
      setPointAttr(tileLoop);

    // Annotate the point attribute to the point loop.
    if (annotatePointLoop)
      setPointAttr(pointLoop);
  }

  // Always normalize point loop band.
  for (auto loop : pointBand)
    (void)normalizeAffineFor(loop);

  // Normalize tiled loop band if required.
  if (loopNormalize) {
    band.clear();
    for (auto loop : tiledBand)
      if (failed(promoteIfSingleIteration(loop))) {
        (void)normalizeAffineFor(loop);
        band.push_back(loop);
      }
  } else
    band = tiledBand;
  return true;
}

/// Reduces each tile size to the largest divisor of the corresponding trip
/// count (if the trip count is known).
void sar::adjustToDivisorsOfTripCounts(
    ArrayRef<AffineForOp> band, SmallVectorImpl<unsigned> *tileSizes) {
  assert(band.size() == tileSizes->size() && "invalid tile size count");
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    unsigned &tSizeAdjusted = (*tileSizes)[i];
    std::optional<uint64_t> mayConst = getConstantTripCount(band[i]);
    if (!mayConst)
      continue;

    // Adjust the tile size to largest factor of the trip count less than
    // tSize.
    uint64_t constTripCount = mayConst.value();
    // A tile that already spans the whole loop is a deliberate choice --
    // the dimension carries no reuse and is kept as one contiguous sweep --
    // so leave it be rather than halving it back into two tiles.
    if (tSizeAdjusted >= constTripCount) {
      tSizeAdjusted = constTripCount;
      continue;
    }
    if (constTripCount > 1 && tSizeAdjusted > constTripCount / 2)
      tSizeAdjusted = constTripCount / 2;
    while (constTripCount % tSizeAdjusted != 0)
      tSizeAdjusted--;
  }
}

namespace {
/// A pass to perform loop tiling on all suitable loop nests of a Function.
struct AffineLoopTile
    : public sar::impl::AffineLoopTileBase<AffineLoopTile> {
  AffineLoopTile() = default;
  explicit AffineLoopTile(unsigned loopTileSize, unsigned argTileBufferBytes) {
    tileSize = loopTileSize;
    tileBufferBytes = argTileBufferBytes;
  }

  void runOnOperation() override {
    // Bands of loops to tile.
    std::vector<SmallVector<AffineForOp, 6>> bands;
    getLoopBands(getOperation().front(), bands);
    // getTileableBands(getOperation(), &bands);

    // Tile each band.
    for (auto &band : bands) {
      auto layout = getStreamingLayout(band);

      // Put the dimension the accesses stream along innermost. A sweep of
      // an external buffer is only contiguous when the loop that walks it
      // runs innermost; anywhere else each step jumps a whole row, and the
      // AXI master moves one element per beat. Interchange is what fixes
      // that, and it is free -- unlike burst length, which no later pass
      // can recover.
      if (layout.agreedLevel && *layout.agreedLevel + 1 < band.size()) {
        SmallVector<unsigned> permutation(band.size());
        unsigned next = 0;
        for (unsigned level = 0; level < band.size(); ++level)
          if (level != *layout.agreedLevel)
            permutation[level] = next++;
        permutation[*layout.agreedLevel] = band.size() - 1;

        if (isValidLoopInterchangePermutation(band, permutation)) {
          permuteLoops(band, permutation);
          // The band is rebuilt in the new order, so re-read it before
          // deciding tile sizes.
          AffineLoopBand permuted;
          getLoopBandFromInnermost(band[*layout.agreedLevel], permuted);
          if (permuted.size() == band.size()) {
            band = permuted;
            layout = getStreamingLayout(band);
          }
        }
      }

      // A transposing band is tiled on every dimension, and its edge comes
      // from the block it has to stage rather than from the default: that
      // edge is the run length each side gets, so it is worth sizing.
      unsigned bandTileSize = tileSize;
      if (layout.conflict)
        bandTileSize = getTransposeTileSize(
            band.size(), getExternalElementBytes(band), tileBufferBytes);

      SmallVector<unsigned, 8> tileSizes;
      for (unsigned level = 0; level < band.size(); ++level) {
        // Keep a dimension whole when tiling it cannot pay: either it
        // carries no reuse to capture, or it is the contiguous sweep every
        // external access agrees on, where a shorter burst costs more than
        // any reuse a tile could hold.
        auto loop = band[level];
        auto trip = getConstantTripCount(loop);
        bool keepWhole =
            trip && !layout.conflict &&
            (!tilingCapturesReuse(loop) || layout.agreedLevel == level);
        tileSizes.push_back(keepWhole ? (unsigned)*trip : bandTileSize);
      }
      if (avoidMaxMinBounds)
        adjustToDivisorsOfTripCounts(band, &tileSizes);

      applyLoopTiling(band, tileSizes, /*loopNormalize=*/true);
    }
  }

  // If true, tile sizes are set to avoid max/min in bounds if possible.
  bool avoidMaxMinBounds = true;
};
} // namespace

/// Creates a pass to perform loop tiling on all suitable loop nests of a
/// Function.
std::unique_ptr<Pass>
sar::createAffineLoopTilePass(unsigned loopTileSize,
                                   unsigned tileBufferBytes) {
  return std::make_unique<AffineLoopTile>(loopTileSize, tileBufferBytes);
}
