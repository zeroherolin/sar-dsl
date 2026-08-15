//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_AFFINELOOPUNROLLJAM
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir


using namespace mlir;
using namespace mlir::affine;
using namespace sar;

/// Fully unroll all loops insides of a block.
bool sar::applyFullyLoopUnrolling(Block &block, unsigned maxIterNum) {
  for (unsigned i = 0; i < maxIterNum; ++i) {
    bool hasFullyUnrolled = true;
    block.walk([&](AffineForOp loop) {
      if (failed(loopUnrollFull(loop)))
        hasFullyUnrolled = false;
    });

    if (hasFullyUnrolled)
      break;

    if (i == 7)
      return false;
  }
  return true;
}

/// Apply unroll and jam to the loop band with the given overall unroll factor.
bool sar::applyLoopUnrollJam(AffineLoopBand &band,
                                  unsigned overallUnrollFactor) {
  assert(!band.empty() && "no loops provided");
  if (overallUnrollFactor == 1)
    return true;

  // Calculate the tiling size of each loop level.
  auto sizes = getDistributedFactors(overallUnrollFactor, band);
  return applyLoopUnrollJam(band, sizes);
}

/// Apply unroll and jam to the loop band with the given unroll factors.
bool sar::applyLoopUnrollJam(AffineLoopBand &band,
                                  FactorList unrollFactors) {
  assert(!band.empty() && "no loops provided");
  if (llvm::all_of(unrollFactors, [](unsigned factor) { return factor == 1; }))
    return true;

  // Apply loop tiling and then unroll all point loops.
  applyLoopTiling(band, unrollFactors, /*loopNormalize=*/false);
  auto result = applyFullyLoopUnrolling(*band.back().getBody());
  auto tiledBand = band;

  // Promote single iteration tiled loop band.
  band.clear();
  for (auto loop : tiledBand)
    if (failed(promoteIfSingleIteration(loop)))
      band.push_back(loop);
  return result;
}

namespace {
struct AffineLoopUnrollJam
    : public sar::impl::AffineLoopUnrollJamBase<AffineLoopUnrollJam> {
  AffineLoopUnrollJam() = default;
  AffineLoopUnrollJam(unsigned loopUnrollFactor, bool unrollPointLoopOnly) {
    unrollFactor = loopUnrollFactor;
    pointLoopOnly = unrollPointLoopOnly;
  }

  void runOnOperation() override {
    AffineLoopBands targetBands;
    getLoopBands(getOperation().front(), targetBands);
    // getTileableBands(getOperation(), &targetBands);

    for (auto &band : targetBands) {
      // For loop band that has effect on external buffers, we should directly
      // unroll them without considering whether it's point loop.
      if (hasEffectOnExternalBuffer(band.front())) {
        applyLoopUnrollJam(band, unrollFactor.getValue());
        continue;
      }

      if (pointLoopOnly) {
        AffineLoopBand tileBand;
        AffineLoopBand pointBand;
        if (!getTileAndPointLoopBand(band, tileBand, pointBand) ||
            pointBand.empty())
          continue;
        band = pointBand;
      }
      applyLoopUnrollJam(band, unrollFactor.getValue());
    }
  }
};
} // namespace

std::unique_ptr<Pass>
sar::createAffineLoopUnrollJamPass(unsigned loopUnrollFactor,
                                        bool unrollPointLoopOnly) {
  return std::make_unique<AffineLoopUnrollJam>(loopUnrollFactor,
                                               unrollPointLoopOnly);
}
