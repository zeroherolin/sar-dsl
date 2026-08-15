//===- LoopUnrolling.cpp - Fully unroll a loop band ----------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Unrolls a band outright when its trip count is small enough to be worth
// it, which is what lets pipelining schedule the whole body as one region.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

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


