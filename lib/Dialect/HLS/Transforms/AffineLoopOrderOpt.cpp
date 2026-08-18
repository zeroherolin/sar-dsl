//===- AffineLoopOrderOpt.cpp - affine loop order opt ---------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_AFFINELOOPORDEROPT
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

#define DEBUG_TYPE "hls"

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace sar::hls;

/// Optimize loop order. Loops associated with memory access dependencies are
/// moved to an as outer as possible location of the input loop band. If
/// "reverse" is true, as inner as possible.
bool sar::applyAffineLoopOrderOpt(AffineLoopBand &band,
                                  ArrayRef<unsigned> permMap, bool reverse) {
  LLVM_DEBUG(llvm::dbgs() << "Loop order opt ";);
  assert(!band.empty() && "no loops provided");

  if (!isPerfectlyNested(band))
    return false;

  auto &loopBlock = *band.back().getBody();
  auto bandDepth = band.size();

  if (!permMap.empty() && isValidLoopInterchangePermutation(band, permMap)) {
    auto newRoot = band[permuteLoops(band, permMap)];
    band.clear();
    getLoopBandFromOutermost(newRoot, band);
    band.resize(bandDepth);
    return true;
  }

  // Collect all load and store operations for each memory in the loop block,
  // and calculate the number of common surrouding loops for later uses.
  MemAccessesMap loadStoresMap;
  getMemAccessesMap(loopBlock, loadStoresMap);

  // A map of dependency distances indexed by the loop in the band.
  SmallVector<AffineForOp, 8> targetLoops;
  for (auto loop : band)
    if (!isLoopParallel(loop))
      targetLoops.push_back(loop);

  // Permute the target loops one by one.
  for (auto loop : targetLoops) {
    unsigned targetLoopLoc =
        std::find(band.begin(), band.end(), loop) - band.begin();

    if (!reverse)
      // Permute the target loop to an as outer as possible location.
      for (unsigned dstLoc = 0; dstLoc < targetLoopLoc; ++dstLoc) {
        SmallVector<unsigned, 4> permMap;

        // Construct permutation map.
        for (unsigned loc = 0; loc < bandDepth; ++loc) {
          if (loc < dstLoc)
            permMap.push_back(loc);
          else if (loc < targetLoopLoc)
            permMap.push_back(loc + 1);
          else if (loc == targetLoopLoc)
            permMap.push_back(dstLoc);
          else
            permMap.push_back(loc);
        }

        // Check the validation of the current permutation.
        if (isValidLoopInterchangePermutation(band, permMap)) {
          LLVM_DEBUG(llvm::dbgs() << "(";);
          LLVM_DEBUG(for (unsigned i = 0, e = permMap.size(); i < e; ++i) {
            llvm::dbgs() << permMap[i];
            if (i != e - 1)
              llvm::dbgs() << ",";
          });
          LLVM_DEBUG(llvm::dbgs() << ") ";);

          auto newRoot = band[permuteLoops(band, permMap)];
          band.clear();
          getLoopBandFromOutermost(newRoot, band);
          band.resize(bandDepth);
          break;
        }
      }
    else
      // Permute the target loop to an as inner as possible location.
      for (unsigned dstLoc = targetLoopLoc + 1; dstLoc < bandDepth; ++dstLoc) {
        SmallVector<unsigned, 4> permMap;

        // Construct permutation map.
        for (unsigned loc = 0; loc < bandDepth; ++loc) {
          if (loc < targetLoopLoc)
            permMap.push_back(loc);
          else if (loc == targetLoopLoc)
            permMap.push_back(dstLoc);
          else if (loc <= dstLoc)
            permMap.push_back(loc - 1);
          else
            permMap.push_back(loc);
        }

        // Check the validation of the current permutation.
        if (isValidLoopInterchangePermutation(band, permMap)) {
          auto newRoot = band[permuteLoops(band, permMap)];
          band.clear();
          getLoopBandFromOutermost(newRoot, band);
          band.resize(bandDepth);
          break;
        }
      }
  }

  LLVM_DEBUG(llvm::dbgs() << "\n";);
  return true;
}

namespace {
struct AffineLoopOrderOpt
    : public sar::impl::AffineLoopOrderOptBase<AffineLoopOrderOpt> {
  void runOnOperation() override {
    AffineLoopBands targetBands;
    getLoopBands(getOperation().front(), targetBands);

    // Apply loop order optimization to each loop band.
    for (auto &band : targetBands) {
      AffineLoopBand tileBand;
      AffineLoopBand pointBand;

      // If the current loop band can be split into a tile band and point band,
      // apply to these two partial bands separately. Otherwise, apply to the
      // whole loop band.
      if (getTileAndPointLoopBand(band, tileBand, pointBand)) {
        if (!tileBand.empty())
          applyAffineLoopOrderOpt(tileBand);
        if (!pointBand.empty())
          applyAffineLoopOrderOpt(pointBand);
      } else
        applyAffineLoopOrderOpt(band);
    }
  }
};
} // namespace

std::unique_ptr<Pass> sar::createAffineLoopOrderOptPass() {
  return std::make_unique<AffineLoopOrderOpt>();
}
