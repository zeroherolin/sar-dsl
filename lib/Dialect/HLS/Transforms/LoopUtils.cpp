//===- LoopUtils.cpp - affine loop band helpers ---------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"
#include "sar/Support/HLSHints.h"

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

/// Loops contained immediately by `op`, not counting deeper nesting.
static unsigned getChildLoopNum(Operation *op) {
  unsigned childNum = 0;
  for (auto &region : op->getRegions())
    for (auto &block : region)
      for (auto &op : block)
        if (isa<AffineForOp>(op))
          ++childNum;

  return childNum;
}

/// Given a tiled loop band, return true and get the tile (tile-space) loop band
/// and the point (intra-tile) loop band. If failed, return false.
bool sar::getTileAndPointLoopBand(const AffineLoopBand &band,
                                  AffineLoopBand &tileBand,
                                  AffineLoopBand &pointBand) {
  tileBand.clear();
  pointBand.clear();
  bool isPointLoop = false;

  for (auto loop : band) {
    if (!isPointLoop && !hasPointAttr(loop))
      tileBand.push_back(loop);

    else if (isPointLoop && hasPointAttr(loop))
      pointBand.push_back(loop);

    else if (!isPointLoop && hasPointAttr(loop)) {
      isPointLoop = true;
      pointBand.push_back(loop);

    } else {
      tileBand.clear();
      pointBand.clear();
      return false;
    }
  }
  return true;
}

/// Get the whole loop band given the outermost loop and return it in "band".
/// Meanwhile, the return value is the innermost loop of this loop band.
AffineForOp sar::getLoopBandFromOutermost(AffineForOp forOp,
                                          AffineLoopBand &band) {
  band.clear();
  auto currentLoop = forOp;
  while (true) {
    band.push_back(currentLoop);

    if (getChildLoopNum(currentLoop) == 1)
      currentLoop = *currentLoop.getOps<AffineForOp>().begin();
    else
      break;
  }
  return band.back();
}
AffineForOp sar::getLoopBandFromInnermost(AffineForOp forOp,
                                          AffineLoopBand &band) {
  band.clear();
  AffineLoopBand reverseBand;

  auto currentLoop = forOp;
  while (true) {
    reverseBand.push_back(currentLoop);

    auto parentLoop = currentLoop->getParentOfType<AffineForOp>();
    if (!parentLoop)
      break;

    if (getChildLoopNum(parentLoop) == 1)
      currentLoop = parentLoop;
    else
      break;
  }

  band.append(reverseBand.rbegin(), reverseBand.rend());
  return band.front();
}

/// Collect all loop bands in the "block" and return them in "bands". If
/// "allowHavingChilds" is true, loop bands containing more than 1 other loop
/// bands are also collected. Otherwise, only loop bands that contain no child
/// loops are collected.
void sar::getLoopBands(Block &block, AffineLoopBands &bands,
                       bool allowHavingChilds) {
  bands.clear();
  block.walk([&](AffineForOp loop) {
    // A loop carrying an unroll directive is a compact stand-in for
    // parallel lanes: it belongs to the body of the loop above it, not to
    // the band, so neither it nor its presence as a child counts here.
    if (loop->hasAttr(kUnrollFactorAttr))
      return;
    unsigned childNum = 0;
    for (auto &region : loop->getRegions())
      for (auto &childBlock : region)
        for (auto &op : childBlock)
          if (auto child = dyn_cast<AffineForOp>(op))
            if (!child->hasAttr(kUnrollFactorAttr))
              ++childNum;

    if (childNum == 0 || (childNum > 1 && allowHavingChilds)) {
      AffineLoopBand band;
      getLoopBandFromInnermost(loop, band);
      bands.push_back(band);
    }
  });
}
