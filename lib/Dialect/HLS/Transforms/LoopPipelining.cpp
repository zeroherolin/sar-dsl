//===- LoopPipelining.cpp - loop pipelining -------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_LOOPPIPELINING
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

/// Apply loop pipelining to the input loop, all inner loops are automatically
/// fully unrolled.
bool sar::applyLoopPipelining(AffineLoopBand &band, unsigned pipelineLoc,
                              unsigned targetII) {
  auto targetLoop = band[pipelineLoc];

  if (!targetLoop.getOps<func::CallOp>().empty())
    return false;

  // All inner loops of the pipelined loop are automatically unrolled.
  if (!applyFullyLoopUnrolling(*targetLoop.getBody()))
    return false;

  // Erase all loops in loop band that are inside of the pipelined loop.
  band.resize(pipelineLoc + 1);

  setLoopDirective(targetLoop, true, targetII);

  // Outer loops that perfectly nest the pipelined loop also get a directive:
  // its presence is what makes the emitter give them `#pragma HLS dependence
  // false`. Vitis flattens such nests into the pipelined loop on its own.
  auto currentLoop = targetLoop;
  while (true) {
    if (auto outerLoop = currentLoop->getParentOfType<AffineForOp>()) {
      // Only if the current loop is the only child loop of the outer loop
      // does the nest stay perfect.
      bool perfectNest = true;
      for (auto &op : outerLoop)
        if (&op != currentLoop && !isa<AffineApplyOp, AffineYieldOp>(op)) {
          perfectNest = false;
          break;
        }

      if (perfectNest) {
        currentLoop = outerLoop;
        setLoopDirective(outerLoop, false, 1);
        continue;
      }
    }
    break;
  }

  return true;
}

namespace {
struct LoopPipelining : public sar::impl::LoopPipeliningBase<LoopPipelining> {
  void runOnOperation() override {
    AffineLoopBands targetBands;
    getLoopBands(getOperation().front(), targetBands);

    // Apply loop pipelining to corresponding level of each innermost loop.
    for (auto &band : targetBands) {
      auto currentLoop = band.back();
      unsigned loopLevel = 0;
      while (true) {
        auto parentLoop = currentLoop->getParentOfType<AffineForOp>();

        // If meet the outermost loop, pipeline the current loop.
        if (!parentLoop || pipelineLevel == loopLevel) {
          applyLoopPipelining(band, band.size() - loopLevel - 1, targetII);
          break;
        }

        // Move to the next loop level.
        currentLoop = parentLoop;
        ++loopLevel;
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> sar::createLoopPipeliningPass() {
  return std::make_unique<LoopPipelining>();
}
