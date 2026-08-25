//===- LoopPipelining.cpp - attach pipeline pragmas and unroll inside -----===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"
#include "sar/Support/HLSHints.h"

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

/// Fully unroll small loops inside a block. A large loop remains compact and
/// receives a bounded pragma factor: cloning `trip-count * body-ops` without a
/// budget can make both the generated source and Vitis binding grow by orders
/// of magnitude.
static bool applyFullyLoopUnrolling(Block &block, unsigned maxUnrolledOps,
                                    unsigned maxUnrollFactor,
                                    unsigned maxIterNum = 10) {
  for (unsigned i = 0; i < maxIterNum; ++i) {
    // `settled` tracks whether the sweep left the block alone. Unrolling a
    // loop clones its body -- inner loops included -- into the parent, and
    // the walk skips the erased loop's region, so those clones are only
    // seen on the next sweep. Repeating until nothing changes is what
    // flattens a nest; stopping after one pass would leave inner loops
    // standing inside a body the caller is about to mark pipelined.
    bool settled = true;
    bool failedUnroll = false;
    block.walk<WalkOrder::PreOrder>([&](AffineForOp loop) {
      if (loop->hasAttr(kUnrollFactorAttr))
        return WalkResult::skip();

      uint64_t bodyOps = 0;
      loop.getBody()->walk([&](Operation *op) {
        if (!isa<AffineYieldOp>(op))
          ++bodyOps;
      });
      auto tripCount = getConstantTripCount(loop);
      bool materialize =
          tripCount && bodyOps <= maxUnrolledOps &&
          uint64_t(*tripCount) * std::max<uint64_t>(1, bodyOps) <=
              maxUnrolledOps;
      if (!materialize) {
        unsigned factor = 1;
        if (tripCount && maxUnrollFactor > 1) {
          uint64_t affordable = maxUnrolledOps / std::max<uint64_t>(1, bodyOps);
          factor = std::min<uint64_t>(
              {*tripCount, maxUnrollFactor, std::max<uint64_t>(1, affordable)});
          unsigned powerOfTwo = 1;
          while (powerOfTwo <= factor / 2)
            powerOfTwo *= 2;
          factor = powerOfTwo;
        }
        loop->setAttr(
            kUnrollFactorAttr,
            IntegerAttr::get(IntegerType::get(loop.getContext(), 64), factor));
        settled = false;
        return WalkResult::skip();
      }
      // Full unroll erases `loop` and clones its body into the parent, so
      // the walk must not descend into it: a pre-order walk may only erase
      // the op it is visiting when it also skips that op's region. The
      // clones land in the parent block and are reached by the next
      // iteration of the retry loop below.
      if (failed(loopUnrollFull(loop))) {
        failedUnroll = true;
        return WalkResult::advance();
      }
      settled = false;
      return WalkResult::skip();
    });

    if (failedUnroll)
      return false;
    if (settled)
      return true;
  }
  // The iteration budget ran out with loops still standing.
  return false;
}

static bool applyLoopPipeliningImpl(AffineLoopBand &band, unsigned pipelineLoc,
                                    unsigned targetII, unsigned maxUnrolledOps,
                                    unsigned maxUnrollFactor) {
  auto targetLoop = band[pipelineLoc];
  if (auto minimum = targetLoop->getAttrOfType<IntegerAttr>(kMinIIAttr))
    targetII = std::max<int64_t>(targetII, minimum.getInt());

  if (!targetLoop.getOps<func::CallOp>().empty())
    return false;

  // Vitis unrolls inner loops of a pipelined loop; reject a choice whose
  // resulting source and hardware expansion exceed the configured bounds.
  if (!applyFullyLoopUnrolling(*targetLoop.getBody(), maxUnrolledOps,
                               maxUnrollFactor))
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

/// Apply loop pipelining to the input loop, all inner loops are automatically
/// fully unrolled.
bool sar::applyLoopPipelining(AffineLoopBand &band, unsigned pipelineLoc,
                              unsigned targetII) {
  return applyLoopPipeliningImpl(band, pipelineLoc, targetII,
                                 /*maxUnrolledOps=*/4096,
                                 /*maxUnrollFactor=*/32);
}

namespace {
struct LoopPipelining : public sar::impl::LoopPipeliningBase<LoopPipelining> {
  LoopPipelining() = default;
  LoopPipelining(unsigned maxOps, unsigned maxFactor) {
    maxUnrolledOps = maxOps;
    maxUnrollFactor = maxFactor;
  }

  void runOnOperation() override {
    AffineLoopBands targetBands;
    getLoopBands(getOperation().front(), targetBands);

    for (auto &band : targetBands) {
      auto currentLoop = band.back();
      unsigned loopLevel = 0;
      while (true) {
        auto parentLoop = currentLoop->getParentOfType<AffineForOp>();

        if (!parentLoop || pipelineLevel == loopLevel) {
          applyLoopPipeliningImpl(band, band.size() - loopLevel - 1, targetII,
                                  maxUnrolledOps, maxUnrollFactor);
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

std::unique_ptr<Pass> sar::createLoopPipeliningPass(unsigned maxUnrolledOps,
                                                    unsigned maxUnrollFactor) {
  return std::make_unique<LoopPipelining>(maxUnrolledOps, maxUnrollFactor);
}
