//===- RaiseAffineToCopy.cpp - recognize copy loops as memref copies ------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_RAISEAFFINETOCOPY
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace sar::hls;

namespace {
struct RaiseAffineToCopy
    : public sar::impl::RaiseAffineToCopyBase<RaiseAffineToCopy> {
  void runOnOperation() override {
    auto func = getOperation();
    auto builder = OpBuilder(func);

    AffineLoopBands targetBands;
    getLoopBands(func.front(), targetBands);

    for (auto &band : targetBands) {
      auto &bodyOps = band.back().getBody()->getOperations();
      if (!isPerfectlyNested(band) || bodyOps.size() != 3)
        continue;

      // Check the copy semantic and make sure the load and store have the same
      // memory access.
      auto load = dyn_cast<AffineLoadOp>(*bodyOps.begin());
      auto store = dyn_cast<AffineStoreOp>(*std::next(bodyOps.begin()));
      if (!load || !store || load.getResult() != store.getValue() ||
          load.getMemref().getType().getShape() !=
              store.getMemref().getType().getShape() ||
          store.getMapOperands() != load.getMapOperands() ||
          store.getAffineMap() != load.getAffineMap())
        continue;

      // Make sure the all loops in the band have constant trip count.
      llvm::SmallDenseMap<Value, unsigned, 4> shapeMap;
      if (llvm::any_of(band, [&](mlir::affine::AffineForOp loop) {
            auto maybeTripCount = getConstantTripCount(loop);
            if (!maybeTripCount.has_value())
              return true;
            shapeMap[loop.getInductionVar()] = maybeTripCount.value();
            return false;
          }))
        continue;

      // Make sure the loop nest accesses each element in the memory once. A
      // dimension reused across results -- `(d0) -> (d0, d0)` sweeps the
      // diagonal -- covers only a slice of the buffer, and raising it to a
      // whole-buffer copy would overwrite elements the nest never touched.
      llvm::SmallDenseSet<unsigned, 4> seenDims;
      auto exprAndShapeRange = llvm::zip(load.getAffineMap().getResults(),
                                         load.getMemRefType().getShape());
      if (llvm::any_of(exprAndShapeRange, [&](auto exprAndShape) {
            AffineExpr expr = std::get<0>(exprAndShape);
            unsigned shape = std::get<1>(exprAndShape);

            if (auto constExpr = dyn_cast<AffineConstantExpr>(expr))
              return constExpr.getValue() != 0 || shape != 1;
            else if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
              if (!seenDims.insert(dimExpr.getPosition()).second)
                return true;
              auto index = load.getMapOperands()[dimExpr.getPosition()];
              return shapeMap.lookup(index) != shape;
            } else
              return true;
          }))
        continue;

      // Replace the loop nest with a copy op.
      builder.setInsertionPoint(band.front());
      memref::CopyOp::create(builder, band.front().getLoc(), load.getMemref(),
                             store.getMemref());
      band.front().erase();
    }
  }
};
} // namespace

std::unique_ptr<Pass> sar::createRaiseAffineToCopyPass() {
  return std::make_unique<RaiseAffineToCopy>();
}
