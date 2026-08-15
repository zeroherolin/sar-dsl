//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_PLACEDATAFLOWBUFFER
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace sar;
using namespace hls;

namespace {
struct PlaceBuffer : public OpRewritePattern<func::FuncOp> {
  PlaceBuffer(MLIRContext *context, unsigned threshold,
              bool placeExternalBuffer)
      : OpRewritePattern<func::FuncOp>(context), threshold(threshold),
        placeExternalBuffer(placeExternalBuffer) {}

  // TODO: For now, we use a heuristic to determine the buffer location.
  MemRefType getPlacedType(MemRefType type, bool isConstBuffer) const {
    auto kind = MemoryKind::BRAM_T2P;
    if (placeExternalBuffer)
      kind = type.getNumElements() >= threshold ? MemoryKind::DRAM
                                                : MemoryKind::BRAM_T2P;
    auto newType = MemRefType::get(
        type.getShape(), type.getElementType(), type.getLayout().getAffineMap(),
        MemoryKindAttr::get(type.getContext(), kind));
    return newType;
  }

  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter &rewriter) const override {
    for (auto arg : func.getArguments())
      if (auto type = dyn_cast<MemRefType>(arg.getType()))
        arg.setType(getPlacedType(type, false));

    func.walk([&](hls::BufferLikeInterface buffer) {
      buffer.getMemref().setType(getPlacedType(
          buffer.getMemrefType(), isa<ConstBufferOp>(buffer.getOperation())));
    });

    // Buffer placement rewrites the memory kind in place, so any subview
    // taken before this point still carries the old (space-less) result
    // type. Re-infer those so the view keeps agreeing with its source.
    func.walk([](memref::SubViewOp subview) {
      auto sourceType = subview.getSourceType();
      auto resultType = cast<MemRefType>(
          memref::SubViewOp::inferRankReducedResultType(
              subview.getType().getShape(), sourceType,
              subview.getMixedOffsets(), subview.getMixedSizes(),
              subview.getMixedStrides()));
      // Inference reports the strided layout but not where the buffer
      // lives, so the space has to be carried over from the source or the
      // view and its base end up disagreeing.
      subview.getResult().setType(MemRefType::get(
          resultType.getShape(), resultType.getElementType(),
          resultType.getLayout(), sourceType.getMemorySpace()));
    });

    func.walk([](YieldOp yield) {
      for (auto t : llvm::zip(yield->getParentOp()->getResults(),
                              yield.getOperandTypes()))
        std::get<0>(t).setType(std::get<1>(t));
    });

    func.setType(rewriter.getFunctionType(
        func.front().getArgumentTypes(),
        func.front().getTerminator()->getOperandTypes()));
    return success();
  }

private:
  unsigned threshold;
  bool placeExternalBuffer;
};
} // namespace

namespace {
/// FIXME: This is super hacky for hoisting all buffers placed in dram to the
/// top level dispatch.
struct HoistDramBuffer
    : public OpInterfaceRewritePattern<hls::BufferLikeInterface> {
  using OpInterfaceRewritePattern<
      hls::BufferLikeInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(hls::BufferLikeInterface buffer,
                                PatternRewriter &rewriter) const override {
    if (!isExtBuffer(buffer.getMemref()))
      return failure();
    // Alwasy move external buffer out of task.
    if (auto task = buffer->getParentOfType<TaskOp>()) {
      buffer->moveBefore(task);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
struct PlaceDataflowBuffer
    : public sar::impl::PlaceDataflowBufferBase<PlaceDataflowBuffer> {
  PlaceDataflowBuffer() = default;
  explicit PlaceDataflowBuffer(unsigned argThreshold,
                               bool argPlaceExternalBuffer) {
    threshold = argThreshold;
    placeExternalBuffer = argPlaceExternalBuffer;
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<PlaceBuffer>(context, threshold, placeExternalBuffer);
    (void)applyOpPatternsGreedily(ArrayRef<Operation *>{func.getOperation()},
                                  std::move(patterns));

    patterns.clear();
    patterns.add<HoistDramBuffer>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass>
sar::createPlaceDataflowBufferPass(unsigned threshold,
                                        bool placeExternalBuffer) {
  return std::make_unique<PlaceDataflowBuffer>(threshold, placeExternalBuffer);
}
