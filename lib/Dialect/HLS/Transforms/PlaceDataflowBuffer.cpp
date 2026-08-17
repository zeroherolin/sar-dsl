//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
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
/// On-chip tier capacities and the size thresholds that steer a buffer to
/// one of them. A budget of 0 means the tier is unbounded.
struct TierBudget {
  uint64_t bram = 0;
  uint64_t uram = 0;
  uint64_t lutram = 0;
  uint64_t lutramMax = 256;
  uint64_t uramMin = 36864;
};

/// Running occupancy, charged in whole primitives: a 36Kb block holding one
/// kilobyte is spent, so a budget cannot over-promise.
struct TierUse {
  uint64_t bram = 0;
  uint64_t uram = 0;
  uint64_t lutram = 0;
};

constexpr uint64_t kBramBlockBytes = 4608;  // 36 Kb
constexpr uint64_t kUramBlockBytes = 36864; // 288 Kb

static uint64_t roundUpTo(uint64_t bytes, uint64_t grain) {
  return ((bytes + grain - 1) / grain) * grain;
}

/// Not a rewrite pattern: placement is one deterministic sweep over the
/// function, and running it under a pattern driver would only replay the
/// same decisions until the iteration limit.
class Placer {
public:
  Placer(unsigned threshold, TierBudget budget)
      : threshold(threshold), budget(budget) {}

  LogicalResult run(func::FuncOp func) {
    // A declaration has no entry block to read arguments or a terminator
    // from; there is also nothing in it to place.
    if (func.isExternal())
      return success();

    // Placement retypes values in place, and the only view whose result it
    // knows how to re-infer is `memref.subview`. Any other view-like op
    // would be left disagreeing with its retyped source, so reject it here
    // by name rather than let the verifier report a puzzling type mismatch
    // two passes later.
    auto invalid = func.walk([&](Operation *op) {
      if (isa<memref::CastOp, memref::ReinterpretCastOp, memref::ExpandShapeOp,
              memref::CollapseShapeOp>(op)) {
        op->emitOpError("cannot be retyped by buffer placement; fold or "
                        "eliminate it before -hls-place-dataflow-buffer");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (invalid.wasInterrupted())
      return failure();

    // Function arguments participate in the same budget as local buffers:
    // an `ap_memory` interface array occupies the same on-chip primitives
    // as anything the design allocates itself.
    TierUse use;
    for (auto arg : func.getArguments())
      if (auto type = dyn_cast<MemRefType>(arg.getType()))
        if (isPlaceable(type))
          arg.setType(getPlacedType(type, /*isConstBuffer=*/false, use));

    func.walk([&](hls::BufferLikeInterface buffer) {
      if (isPlaceable(buffer.getMemrefType()))
        buffer.getMemref().setType(
            getPlacedType(buffer.getMemrefType(),
                          isa<ConstBufferOp>(buffer.getOperation()), use));
    });

    // A compiled loop (`scf.for` from `sar.iterate`) threads one buffer
    // through its region: the block argument, the yielded buffer and the
    // loop result all stand for the init, so they take its placement --
    // a carry whose ends were placed differently would change memory
    // space mid-loop, which no hardware buffer can do.
    auto carried = func.walk([&](scf::ForOp loop) {
      auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
      for (auto [init, arg, result, yielded] :
           llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs(),
                     loop.getResults(), yield.getOperands())) {
        auto type = dyn_cast<MemRefType>(init.getType());
        if (!type)
          continue;
        arg.setType(type);
        result.setType(type);
        if (yielded == arg || yielded.getType() == type)
          continue;
        Operation *producer = yielded.getDefiningOp();
        if (!producer || !isa<hls::BufferLikeInterface>(producer)) {
          Operation *at = producer ? producer : loop.getOperation();
          at->emitOpError("loop carries must yield a whole buffer or the "
                          "carried value itself");
          return WalkResult::interrupt();
        }
        yielded.setType(type);
      }
      return WalkResult::advance();
    });
    if (carried.wasInterrupted())
      return failure();

    // Buffer placement rewrites the memory kind in place, so any subview
    // taken before this point still carries the old (space-less) result
    // type. Re-infer those so the view keeps agreeing with its source.
    func.walk([](memref::SubViewOp subview) {
      auto sourceType = subview.getSourceType();
      auto resultType =
          cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
              subview.getType().getShape(), sourceType,
              subview.getMixedOffsets(), subview.getMixedSizes(),
              subview.getMixedStrides()));
      // Inference reports the strided layout but not where the buffer
      // lives, so the space has to be carried over from the source or the
      // view and its base end up disagreeing.
      subview.getResult().setType(
          MemRefType::get(resultType.getShape(), resultType.getElementType(),
                          resultType.getLayout(), sourceType.getMemorySpace()));
    });

    func.walk([](YieldOp yield) {
      for (auto t : llvm::zip(yield->getParentOp()->getResults(),
                              yield.getOperandTypes()))
        std::get<0>(t).setType(std::get<1>(t));
    });

    auto builder = Builder(func.getContext());
    func.setType(builder.getFunctionType(
        func.front().getArgumentTypes(),
        func.front().getTerminator()->getOperandTypes()));
    return success();
  }

private:
  /// Bytes a buffer occupies, or 0 when the type cannot be measured (a
  /// dynamic shape, or an element type without an int/float bit width).
  static uint64_t byteSize(MemRefType type) {
    if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat())
      return 0;
    auto width = type.getElementTypeBitWidth();
    return (uint64_t)type.getNumElements() * ((width + 7) / 8);
  }

  /// Whether placement can reason about this type at all. Everything the
  /// affine flow produces qualifies; anything else (dynamic shapes, complex
  /// elements) is left unplaced rather than asserting on its bit width.
  static bool isPlaceable(MemRefType type) {
    return type.hasStaticShape() && type.getElementType().isIntOrFloat();
  }

  /// Assign a tier from the buffer's measured size alone: small buffers go
  /// to distributed RAM where they do not consume a block, planes large
  /// enough to fill ultra RAM go there, everything else to block RAM.
  /// Spilling to DRAM is the last resort, once no tier has room.
  MemoryKind chooseTier(uint64_t bytes, TierUse &use) const {
    auto fits = [](uint64_t budget, uint64_t used, uint64_t need) {
      return budget == 0 || used + need <= budget;
    };

    if (bytes <= budget.lutramMax && fits(budget.lutram, use.lutram, bytes)) {
      use.lutram += bytes;
      return MemoryKind::LUTRAM_S2P;
    }
    if (bytes >= budget.uramMin) {
      uint64_t need = roundUpTo(bytes, kUramBlockBytes);
      if (fits(budget.uram, use.uram, need)) {
        use.uram += need;
        return MemoryKind::URAM_T2P;
      }
    }
    uint64_t need = roundUpTo(bytes, kBramBlockBytes);
    if (fits(budget.bram, use.bram, need)) {
      use.bram += need;
      return MemoryKind::BRAM_T2P;
    }
    return MemoryKind::DRAM;
  }

  MemRefType getPlacedType(MemRefType type, bool isConstBuffer,
                           TierUse &use) const {
    // Two independent decisions, in order. The threshold answers whether a
    // buffer belongs on chip at all -- it is what lets scene size grow past
    // the device -- and the tiers only choose where among the on-chip
    // memories the survivors sit. Letting the tiers answer the first
    // question would keep a plane resident whenever any tier had room,
    // silently overrunning the on-chip budget.
    //
    // Constant buffers never stream: they are the ROM tables the design
    // needs on entry, and `CreateAxiInterface` deliberately keeps them out
    // of the DRAM scratch -- a streamed one would grow the port count past
    // the algorithm's own I/O.
    if (!isConstBuffer && type.getNumElements() >= threshold)
      return withKind(type, MemoryKind::DRAM);

    auto kind = MemoryKind::BRAM_T2P;
    if (uint64_t bytes = byteSize(type))
      kind = chooseTier(bytes, use);
    return withKind(type, kind);
  }

  static MemRefType withKind(MemRefType type, MemoryKind kind) {
    return MemRefType::get(type.getShape(), type.getElementType(),
                           type.getLayout().getAffineMap(),
                           MemoryKindAttr::get(type.getContext(), kind));
  }

  unsigned threshold;
  TierBudget budget;
};
} // namespace

namespace {
/// Moves DRAM-placed buffers out of the dataflow task they were allocated
/// in, up to the level the AXI interface pass expects them at.
struct HoistDramBuffer
    : public OpInterfaceRewritePattern<hls::BufferLikeInterface> {
  using OpInterfaceRewritePattern<
      hls::BufferLikeInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(hls::BufferLikeInterface buffer,
                                PatternRewriter &rewriter) const override {
    if (!isExtBuffer(buffer.getMemref()))
      return failure();
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
  using sar::impl::PlaceDataflowBufferBase<
      PlaceDataflowBuffer>::PlaceDataflowBufferBase;

  PlaceDataflowBuffer() = default;
  PlaceDataflowBuffer(unsigned argThreshold, unsigned argBramBytes,
                      unsigned argUramBytes, unsigned argLutramBytes,
                      unsigned argLutramMaxBytes, unsigned argUramMinBytes) {
    threshold = argThreshold;
    bramBytes = argBramBytes;
    uramBytes = argUramBytes;
    lutramBytes = argLutramBytes;
    lutramMaxBytes = argLutramMaxBytes;
    uramMinBytes = argUramMinBytes;
  }

  void runOnOperation() override {
    auto func = getOperation();
    TierBudget budget{bramBytes, uramBytes, lutramBytes, lutramMaxBytes,
                      uramMinBytes};
    if (failed(Placer(threshold, budget).run(func)))
      return signalPassFailure();

    mlir::RewritePatternSet patterns(func.getContext());
    patterns.add<HoistDramBuffer>(func.getContext());
    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> sar::createPlaceDataflowBufferPass(
    unsigned threshold, unsigned bramBytes, unsigned uramBytes,
    unsigned lutramBytes, unsigned lutramMaxBytes, unsigned uramMinBytes) {
  return std::make_unique<PlaceDataflowBuffer>(threshold, bramBytes, uramBytes,
                                               lutramBytes, lutramMaxBytes,
                                               uramMinBytes);
}
