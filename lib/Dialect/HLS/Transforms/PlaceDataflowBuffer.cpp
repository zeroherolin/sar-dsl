//===- PlaceDataflowBuffer.cpp - place dataflow buffer --------------------===//
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
/// On-chip tier capacities: hard caps the placed design never exceeds. A
/// budget of 0 forbids the tier. `lutramMax` steers small buffers to
/// distributed RAM; the URAM floor is the physical block size below.
struct TierBudget {
  uint64_t bram = 0;
  uint64_t uram = 0;
  uint64_t lutram = 0;
  uint64_t lutramMax = 256;
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
  Placer(unsigned threshold, TierBudget budget, bool rebalanceOnly,
         bool allowDram)
      : threshold(threshold), budget(budget), rebalanceOnly(rebalanceOnly),
        allowDram(allowDram) {}

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
      if (!isPlaceable(buffer.getMemrefType()))
        return;
      // A buffer allocated inside a compiled loop lives for one
      // iteration: DRAM placement would promise it an AXI port or a
      // scratch slot, neither of which can be carved through an
      // `scf.for` region -- and a one-iteration working buffer belongs
      // on chip in any case. Treating it like a constant buffer keeps
      // the threshold rule off it while the tier budgets still apply.
      bool perIteration = buffer->getParentOfType<scf::ForOp>() != nullptr;
      buffer.getMemref().setType(getPlacedType(
          buffer.getMemrefType(),
          isa<ConstBufferOp>(buffer.getOperation()) || perIteration, use));
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

    // Buffer placement rewrites memory kinds in place. Re-infer subviews so
    // their result types retain the source memory space.
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

    // Re-placement after forking or balancing must propagate updated types
    // through isolated schedule and node regions.
    for (auto schedule : func.getOps<ScheduleOp>())
      schedule.updateSignatureRecursively();

    auto builder = Builder(func.getContext());
    func.setType(builder.getFunctionType(
        func.front().getArgumentTypes(),
        func.front().getTerminator()->getOperandTypes()));
    lastUse = use;
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
  ///
  /// A dataflow channel is charged at twice its primitive count: Vitis
  /// double-buffers every channel (ping-pong), so the synthesized design
  /// holds two copies of each. Constant buffers are ROMs and are read in
  /// place -- they stay at one copy. Charging the doubling here is what
  /// keeps the budgets honest: a budget set to a device fraction limits the
  /// synthesized channels to that fraction.
  MemoryKind chooseTier(MemRefType type, uint64_t bytes, TierUse &use,
                        bool isConst) const {
    auto fits = [](uint64_t budget, uint64_t used, uint64_t need) {
      return used + need <= budget;
    };
    uint64_t copies = isConst ? 1 : 2;
    uint64_t banks = std::max<int64_t>(1, getPartitionFactors(type));
    uint64_t bankBytes = (bytes + banks - 1) / banks;

    if (bytes <= budget.lutramMax &&
        fits(budget.lutram, use.lutram, copies * bytes)) {
      use.lutram += copies * bytes;
      return MemoryKind::LUTRAM_S2P;
    }
    uint64_t uramNeed = copies * banks * roundUpTo(bankBytes, kUramBlockBytes);
    if (bytes >= kUramBlockBytes && fits(budget.uram, use.uram, uramNeed)) {
      use.uram += uramNeed;
      return MemoryKind::URAM_T2P;
    }
    uint64_t need = copies * banks * roundUpTo(bankBytes, kBramBlockBytes);
    if (fits(budget.bram, use.bram, need)) {
      use.bram += need;
      return MemoryKind::BRAM_T2P;
    }
    // Block RAM is exhausted. A buffer below the URAM block size now takes
    // a URAM block anyway when one is free: it wastes part of the block,
    // but overflowing the device's block RAM would waste the whole design
    // -- this is what soaks up the mid-size working set (row buffers,
    // banded gather tables) of large-scene designs.
    if (bytes < kUramBlockBytes && fits(budget.uram, use.uram, uramNeed)) {
      use.uram += uramNeed;
      return MemoryKind::URAM_T2P;
    }
    return MemoryKind::DRAM;
  }

  MemRefType getPlacedType(MemRefType type, bool isConstBuffer, TierUse &use) {
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
    //
    // A rebalance keeps DRAM placements as they are and never adds new
    // ones: this late the graph is scheduled and the external set is
    // already carved into ports.
    if (rebalanceOnly) {
      if (getMemoryKind(type) == MemoryKind::DRAM)
        return type;
    } else if (allowDram && !isConstBuffer &&
               type.getNumElements() >= threshold) {
      return withKind(type, MemoryKind::DRAM);
    }

    auto kind = MemoryKind::BRAM_T2P;
    if (uint64_t bytes = byteSize(type)) {
      kind = chooseTier(type, bytes, use, isConstBuffer);
      if (rebalanceOnly && kind == MemoryKind::DRAM) {
        // No tier has budget left and nothing may newly stream this late.
        // Keep the placement the buffer already carries and record the
        // overflow; the driver turns it into a hard failure.
        auto placed = getMemoryKind(type);
        kind = placed == MemoryKind::UNKNOWN ? MemoryKind::BRAM_T2P : placed;
        uint64_t banks = std::max<int64_t>(1, getPartitionFactors(type));
        uint64_t bankBytes = (bytes + banks - 1) / banks;
        overflowBytes += (isConstBuffer ? 1 : 2) * banks *
                         roundUpTo(bankBytes, kBramBlockBytes);
      } else if (kind == MemoryKind::DRAM && (isConstBuffer || !allowDram)) {
        // Constant tables and per-iteration scratch cannot stream. The local
        // ap_memory protocol also has no external master through which any
        // ordinary buffer could spill.
        uint64_t banks = std::max<int64_t>(1, getPartitionFactors(type));
        uint64_t bankBytes = (bytes + banks - 1) / banks;
        uint64_t cost = (isConstBuffer ? 1 : 2) * banks *
                        roundUpTo(bankBytes, kBramBlockBytes);
        if (isConstBuffer)
          constOverflowBytes += cost;
        else
          externalOverflowBytes += cost;
        kind = MemoryKind::BRAM_T2P;
      }
    }
    return withKind(type, kind);
  }

  static MemRefType withKind(MemRefType type, MemoryKind kind) {
    return MemRefType::get(type.getShape(), type.getElementType(),
                           type.getLayout().getAffineMap(),
                           MemoryKindAttr::get(type.getContext(), kind));
  }

  unsigned threshold;
  TierBudget budget;
  bool rebalanceOnly;
  bool allowDram;

public:
  /// Bytes a rebalance could not fit into any tier budget and left at the
  /// buffers' original placement (charged in whole primitives, ping-pong
  /// included).
  uint64_t overflowBytesUsed() const { return overflowBytes; }
  uint64_t constOverflowBytesUsed() const { return constOverflowBytes; }
  uint64_t externalOverflowBytesUsed() const { return externalOverflowBytes; }
  uint64_t lutramBytesUsed() const { return lastUse.lutram; }

private:
  TierUse lastUse;
  uint64_t overflowBytes = 0;
  uint64_t constOverflowBytes = 0;
  uint64_t externalOverflowBytes = 0;
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
                      unsigned argLutramMaxBytes, bool argRebalanceOnly,
                      bool argAllowDram) {
    threshold = argThreshold;
    bramBytes = argBramBytes;
    uramBytes = argUramBytes;
    lutramBytes = argLutramBytes;
    lutramMaxBytes = argLutramMaxBytes;
    rebalanceOnly = argRebalanceOnly;
    allowDram = argAllowDram;
  }

  void runOnOperation() override {
    auto func = getOperation();
    TierBudget budget{bramBytes, uramBytes, lutramBytes, lutramMaxBytes};
    Placer placer(threshold, budget, rebalanceOnly, allowDram);
    if (failed(placer.run(func)))
      return signalPassFailure();

    // The banking pass later spends distributed RAM from the same cap, so
    // what placement consumed travels with the function; the second
    // (rebalance) run overwrites the first with the final figure.
    func->setAttr("lutram_spent",
                  IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                   placer.lutramBytesUsed()));

    // The budgets are hard caps: a working set the tiers cannot hold is a
    // design that will not fit the device, and emitting it anyway would
    // hand the user a permanently invalid design. Fail here; the backend
    // reacts by streaming more planes and retrying.
    if (placer.constOverflowBytesUsed()) {
      func.emitError()
          << "constant tables and per-iteration buffers need "
          << placer.constOverflowBytesUsed()
          << " bytes more on-chip memory than the tier budgets allow; they "
             "cannot stream -- raise the budgets or shrink the tables";
      return signalPassFailure();
    }
    if (placer.externalOverflowBytesUsed()) {
      func.emitError()
          << "off-chip DRAM spill is disabled for this interface, but the "
             "working set needs "
          << placer.externalOverflowBytesUsed()
          << " bytes of on-chip storage; raise the memory budgets or use an "
             "AXI interface";
      return signalPassFailure();
    }
    if (rebalanceOnly && placer.overflowBytesUsed()) {
      func.emitError()
          << "SAR_HLS_RETRYABLE_MEMORY_OVERFLOW: on-chip working set "
             "exceeds the memory budgets by "
          << placer.overflowBytesUsed()
          << " bytes (ping-pong buffering included); stream more planes or "
             "raise the tier budgets";
      return signalPassFailure();
    }

    mlir::RewritePatternSet patterns(func.getContext());
    patterns.add<HoistDramBuffer>(func.getContext());
    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass>
sar::createPlaceDataflowBufferPass(unsigned threshold, unsigned bramBytes,
                                   unsigned uramBytes, unsigned lutramBytes,
                                   unsigned lutramMaxBytes, bool rebalanceOnly,
                                   bool allowDram) {
  return std::make_unique<PlaceDataflowBuffer>(threshold, bramBytes, uramBytes,
                                               lutramBytes, lutramMaxBytes,
                                               rebalanceOnly, allowDram);
}
