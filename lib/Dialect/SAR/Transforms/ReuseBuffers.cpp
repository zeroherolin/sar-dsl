//===- ReuseBuffers.cpp - Share buffers with disjoint lifetimes -----------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// A linear scan over the function body: a buffer whose last use has
// passed can carry a later one. Buffers of the same type share that type's
// allocation directly; with `allow-retype`, buffers of different types
// share a byte-addressed backing allocation reinterpreted by `memref.view`.
// See the pass description in Passes.td for what the thresholds select.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Support/MathExtras.h"

#include "sar/Dialect/SAR/Transforms/Passes.h"
#include "sar/Support/HLSHints.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SARREUSEBUFFERS
#include "sar/Dialect/SAR/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;

namespace {

/// An allocation and the span of top-level operations over which it is live.
struct BufferLifetime {
  memref::AllocOp alloc;
  /// Top-level position of the first use; before it the contents are
  /// undefined and the buffer is not yet live.
  unsigned def;
  unsigned lastUse;
  /// Position of the allocation itself, which bounds where a shared backing
  /// allocation may be placed.
  unsigned allocPos;
  /// Footprint in bytes, measured through the data layout so that complex
  /// element types account for the padding between the two halves.
  uint64_t bytes;
  /// Alignment the backing storage has to satisfy for this buffer's type.
  uint64_t alignment;
};

/// A set of buffers that will share one allocation, in first-use order.
struct Slot {
  SmallVector<unsigned> members;
  /// Capacity of the backing storage: the first member's footprint. A later
  /// member has to fit inside it -- the retired buffer backs the newer one,
  /// never the other way round, so the footprint never grows.
  uint64_t bytes;
  uint64_t alignment;
  unsigned lastUse;
  /// Null once a member of a different type joins; the group then needs a
  /// byte-addressed backing allocation rather than a typed one.
  MemRefType uniformType;
};

/// Whether this use keeps the buffer's contents to itself.
///
/// An op that hands out a memref or a tensor may create an alias that
/// outlives the visible uses -- `bufferization.to_tensor` is the case
/// that does not return a memref, and an unranked `memref.cast` is why the
/// check reads BaseMemRefType -- and a call or return lets the buffer
/// escape the function entirely. So does a terminator forwarding it to its
/// parent op (`scf.yield`), the buffer being stored *as a value* into
/// another memref, and raw pointer extraction. Either way its lifetime is
/// no longer the span measured here, and a buffer with any such use is
/// left alone. That is also what keeps `memref.subview` and `memref.view`
/// safe: a buffer carrying live views is never a candidate, so sharing one
/// can never alias.
static bool isContainedUse(OpOperand &use) {
  Operation *user = use.getOwner();
  if (isa<func::ReturnOp>(user) || isa<CallOpInterface>(user) ||
      isa<memref::DeallocOp>(user))
    return false;
  if (user->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isa<memref::StoreOp, affine::AffineStoreOp>(user) &&
      use.getOperandNumber() == 0)
    return false;
  if (isa<memref::ExtractAlignedPointerAsIndexOp>(user))
    return false;
  return llvm::none_of(user->getResultTypes(), [](Type type) {
    return isa<BaseMemRefType>(type) || isa<TensorType>(type);
  });
}

/// Whether the data layout can measure `type` without reporting a fatal
/// error. Restricting to the element types the pipelines actually produce
/// keeps an exotic one from aborting the compiler.
static bool isMeasurable(Type type) {
  if (auto complexType = dyn_cast<ComplexType>(type))
    return complexType.getElementType().isIntOrFloat();
  return type.isIntOrFloat();
}

/// Alignment the backing storage must have to hold `type`.
///
/// MLIR's data layout gives a complex type the alignment of its element
/// type, but the value is twice that wide, so a target may want it aligned
/// to its own size; taking the natural power-of-two size (capped at 16, the
/// widest alignment any of these element types asks for) as a floor covers
/// that without ever falling below what the data layout requires.
/// Over-aligning is always sound -- the cost is at most a few bytes.
static uint64_t alignmentFor(Type elementType, const DataLayout &layout) {
  uint64_t abi = layout.getTypeABIAlignment(elementType);
  uint64_t bytes = layout.getTypeSize(elementType).getFixedValue();
  uint64_t natural = std::min<uint64_t>(llvm::PowerOf2Ceil(bytes), 16);
  return std::max(abi, natural);
}

struct SARReuseBuffers
    : public sar::impl::SARReuseBuffersBase<SARReuseBuffers> {
  using SARReuseBuffersBase::SARReuseBuffersBase;

  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal())
      return;

    // Lifetimes are measured as a linear scan, which orders operations only
    // where there is a single straight-line sequence of them. A function
    // with branches has no such order -- a later position is not a later
    // execution -- so the scan does not apply and nothing is shared.
    if (!func.getBody().hasOneBlock())
      return;
    Block &entry = func.front();

    DataLayout layout = DataLayout::closest(func);

    // Lifetimes are measured in top-level operations. The lowered body is a
    // flat sequence of loop nests, so an index into it orders the passes over
    // the raster, which is the granularity buffers are handed between.
    DenseMap<Operation *, unsigned> position;
    unsigned index = 0;
    for (auto &op : entry)
      position[&op] = index++;

    auto topLevelPosition = [&](Operation *op) -> unsigned {
      while (op->getBlock() != &entry)
        op = op->getParentOp();
      return position.lookup(op);
    };

    SmallVector<BufferLifetime> candidates;
    for (auto alloc : entry.getOps<memref::AllocOp>()) {
      auto type = alloc.getType();
      // Banking hints describe an access contract, not merely storage
      // placement. Keep such line/tile buffers distinct: merging them into an
      // unhinted whole-plane slot silently drops the ports their unrolled
      // consumers require and can move the buffer off chip later.
      if (alloc->hasAttr(kPartitionFactorsAttr) ||
          alloc->hasAttr(kPartitionKindsAttr))
        continue;
      if (!type.hasStaticShape() ||
          (uint64_t)type.getNumElements() < minElements)
        continue;
      // A non-identity layout means the buffer's elements are not the
      // contiguous run a shared allocation would hand back.
      if (!type.getLayout().isIdentity() || type.getMemorySpace())
        continue;
      if (!isMeasurable(type.getElementType()))
        continue;
      if (!llvm::all_of(alloc->getUses(),
                        [](OpOperand &use) { return isContainedUse(use); }))
        continue;

      // A buffer is live from its first use, not from its allocation: what
      // sits in between is undefined, so an earlier buffer may still be
      // carrying its own data there. Allocations are commonly hoisted to
      // the top of the block, which would otherwise overlap every lifetime
      // with every other.
      unsigned firstUse = ~0u;
      unsigned lastUse = 0;
      for (auto *user : alloc->getUsers()) {
        unsigned at = topLevelPosition(user);
        firstUse = std::min(firstUse, at);
        lastUse = std::max(lastUse, at);
      }
      if (firstUse == ~0u)
        continue;

      uint64_t elementBytes =
          layout.getTypeSize(type.getElementType()).getFixedValue();
      uint64_t bytes = elementBytes * (uint64_t)type.getNumElements();
      uint64_t alignment = alignmentFor(type.getElementType(), layout);
      // An explicit alignment on the allocation is a promise its users may
      // depend on, so the shared allocation has to keep it.
      if (auto explicitAlignment = alloc.getAlignment())
        alignment = std::max(alignment, *explicitAlignment);

      candidates.push_back({alloc, firstUse, lastUse,
                            position.lookup(alloc.getOperation()), bytes,
                            alignment});
    }

    // Linear scan in first-use order: join the first group that has gone
    // dead and can hold this buffer, otherwise open a new one.
    llvm::stable_sort(candidates,
                      [](const BufferLifetime &a, const BufferLifetime &b) {
                        return a.def < b.def;
                      });

    SmallVector<Slot> slots;
    for (auto [candidateIndex, candidate] : llvm::enumerate(candidates)) {
      auto *slot = llvm::find_if(slots, [&](Slot &live) {
        if (live.lastUse >= candidate.def)
          return false;
        if (live.uniformType == candidate.alloc.getType())
          return true;
        // Types differ: the group needs a byte-addressed backing
        // allocation, which only the CPU path can consume.
        if (!allowRetype)
          return false;
        return live.bytes >= candidate.bytes;
      });
      if (slot == slots.end()) {
        slots.push_back({{(unsigned)candidateIndex},
                         candidate.bytes,
                         candidate.alignment,
                         candidate.lastUse,
                         candidate.alloc.getType()});
        continue;
      }
      slot->members.push_back(candidateIndex);
      slot->lastUse = std::max(slot->lastUse, candidate.lastUse);
      slot->alignment = std::max(slot->alignment, candidate.alignment);
      if (slot->uniformType != candidate.alloc.getType())
        slot->uniformType = nullptr;
    }

    for (Slot &slot : slots)
      materialize(slot, candidates);
  }

  /// Replace a group's separate allocations with the one they share.
  void materialize(Slot &slot, MutableArrayRef<BufferLifetime> candidates) {
    if (slot.members.size() < 2)
      return;

    // The shared allocation goes where the earliest member's did. Every use
    // of every member follows that member's own allocation, so this point
    // dominates all of them; `memref.alloc` of a static shape takes no
    // operands, so nothing it depends on is left behind.
    unsigned earliestIdx = slot.members.front();
    for (unsigned member : slot.members)
      if (candidates[member].allocPos < candidates[earliestIdx].allocPos)
        earliestIdx = member;

    OpBuilder builder(candidates[earliestIdx].alloc);
    Location loc = candidates[earliestIdx].alloc.getLoc();

    if (slot.uniformType) {
      // One type throughout: the members hand their uses to the allocation
      // that already sits earliest, which every backend already understands
      // and which leaves the surrounding IR exactly as it was.
      memref::AllocOp survivor = candidates[earliestIdx].alloc;
      uint64_t strictest = 0;
      for (unsigned member : slot.members) {
        memref::AllocOp alloc = candidates[member].alloc;
        if (auto explicitAlignment = alloc.getAlignment())
          strictest = std::max(strictest, *explicitAlignment);
      }
      for (unsigned member : slot.members)
        if (member != earliestIdx)
          replaceAlloc(candidates[member].alloc, survivor.getResult());
      // An explicit alignment is a promise the buffer's users may depend on,
      // so the one allocation left has to keep the strictest of them.
      if (strictest)
        survivor.setAlignmentAttr(builder.getI64IntegerAttr(strictest));
      return;
    }

    // Types differ: allocate the bytes and let each member reinterpret them.
    // The backing allocation carries the strictest alignment any member
    // needs, since `i8` on its own promises nothing.
    auto backingType =
        MemRefType::get({(int64_t)slot.bytes}, builder.getI8Type());
    auto backing = memref::AllocOp::create(
        builder, loc, backingType, builder.getI64IntegerAttr(slot.alignment));
    auto zero = arith::ConstantIndexOp::create(builder, loc, 0);
    // Build every view before erasing anything. The builder inserts ahead of
    // the earliest member's allocation, so erasing that allocation while more
    // views are still to come would leave the insertion point dangling.
    SmallVector<Value> views;
    for (unsigned member : slot.members)
      views.push_back(memref::ViewOp::create(
                          builder, loc, candidates[member].alloc.getType(),
                          backing.getResult(), zero, /*sizes=*/ValueRange{})
                          .getResult());
    for (auto [view, member] : llvm::zip_equal(views, slot.members))
      replaceAlloc(candidates[member].alloc, view);
  }

  static void replaceAlloc(memref::AllocOp alloc, Value replacement) {
    alloc.getResult().replaceAllUsesWith(replacement);
    alloc.erase();
  }
};

} // namespace
