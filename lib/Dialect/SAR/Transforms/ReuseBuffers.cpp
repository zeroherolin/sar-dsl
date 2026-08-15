//===- ReuseBuffers.cpp - Share buffers with disjoint lifetimes -----------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Bufferization gives every intermediate tensor its own allocation. An
// imaging chain is a sequence of whole-raster passes, so that comes to
// dozens of full rasters -- over a hundred gigabytes at the ALOS size --
// even though only a handful are ever live at the same time.
//
// This pass runs a linear scan over the function body: a buffer whose last
// use precedes another buffer's allocation can carry that one too. Only
// buffers of at least `minElements` take part. Below that size a buffer is a
// dataflow channel, where distinct producers are what let a backend pipeline
// the stages; at and above it the buffer is memory, where sharing costs
// nothing and saves the capacity that decides whether a design fits at all.
// Passing the same threshold the backend uses to place buffers off chip
// keeps the two decisions consistent.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "sar/Dialect/SAR/Transforms/Passes.h"

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
};

/// Whether `user` keeps the buffer's contents to itself.
///
/// An op that returns a memref may hand out an alias that outlives the uses
/// we can see, and a call or return lets the buffer escape the function
/// entirely. Either way its lifetime is no longer the span measured here.
static bool isContainedUser(Operation *user) {
  if (isa<func::ReturnOp>(user) || isa<CallOpInterface>(user) ||
      isa<memref::DeallocOp>(user))
    return false;
  return llvm::none_of(user->getResultTypes(),
                       [](Type type) { return isa<MemRefType>(type); });
}

struct SARReuseBuffers
    : public sar::impl::SARReuseBuffersBase<SARReuseBuffers> {
  using SARReuseBuffersBase::SARReuseBuffersBase;

  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal())
      return;
    Block &entry = func.front();

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
      if (!type.hasStaticShape() ||
          (uint64_t)type.getNumElements() < minElements)
        continue;
      if (!llvm::all_of(alloc->getUsers(), isContainedUser))
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
      candidates.push_back({alloc, firstUse, lastUse});
    }

    // Linear scan in first-use order: reuse the first retained buffer of the
    // same type that has gone dead, otherwise retain this one. The retained
    // buffer's allocation has to dominate the uses it takes over, which the
    // check below enforces directly rather than assuming.
    llvm::stable_sort(candidates,
                      [](const BufferLifetime &a, const BufferLifetime &b) {
                        return a.def < b.def;
                      });
    SmallVector<BufferLifetime> retained;
    for (auto &candidate : candidates) {
      auto *slot = llvm::find_if(retained, [&](BufferLifetime &live) {
        return live.alloc.getType() == candidate.alloc.getType() &&
               live.lastUse < candidate.def &&
               position.lookup(live.alloc) < candidate.def;
      });
      if (slot == retained.end()) {
        retained.push_back(candidate);
        continue;
      }
      candidate.alloc.getResult().replaceAllUsesWith(slot->alloc.getResult());
      candidate.alloc.erase();
      slot->lastUse = candidate.lastUse;
    }
  }
};

} // namespace
