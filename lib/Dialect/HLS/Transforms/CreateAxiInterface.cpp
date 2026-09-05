//===- CreateAxiInterface.cpp - create AXI interfaces ---------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

#include "llvm/ADT/MapVector.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_CREATEAXIINTERFACE
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

//===----------------------------------------------------------------------===//
// Internal DRAM scratch
//
// Spilled internal buffers are carved into typed arenas at aligned offsets.
// Buffers participating in a node's writes use different arenas so AXI
// requests can proceed concurrently; pure read fan-in remains co-located for
// locality. Coloring reuses masters across non-overlapping nodes while the
// configured arena limit bounds the external interface.
//===----------------------------------------------------------------------===//

namespace {

/// Bytes a carved buffer starts on, so no slot straddles an AXI beat.
constexpr uint64_t kScratchAlignBytes = 64;

/// Element count below which a read-only input shares an AXI master rather
/// than claiming one. Sized as a raster line: an axis, a window or a
/// reference chirp is a table of that order, read a row at a time, so
/// several fit one master without the bursts colliding. A full plane is
/// orders of magnitude larger and gets its own.
constexpr int64_t kSharedInputMaxElements = 1 << 16;

/// One carved buffer: where its elements start in the scratch allocation.
struct ScratchSlot {
  hls::BufferLikeInterface buffer;
  int64_t offset;
};

/// Accesses to redirect once a buffer has been accepted for carving. Collected
/// before anything is rewritten, so a buffer reaching a use that cannot be
/// redirected is left alone rather than half-carved.
struct ScratchUses {
  SmallVector<AffineLoadOp> loads;
  SmallVector<AffineStoreOp> stores;
  /// Accesses whose indices are values rather than an affine map -- the
  /// data-dependent gathers of interpolation.
  SmallVector<memref::LoadOp> valueLoads;
  SmallVector<memref::StoreOp> valueStores;
  /// Dataflow node arguments the buffer is bound to, in visit order.
  SmallVector<BlockArgument> args;
};

/// Row-major strides of a contiguous buffer.
static SmallVector<int64_t> rowMajorStrides(MemRefType type) {
  auto shape = type.getShape();
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = (int)shape.size() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

/// A buffer can be carved when it is a plain contiguous array: the offset
/// arithmetic below assumes row-major strides and nothing else.
static bool isCarvable(MemRefType type) {
  return type.hasStaticShape() && type.getLayout().isIdentity() &&
         type.getElementTypeBitWidth() != 0;
}

/// Convert one proven row-major public buffer to an HLS stream.  The affine
/// pipeline keeps stream candidates as memrefs until the interface pass so
/// dataflow and banking can still reason about ordinary memory.  At the ABI
/// boundary, replace the complete access chain with FIFO operations and
/// update every callee argument reached by the chain.  The caller has already
/// proved that there is exactly one monotonic sweep, so dropping the indices
/// is semantics-preserving.
static LogicalResult convertSequentialStream(Value value, StreamType streamType,
                                             DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return success();

  SmallVector<OpOperand *> uses;
  for (OpOperand &use : value.getUses())
    uses.push_back(&use);

  for (OpOperand *use : uses) {
    Operation *owner = use->getOwner();
    if (auto load = dyn_cast<AffineLoadOp>(owner)) {
      if (use->getOperandNumber() != 0)
        return failure();
      OpBuilder builder(load);
      Value read = StreamReadOp::create(builder, load.getLoc(),
                                        load.getResult().getType(), value)
                       .getResult();
      load.getResult().replaceAllUsesWith(read);
      load.erase();
      continue;
    }
    if (auto store = dyn_cast<AffineStoreOp>(owner)) {
      if (use->getOperandNumber() != 1)
        return failure();
      OpBuilder builder(store);
      StreamWriteOp::create(builder, store.getLoc(), value,
                            store.getValueToStore());
      store.erase();
      continue;
    }
    if (auto load = dyn_cast<memref::LoadOp>(owner)) {
      if (use->getOperandNumber() != 0)
        return failure();
      OpBuilder builder(load);
      Value read = StreamReadOp::create(builder, load.getLoc(),
                                        load.getResult().getType(), value)
                       .getResult();
      load.getResult().replaceAllUsesWith(read);
      load.erase();
      continue;
    }
    if (auto store = dyn_cast<memref::StoreOp>(owner)) {
      if (use->getOperandNumber() != 1)
        return failure();
      OpBuilder builder(store);
      StreamWriteOp::create(builder, store.getLoc(), value,
                            store.getValueToStore());
      store.erase();
      continue;
    }
    if (auto call = dyn_cast<func::CallOp>(owner)) {
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      if (!callee || callee.isExternal() ||
          use->getOperandNumber() >= callee.getNumArguments())
        return failure();
      BlockArgument argument = callee.getArgument(use->getOperandNumber());
      if (argument.getType() != streamType)
        argument.setType(streamType);
      if (failed(convertSequentialStream(argument, streamType, visited)))
        return failure();
      continue;
    }
    return failure();
  }
  return success();
}

static void refreshFunctionTypes(ModuleOp module, MLIRContext *context) {
  Builder builder(context);
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal() || func.empty() ||
        !func.front().mightHaveTerminator())
      continue;
    auto returnOp = dyn_cast<func::ReturnOp>(func.front().getTerminator());
    if (!returnOp)
      continue;
    func.setType(builder.getFunctionType(func.front().getArgumentTypes(),
                                         returnOp.getOperandTypes()));
  }
}

/// The flat index a row-major access of `type` lands on, shifted by `offset`.
static AffineMap flattenMap(AffineMap map, MemRefType type, int64_t offset) {
  auto strides = rowMajorStrides(type);
  auto flat = getAffineConstantExpr(offset, map.getContext());
  for (auto [index, result] : llvm::enumerate(map.getResults()))
    flat = flat + result * strides[index];
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), flat,
                        map.getContext());
}

/// The same flattening for indices that are values, materialized as integer
/// arithmetic in front of `op`.
static Value flattenIndices(Operation *op, ValueRange indices, MemRefType type,
                            int64_t offset, OpBuilder &builder) {
  builder.setInsertionPoint(op);
  auto loc = op->getLoc();
  auto strides = rowMajorStrides(type);

  Value flat;
  for (auto [index, value] : llvm::enumerate(indices)) {
    Value term = value;
    if (strides[index] != 1)
      term = arith::MulIOp::create(
          builder, loc, term,
          arith::ConstantIndexOp::create(builder, loc, strides[index]));
    flat = flat ? arith::AddIOp::create(builder, loc, flat, term).getResult()
                : term;
  }
  if (offset)
    flat = arith::AddIOp::create(
        builder, loc, flat,
        arith::ConstantIndexOp::create(builder, loc, offset));
  return flat;
}

/// Walks everything reached by `buffer` -- accesses in this function, and the
/// dataflow node calls it is handed to -- collecting what a rewrite would have
/// to touch. Fails, without recording anything, on an unredirectable use or
/// a callee shared by several call sites (which would need one offset to serve
/// two buffers).
static LogicalResult
collectScratchUses(Value buffer, ModuleOp module,
                   const DenseMap<StringRef, unsigned> &callerCounts,
                   ScratchUses &uses, DenseSet<Value> &visited) {
  if (!visited.insert(buffer).second)
    return success();

  for (auto &use : buffer.getUses()) {
    auto *owner = use.getOwner();
    if (auto load = dyn_cast<AffineLoadOp>(owner)) {
      uses.loads.push_back(load);
    } else if (auto store = dyn_cast<AffineStoreOp>(owner)) {
      if (use.getOperandNumber() != 1)
        return failure();
      uses.stores.push_back(store);
    } else if (auto load = dyn_cast<memref::LoadOp>(owner)) {
      uses.valueLoads.push_back(load);
    } else if (auto store = dyn_cast<memref::StoreOp>(owner)) {
      if (use.getOperandNumber() != 1)
        return failure();
      uses.valueStores.push_back(store);
    } else if (auto call = dyn_cast<func::CallOp>(owner)) {
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      if (!callee || callee.isExternal())
        return failure();
      auto count = callerCounts.find(callee.getName());
      if (count == callerCounts.end() || count->second != 1)
        return failure();

      auto arg = callee.getArgument(use.getOperandNumber());
      if (!visited.contains(arg))
        uses.args.push_back(arg);
      if (failed(collectScratchUses(arg, module, callerCounts, uses, visited)))
        return failure();
    } else {
      return failure();
    }
  }
  return success();
}

/// Points every collected access at the scratch layout: the indices are
/// flattened and shifted, and the buffer itself is replaced by `scratch`
/// wherever it was named. Accesses inside a dataflow node keep naming that
/// node's own argument, which is retyped in place.
static void redirectToScratch(const ScratchUses &uses, Value buffer,
                              Value scratch, MemRefType oldType, int64_t offset,
                              OpBuilder &builder) {
  auto scratchType = cast<MemRefType>(scratch.getType());

  for (auto load : uses.loads)
    load.setMap(flattenMap(load.getAffineMap(), oldType, offset));
  for (auto store : uses.stores)
    store.setMap(flattenMap(store.getAffineMap(), oldType, offset));
  for (auto load : uses.valueLoads) {
    auto flat =
        flattenIndices(load, load.getIndices(), oldType, offset, builder);
    load.getIndicesMutable().assign(flat);
  }
  for (auto store : uses.valueStores) {
    auto flat =
        flattenIndices(store, store.getIndices(), oldType, offset, builder);
    store.getIndicesMutable().assign(flat);
  }

  for (auto arg : uses.args) {
    arg.setType(scratchType);
    auto func = cast<func::FuncOp>(arg.getOwner()->getParentOp());
    func.setType(builder.getFunctionType(
        func.front().getArgumentTypes(),
        func.front().getTerminator()->getOperandTypes()));
  }
  buffer.replaceAllUsesWith(scratch);
}

/// What a dataflow node does with a buffer it was handed.
struct AccessRole {
  bool reads = false;
  bool writes = false;
};

/// Records how `value` is used, following into the callees it is passed to.
/// A use this walk does not model counts as both a read and a write, which
/// keeps the conflict graph conservative without inventing alias pressure.
static void classifyAccess(Value value, AccessRole &role,
                           DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return;

  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (isa<AffineLoadOp, memref::LoadOp>(owner)) {
      role.reads = true;
    } else if (isa<AffineStoreOp, memref::StoreOp>(owner)) {
      role.writes = true;
    } else if (auto call = dyn_cast<func::CallOp>(owner)) {
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      if (!callee || callee.isExternal()) {
        role.reads = role.writes = true;
        continue;
      }
      classifyAccess(callee.getArgument(use.getOperandNumber()), role, visited);
    } else {
      role.reads = role.writes = true;
    }
  }
}

/// Returns the outlined dataflow call that consumes a read-only public input,
/// or null when the input is read by more than one call (or directly in the
/// implementation). Vitis HLS 2022.2 does not permit one m_axi bundle to be
/// read by multiple dataflow processes: it reports HLS 200-984 and refuses
/// synthesis. Inputs consumed by the same process may still share a master,
/// which preserves the useful channel reduction for, for example, a pair of
/// windows used by one phase.
static Operation *singleReadCallSite(Value value, DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return nullptr;

  Operation *site = nullptr;
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (auto call = dyn_cast<func::CallOp>(owner)) {
      if (site && site != call.getOperation())
        return nullptr;
      site = call.getOperation();
      continue;
    }

    // Public buffers normally reach outlined nodes directly. If a harmless
    // forwarding op appears between the argument and that call, recurse only
    // through its single result; unknown users are conservatively rejected so
    // the interface pass never creates a bundle whose access shape it cannot
    // prove safe.
    if (isa<memref::CastOp, memref::SubViewOp, memref::ViewOp>(owner) &&
        owner->getNumResults() == 1 && use.getOperandNumber() == 0) {
      Operation *forwarded = singleReadCallSite(owner->getResult(0), visited);
      if (!forwarded)
        return nullptr;
      if (site && site != forwarded)
        return nullptr;
      site = forwarded;
      continue;
    }
    return nullptr;
  }
  return site;
}

/// Check the concrete access shape before converting a memref to a FIFO.  A
/// stream has no address channel, so every dimension must be driven by its
/// zero-based unit-step loop and the complete row-major extent must be
/// consumed exactly once.
static bool isProvablySequentialStream(Value value, bool input,
                                       unsigned &accesses,
                                       DenseSet<Value> &active) {
  if (!active.insert(value).second)
    return false;
  bool valid = true;
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (auto call = dyn_cast<func::CallOp>(owner)) {
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      if (!callee || callee.isExternal() || isNestedInLoop(call) ||
          use.getOperandNumber() >= callee.getNumArguments() ||
          !isProvablySequentialStream(
              callee.getArgument(use.getOperandNumber()), input, accesses,
              active)) {
        valid = false;
        break;
      }
      continue;
    }
    bool expected =
        input ? isa<AffineLoadOp>(owner) : isa<AffineStoreOp>(owner);
    if (!expected || !isCompleteRowMajorSweep(owner, value)) {
      valid = false;
      break;
    }
    ++accesses;
  }
  active.erase(value);
  return valid;
}

static bool isProvablySequentialStream(Value value, const AccessRole &role) {
  if (role.reads == role.writes || (!role.reads && !role.writes))
    return false;
  unsigned accesses = 0;
  DenseSet<Value> active;
  return isProvablySequentialStream(value, role.reads, accesses, active) &&
         accesses == 1;
}

/// Partitions `candidates` into arenas, returning each one's arena index.
///
/// A read/write pair used by one node is a conflict: a single pointer cannot
/// carry both sides without alias serialization. Pure read/read fan-in and
/// sequential writes may share an arena.
///
/// Colors most-constrained-first. The order matters for the count, not just
/// the assignment: taking buffers in declaration order makes a graph two
/// arenas cover ask for three, because a buffer conflicting with everything
/// claims a color before the pair it separates has been placed.
static SmallVector<unsigned>
colorScratchBanks(ArrayRef<hls::BufferLikeInterface> candidates,
                  unsigned maxArenas, bool &overflowed,
                  uint64_t &overflowPenalty) {
  unsigned count = candidates.size();
  SmallVector<unsigned> colors(count, 0);
  if (count < 2)
    return colors;

  llvm::MapVector<Operation *, SmallVector<unsigned>> readers, writers;
  for (auto [index, entry] : llvm::enumerate(candidates)) {
    hls::BufferLikeInterface buffer = entry;
    for (OpOperand &use : buffer.getMemref().getUses()) {
      auto call = dyn_cast<func::CallOp>(use.getOwner());
      if (!call)
        continue;
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      AccessRole role;
      if (!callee || callee.isExternal()) {
        role.reads = role.writes = true;
      } else {
        DenseSet<Value> visited;
        classifyAccess(callee.getArgument(use.getOperandNumber()), role,
                       visited);
      }
      if (role.reads)
        readers[call].push_back(index);
      if (role.writes)
        writers[call].push_back(index);
    }
  }

  SmallVector<DenseSet<unsigned>> conflicts(count);
  for (auto &[call, read] : readers) {
    auto written = writers.find(call);
    if (written == writers.end())
      continue;
    for (unsigned source : read)
      for (unsigned target : written->second)
        if (source != target) {
          conflicts[source].insert(target);
          conflicts[target].insert(source);
        }
  }
  // With no more buffers than arenas, any conflict graph still colors within
  // the limit, so separating simultaneous writers as well costs nothing.
  if (count <= maxArenas)
    for (auto &[call, written] : writers)
      for (auto [position, source] : llvm::enumerate(written))
        for (unsigned target : ArrayRef(written).drop_front(position + 1))
          if (source != target) {
            conflicts[source].insert(target);
            conflicts[target].insert(source);
          }

  // A pure producer has no read/write alias edges, but its parallel stores
  // still benefit from every master the interface contract permits.
  if (readers.empty() && !writers.empty() &&
      llvm::all_of(conflicts,
                   [](const DenseSet<unsigned> &set) { return set.empty(); })) {
    for (unsigned index = 0; index < count; ++index)
      colors[index] = index % maxArenas;
    if (count > maxArenas) {
      overflowed = true;
      overflowPenalty = count - maxArenas;
    }
    return colors;
  }

  SmallVector<unsigned> order(count);
  std::iota(order.begin(), order.end(), 0u);
  llvm::stable_sort(order, [&](unsigned lhs, unsigned rhs) {
    return conflicts[lhs].size() > conflicts[rhs].size();
  });

  SmallVector<bool> assigned(count, false);
  for (unsigned index : order) {
    SmallVector<bool> taken(maxArenas, false);
    for (unsigned neighbor : conflicts[index])
      if (assigned[neighbor])
        taken[colors[neighbor]] = true;
    unsigned color = 0;
    while (color < maxArenas && taken[color])
      ++color;
    if (color == maxArenas) {
      overflowed = true;
      color = 0;
    }
    colors[index] = color;
    assigned[index] = true;
  }

  // Also evaluate the stricter graph that separates simultaneous writers.
  // It is profitable while only a small number of edges must be compressed;
  // dense graphs are better served by preserving every read/write split and
  // letting sequential writers reuse those masters.
  SmallVector<DenseSet<unsigned>> writeConflicts = conflicts;
  for (auto &[call, written] : writers)
    for (auto [position, source] : llvm::enumerate(written))
      for (unsigned target : ArrayRef(written).drop_front(position + 1))
        if (source != target) {
          writeConflicts[source].insert(target);
          writeConflicts[target].insert(source);
        }
  SmallVector<unsigned> writeOrder(count);
  std::iota(writeOrder.begin(), writeOrder.end(), 0u);
  llvm::stable_sort(writeOrder, [&](unsigned lhs, unsigned rhs) {
    return writeConflicts[lhs].size() > writeConflicts[rhs].size();
  });
  SmallVector<unsigned> writeColors(count, 0);
  SmallVector<bool> writeAssigned(count, false);
  for (unsigned index : writeOrder) {
    SmallVector<bool> taken(maxArenas, false);
    for (unsigned neighbor : writeConflicts[index])
      if (writeAssigned[neighbor])
        taken[writeColors[neighbor]] = true;
    unsigned color = 0;
    while (color < maxArenas && taken[color])
      ++color;
    writeColors[index] = color < maxArenas ? color : 0;
    writeAssigned[index] = true;
  }
  uint64_t writePenalty = 0;
  for (unsigned lhs = 0; lhs < count; ++lhs)
    for (unsigned rhs = lhs + 1; rhs < count; ++rhs)
      if (writeColors[lhs] == writeColors[rhs] &&
          writeConflicts[lhs].contains(rhs))
        ++writePenalty;
  if (writePenalty <= 2) {
    colors = std::move(writeColors);
    overflowed = writePenalty != 0;
    overflowPenalty = writePenalty;
    return colors;
  }

  for (unsigned lhs = 0; lhs < count; ++lhs)
    for (unsigned rhs = lhs + 1; rhs < count; ++rhs)
      if (colors[lhs] == colors[rhs] && conflicts[lhs].contains(rhs))
        ++overflowPenalty;
  return colors;
}

/// Replaces one group of internal DRAM buffers with slices of a single scratch
/// allocation, and returns that allocation for the caller to turn into a port.
static FailureOr<Value>
createScratchBuffer(func::FuncOp func, ModuleOp module, OpBuilder &builder,
                    Type elementType,
                    ArrayRef<hls::BufferLikeInterface> candidates) {
  DenseMap<StringRef, unsigned> callerCounts;
  module.walk([&](func::CallOp call) { ++callerCounts[call.getCallee()]; });

  unsigned widest = elementType.getIntOrFloatBitWidth();

  // A buffer may ask to start at a value. One allocation carries one such
  // value, so the scratch adopts it -- initializing a buffer that asked for
  // nothing is harmless, contradicting one that asked for something is not,
  // so a second, different request is rejected below.
  TypedAttr initValue;
  for (auto buffer : candidates)
    if (auto buf = dyn_cast<BufferOp>(*buffer))
      if (auto init = buf.getInitValueAttr()) {
        initValue = init;
        break;
      }

  // Lay the carvable buffers out back to back, each starting on a beat
  // boundary, recording what each one needs redirecting.
  int64_t align = std::max<int64_t>(
      1, kScratchAlignBytes / std::max<uint64_t>(1, (widest + 7) / 8));
  SmallVector<ScratchSlot> slots;
  SmallVector<ScratchUses, 8> slotUses;
  SmallVector<hls::BufferLikeInterface> skipped;
  int64_t elements = 0;
  for (auto buffer : candidates) {
    auto type = buffer.getMemrefType();
    auto buf = dyn_cast<BufferOp>(*buffer);
    auto init = buf ? buf.getInitValueAttr() : TypedAttr();
    ScratchUses uses;
    DenseSet<Value> visited;
    if (!isCarvable(type) || (init && init != initValue) ||
        failed(collectScratchUses(buffer.getMemref(), module, callerCounts,
                                  uses, visited))) {
      skipped.push_back(buffer);
      continue;
    }

    slots.push_back({buffer, elements});
    slotUses.push_back(std::move(uses));
    elements += ((int64_t)type.getNumElements() + align - 1) / align * align;
  }

  if (!skipped.empty()) {
    for (auto buffer : skipped)
      buffer->emitError("internal DRAM buffer cannot be carved into the "
                        "scratch allocation; a stable AXI interface cannot "
                        "expose optimizer-created buffers as ports");
    return failure();
  }

  // Only spilled buffers reach here, so the arena always has content; the
  // guard keeps a degenerate zero-element buffer from declaring `T arena[0]`.
  auto scratchType = MemRefType::get(
      {std::max<int64_t>(1, elements)}, elementType, AffineMap(),
      MemoryKindAttr::get(func.getContext(), MemoryKind::DRAM));
  builder.setInsertionPointToStart(&func.front());
  Value scratch = BufferOp::create(builder, func.getLoc(), scratchType,
                                   /*depth=*/1, initValue);
  scratch.getDefiningOp()->setAttr("hls.scratch", builder.getUnitAttr());

  for (auto [slot, uses] : llvm::zip(slots, slotUses)) {
    redirectToScratch(uses, slot.buffer.getMemref(), scratch,
                      slot.buffer.getMemrefType(), slot.offset, builder);
    slot.buffer->erase();
  }

  return scratch;
}
} // namespace

namespace {
struct CreateAxiInterface
    : public sar::impl::CreateAxiInterfaceBase<CreateAxiInterface> {
  CreateAxiInterface() = default;
  CreateAxiInterface(std::string hlsTopFunc, bool argStreamInterface,
                     unsigned argMaxScratchArenas) {
    topFunc = hlsTopFunc;
    streamInterface = argStreamInterface;
    maxScratchArenas = argMaxScratchArenas;
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module);
    auto context = builder.getContext();
    auto loc = builder.getUnknownLoc();

    // Get the top function of the module.
    auto func = getTopFunc(module, topFunc);
    if (!func) {
      emitError(module.getLoc(), "failed to find the top function");
      return signalPassFailure();
    }
    setTopFuncAttr(func);

    // An AXI4-Stream design differs from a memory-mapped one only in the
    // protocol its top-level ports speak, so it is recorded as an attribute
    // rather than in the port types: `hls.axi.port` requires a memref to be
    // mapped to an MM bundle, and the buffers reaching the signature are
    // memrefs whichever protocol carries them. The emitter reads this to
    // emit `axis` pragmas in place of `m_axi` ones.
    if (streamInterface)
      func->setAttr("stream_interface", builder.getUnitAttr());

    // Carve the internal DRAM buffers into arenas: one per element type that
    // actually spills, split again wherever a node would end up reading and
    // writing the same pointer. A type with no spill contributes no arena, so
    // the port list stays the algorithm's own I/O plus the scratch the design
    // needs to keep its nodes pipelined.
    llvm::MapVector<Type, SmallVector<hls::BufferLikeInterface>> scratchBanks;
    for (auto buffer : func.getOps<hls::BufferLikeInterface>()) {
      // A constant buffer holds data the design needs on entry, so it is not
      // scratch: it keeps its own storage and its initializer.
      if (isa<ConstBufferOp>(*buffer) || !isExtBuffer(buffer.getMemref()))
        continue;
      Type elementType = buffer.getMemrefType().getElementType();
      if (!elementType.isIntOrFloat()) {
        // Carving computes byte offsets from a scalar width. An aggregate
        // element type has none, and letting the buffer through here would
        // leave it to become a port of its own further down.
        buffer->emitError("internal DRAM buffer of non-scalar element type "
                          "cannot be carved into a scratch arena");
        return signalPassFailure();
      }
      scratchBanks[elementType].push_back(buffer);
    }

    for (auto &[elementType, candidates] : scratchBanks) {
      bool overflowed = false;
      uint64_t overflowPenalty = 0;
      auto colors = colorScratchBanks(candidates,
                                      std::max(1u, maxScratchArenas.getValue()),
                                      overflowed, overflowPenalty);
      if (overflowed) {
        func->setAttr("hls.scratch_arena_overflow", builder.getUnitAttr());
        func->setAttr("hls.scratch_arena_penalty",
                      builder.getI64IntegerAttr(overflowPenalty));
      }
      unsigned arenas = 0;
      for (unsigned color : colors)
        arenas = std::max(arenas, color + 1);
      for (unsigned arena = 0; arena < arenas; ++arena) {
        SmallVector<hls::BufferLikeInterface> group;
        for (auto [index, buffer] : llvm::enumerate(candidates))
          if (colors[index] == arena)
            group.push_back(buffer);
        if (failed(
                createScratchBuffer(func, module, builder, elementType, group)))
          return signalPassFailure();
      }
    }

    // Preserve `main` for the interface wrapper when the implementation itself
    // uses the pipeline's default top-function name.
    if (func.getName() == "main") {
      SymbolTable symbols(module);
      std::string name = "main_impl";
      for (unsigned suffix = 0; symbols.lookup(name); ++suffix)
        name = "main_impl_" + std::to_string(suffix);
      StringAttr implementationName = builder.getStringAttr(name);
      if (failed(symbols.rename(func, implementationName))) {
        func.emitError("failed to rename the AXI implementation");
        return signalPassFailure();
      }
    }

    // An unrelated `main` would collide with the fixed wrapper symbol.
    if (module.lookupSymbol("main")) {
      func.emitError("module already contains a symbol named 'main', which "
                     "the AXI wrapper function needs");
      return signalPassFailure();
    }

    builder.setInsertionPointAfter(func);
    auto mainFunc =
        func::FuncOp::create(builder, loc, "main", func.getFunctionType());
    setRuntimeAttr(mainFunc);
    auto mainBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToEnd(mainBlock);

    // Move all the arguments of the top function to the main function.
    for (auto [funcArg, mainArg] :
         llvm::zip(func.getArguments(), mainBlock->getArguments()))
      funcArg.replaceAllUsesWith(mainArg);
    func.front().eraseArguments([](BlockArgument arg) { return true; });

    // Move buffer arguments of the top function to the main function. Collect
    // all buffers to be converted to AXI interfaces into "buffers". At the same
    // time, scalar arguments go straight into "funcPorts".
    SmallVector<Value, 32> buffers;
    SmallVector<Value, 32> funcPorts;
    for (auto arg : mainBlock->getArguments())
      if (isa<MemRefType, StreamType>(arg.getType())) {
        buffers.push_back(arg);
      } else if (isa<ShapedType>(arg.getType())) {
        emitError(arg.getLoc(), "unsupported argument type");
        return signalPassFailure();
      } else {
        funcPorts.push_back(arg);
        arg.replaceAllUsesWith(
            func.front().addArgument(arg.getType(), arg.getLoc()));
      }

    // Move buffers allocated in the top function to the main function. After
    // the carving above the only ones left are the scratch arenas.
    for (auto buffer :
         llvm::make_early_inc_range(func.getOps<hls::BufferLikeInterface>())) {
      if (!isExtBuffer(buffer.getMemref()) || isa<ConstBufferOp>(*buffer))
        continue;
      buffer->remove();
      builder.insert(buffer);
      buffers.push_back(buffer.getMemref());
    }

    // Bundles are always typed by the underlying element, not the protocol:
    // the distinction between memory-mapped and streaming is expressed in
    // the emitted pragma, not in the IR type, because the verifier requires
    // a memref to be mapped to MM and a StreamType to be mapped to STREAM.
    // When `streamInterface` is set, the top function carries an attribute
    // that the emitter reads to choose `axis` over `m_axi` pragmas.
    auto getBundleType = [&](Value buffer) {
      if (auto memrefType = dyn_cast<MemRefType>(buffer.getType()))
        return BundleType::get(context, memrefType.getElementType(),
                               AxiKind::MM);
      if (auto streamType = dyn_cast<StreamType>(buffer.getType()))
        return BundleType::get(context, streamType.getElementType(),
                               AxiKind::STREAM);
      llvm_unreachable("invalid buffer type");
    };

    // Convert collected buffers to AXI ports and collect them in "funcPorts".
    // The list is now exactly the algorithm's I/O plus the carved scratch.
    //
    // A master is a platform resource, so one is spent only where the design
    // genuinely needs it. Full planes stay separate because sharing would
    // serialize whole-pass sweeps. Small read-only tables may share when the
    // same outlined process consumes them, which keeps the algorithm's public
    // I/O unchanged while avoiding a needless channel. Read-only is necessary
    // but not sufficient: Vitis 2022.2 rejects one m_axi bundle read by
    // multiple dataflow processes. Same element type and the same outlined
    // call are therefore required because a bundle carries one element type
    // and one process is the only safe sharing boundary.
    DenseMap<Operation *, DenseMap<Type, AxiBundleOp>> sharedInputBundles;
    auto getReadShareSite = [&](Value buffer) -> Operation * {
      if (streamInterface || buffer.getDefiningOp<BufferOp>())
        return nullptr;
      auto memrefType = dyn_cast<MemRefType>(buffer.getType());
      if (!memrefType || !memrefType.hasStaticShape())
        return nullptr;
      if (memrefType.getNumElements() > kSharedInputMaxElements)
        return nullptr;
      AccessRole role;
      DenseSet<Value> visited;
      classifyAccess(buffer, role, visited);
      if (!role.reads || role.writes)
        return nullptr;
      visited.clear();
      return singleReadCallSite(buffer, visited);
    };

    unsigned bundleIndex = 0;
    for (auto buffer : buffers) {
      // Public stream ports use a real StreamType all the way through the
      // implementation.  The runtime wrapper is intentionally kept in the
      // IR, but it carries the same stream ABI and is never emitted as C++.
      // Scratch arenas remain addressed memory: they are bidirectional and
      // cannot satisfy a FIFO's single-consumer contract.
      bool publicStream = streamInterface && !buffer.getDefiningOp<BufferOp>();
      SmallVector<int64_t> streamShape;
      if (publicStream) {
        auto memrefType = dyn_cast<MemRefType>(buffer.getType());
        AccessRole role;
        DenseSet<Value> roleVisited;
        classifyAccess(buffer, role, roleVisited);
        if (!memrefType || !isProvablySequentialStream(buffer, role)) {
          emitError(buffer.getLoc(),
                    "cannot use AXI4-Stream without one complete monotonic "
                    "row-major access sweep");
          return signalPassFailure();
        }
        streamShape.assign(memrefType.getShape().begin(),
                           memrefType.getShape().end());
        auto streamType = StreamType::get(context, memrefType.getElementType(),
                                          /*depth=*/2);
        DenseSet<Value> visited;
        buffer.setType(streamType);
        if (failed(convertSequentialStream(buffer, streamType, visited))) {
          emitError(buffer.getLoc(),
                    "failed to lower the sequential AXI4-Stream access chain");
          return signalPassFailure();
        }
        refreshFunctionTypes(module, context);
      }

      auto bundleType = getBundleType(buffer);
      builder.setInsertionPointToStart(&func.front());
      AxiBundleOp bundle;
      if (Operation *shareSite = getReadShareSite(buffer)) {
        auto &shared = sharedInputBundles[shareSite][bundleType];
        if (!shared)
          shared = AxiBundleOp::create(builder, loc, bundleType,
                                       "axi_" + std::to_string(bundleIndex++));
        bundle = shared;
      } else {
        bundle = AxiBundleOp::create(builder, loc, bundleType,
                                     "axi_" + std::to_string(bundleIndex++));
      }
      // Ports go after every bundle, so a reused bundle still dominates the
      // port that refers to it.
      builder.setInsertionPointAfter(bundle);

      auto axiType = AxiType::get(context, buffer.getType());
      auto axiPort =
          AxiPortOp::create(builder, loc, buffer.getType(), bundle,
                            func.front().addArgument(axiType, buffer.getLoc()));
      if (publicStream) {
        SmallVector<int64_t> shape(streamShape);
        axiPort->setAttr("stream_shape", builder.getI64ArrayAttr(shape));
      }
      if (auto scratch = buffer.getDefiningOp<BufferOp>()) {
        if (scratch->hasAttr("hls.scratch"))
          axiPort->setAttr("hls.scratch", builder.getUnitAttr());
        if (auto init = scratch.getInitValueAttr())
          axiPort->setAttr("init_value", init);
      }
      buffer.replaceUsesWithIf(
          axiPort, [&](OpOperand &use) { return use.getOwner() != axiPort; });

      builder.setInsertionPointToEnd(mainBlock);
      funcPorts.push_back(AxiPackOp::create(builder, loc, axiType, buffer));
    }

    // Update the top function and call.
    builder.setInsertionPointToEnd(mainBlock);
    auto call = func::CallOp::create(builder, func.getLoc(), func.getName(),
                                     func.getResultTypes(), funcPorts);
    func.setType(call.getCalleeType());
    func::ReturnOp::create(builder, loc, call.getResults());
    refreshFunctionTypes(module, context);
  }
};
} // namespace

std::unique_ptr<Pass> sar::createCreateAxiInterfacePass(std::string hlsTopFunc,
                                                        bool streamInterface,
                                                        unsigned maxArenas) {
  return std::make_unique<CreateAxiInterface>(hlsTopFunc, streamInterface,
                                              maxArenas);
}
