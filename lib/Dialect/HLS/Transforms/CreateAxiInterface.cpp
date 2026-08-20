//===- CreateAxiInterface.cpp - create axi interface ----------------------===//
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
#include <numeric>

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
// A design's port list is its algorithm's I/O: the kernel's declared arguments
// and results, and nothing the compiler added for its own convenience. Ports
// are the contract a board integrator has to wire to physical memory channels,
// so a compiler that mints extra masters spends a platform resource to buy
// itself scheduling freedom.
//
// Buffers the compiler decides to keep off chip are internal. They are carved
// at compile-time offsets out of scratch allocations, one per element type
// and per group of buffers that can share a pointer -- an arena is `float *`
// or `double *`, and one pointer cannot be both. A type with no spill
// contributes no port at all.
//
// Buffers a single node reads and writes may not share an arena. Merging them
// would hand that node one pointer it both loads from and stores to, and HLS
// must then assume every load may alias a store still in flight: the node's
// bus requests serialize behind their own responses and its initiation
// interval collapses. Splitting on that conflict is what a hand-written design
// does when it ping-pongs between a scratch plane and its output plane, and it
// is the smallest split that keeps the arithmetic pipelined.
//
// Sharing within one arena costs nothing extra: buffers that no node pairs up
// already take turns on that port's data bus, whether or not each of them had
// a port of its own.
//===----------------------------------------------------------------------===//

namespace {

/// Bytes a carved buffer starts on, so no slot straddles an AXI beat.
constexpr uint64_t kScratchAlignBytes = 64;

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
/// keeps the conflict graph below conservative.
static void classifyAccess(Value value, AccessRole &role,
                           DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return;

  for (auto &use : value.getUses()) {
    auto *owner = use.getOwner();
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

/// Arenas one element type may claim. A split-complex chain ping-pongs two
/// planes -- real and imaginary -- and each needs a side to read and a side
/// to write, so four pointers cover the aliasing a deliverable design
/// actually has. A denser conflict graph than that comes from chain depth
/// rather than data layout, and buying it more ports would let a longer
/// kernel spend memory channels its algorithm never asked for. Past the cap
/// the extra buffers share, and the traffic they serialize is traffic one
/// pointer would have serialized anyway.
constexpr unsigned kMaxScratchArenasPerType = 4;

/// Partitions `candidates` into arenas, returning each one's arena index.
///
/// Two buffers conflict when one node reads one and writes the other: giving
/// them the same pointer would make that node's loads and stores alias. The
/// arena count is that conflict graph's coloring, so a chain alternating
/// between a working plane and a result plane gets two arenas and buffers no
/// node ever pairs up keep sharing one.
///
/// Colors most-constrained-first. The order matters for the count, not just
/// the assignment: taking buffers in declaration order makes a graph two
/// arenas cover ask for three, because a buffer conflicting with everything
/// claims a color before the pair it separates has been placed.
static SmallVector<unsigned>
colorScratchBanks(ArrayRef<hls::BufferLikeInterface> candidates) {
  unsigned count = candidates.size();
  SmallVector<unsigned> colors(count, 0);
  if (count < 2)
    return colors;

  llvm::MapVector<Operation *, SmallVector<unsigned>> readers, writers;
  for (auto [index, entry] : llvm::enumerate(candidates)) {
    hls::BufferLikeInterface buffer = entry;
    for (auto &use : buffer.getMemref().getUses()) {
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

  SmallVector<unsigned> order(count);
  std::iota(order.begin(), order.end(), 0u);
  llvm::stable_sort(order, [&](unsigned lhs, unsigned rhs) {
    return conflicts[lhs].size() > conflicts[rhs].size();
  });

  SmallVector<bool> assigned(count, false);
  for (unsigned index : order) {
    SmallVector<bool> taken(kMaxScratchArenasPerType, false);
    for (unsigned neighbor : conflicts[index])
      if (assigned[neighbor])
        taken[colors[neighbor]] = true;
    unsigned color = 0;
    while (color < kMaxScratchArenasPerType && taken[color])
      ++color;
    // Every arena already conflicts with this buffer. Share the first one
    // rather than mint a port beyond the cap.
    colors[index] = color < kMaxScratchArenasPerType ? color : 0;
    assigned[index] = true;
  }
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

  // A constant buffer holds data the design needs on entry, so it is not
  // scratch: it keeps its own storage and its initializer.
  unsigned widest = elementType.getIntOrFloatBitWidth();

  // A buffer may ask to start at a value. One allocation carries one such
  // value, so the scratch adopts it -- initializing a buffer that asked for
  // nothing is harmless, contradicting one that asked for something is not,
  // so a second, different request keeps its own storage.
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
  CreateAxiInterface(std::string hlsTopFunc, bool argStreamInterface) {
    topFunc = hlsTopFunc;
    streamInterface = argStreamInterface;
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module);
    auto context = builder.getContext();
    auto loc = builder.getUnknownLoc();

    // Get the top function of the module.
    auto func = getTopFunc(module, topFunc);
    if (!func) {
      emitError(module.getLoc(), "fail to find the top function");
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
      auto colors = colorScratchBanks(candidates);
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
    // the carving above the only one left is the scratch.
    for (auto buffer :
         llvm::make_early_inc_range(func.getOps<hls::BufferLikeInterface>())) {
      if (!isExtBuffer(buffer.getMemref()) || isa<ConstBufferOp>(*buffer))
        continue;
      buffer->remove();
      builder.insert(buffer);
      buffers.push_back(buffer.getMemref());
    }

    // A helper to get AXI bundle type from a buffer.
    //
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
    // The list is now exactly the algorithm's I/O plus the carved scratch, and
    // each of those gets its own master: a port that exists because the
    // algorithm reads or writes it is one an integrator has to wire anyway,
    // and sharing a bundle between two of them would serialize their bus
    // requests. The system interconnect still concentrates these masters onto
    // however many memory controllers the board has.
    unsigned bundleIndex = 0;
    for (auto buffer : buffers) {
      auto bundleType = getBundleType(buffer);
      builder.setInsertionPointToStart(&func.front());
      auto bundle = AxiBundleOp::create(builder, loc, bundleType,
                                        "axi_" + std::to_string(bundleIndex++));
      // Ports go after every bundle, so a reused bundle still dominates the
      // port that refers to it.
      builder.setInsertionPointAfter(bundle);

      auto axiType = AxiType::get(context, buffer.getType());
      auto axiPort =
          AxiPortOp::create(builder, loc, buffer.getType(), bundle,
                            func.front().addArgument(axiType, buffer.getLoc()));
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
  }
};
} // namespace

std::unique_ptr<Pass> sar::createCreateAxiInterfacePass(std::string hlsTopFunc,
                                                        bool streamInterface) {
  return std::make_unique<CreateAxiInterface>(hlsTopFunc, streamInterface);
}
