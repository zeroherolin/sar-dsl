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

#include "llvm/ADT/SetVector.h"

#include <array>

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
// The top function's signature is the design's contract with its caller, so it
// is fixed: the kernel's declared arguments/results and two scratch arenas per
// scalar element type in the lowered design. Buffers the compiler decides to
// keep off chip are internal, so they may not appear as ports of their own.
// Successive buffers are distributed between the two physical masters and
// carved at compile-time offsets. Placeholder arenas remain in the signature
// when their type has no spill, so placement decisions change extents but never
// the number or order of ports.
//
// Serialization within one scratch port is not a new cost: the carved
// buffers already take turns on the port's data bus, whether or not each
// of them had a port of its own.
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
/// before anything is rewritten so that a buffer reaching a use we cannot
/// redirect is left alone rather than half-carved.
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
/// to touch. Fails, without recording anything, on a use we cannot redirect or
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

/// Replaces the top function's internal DRAM buffers with slices of a single
/// scratch buffer, and returns that buffer for the caller to turn into the one
/// extra port.
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

  // The port exists even when nothing spills, so a caller binds the same
  // interfaces whatever the compiler decided this time round.
  auto scratchType = MemRefType::get(
      {std::max<int64_t>(1, elements)}, elementType, AffineMap(),
      MemoryKindAttr::get(func.getContext(), MemoryKind::DRAM));
  builder.setInsertionPointToStart(&func.front());
  Value scratch =
      BufferOp::create(builder, func.getLoc(), scratchType, /*depth=*/1,
                       slots.empty() ? TypedAttr() : initValue);

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

    // Fold internal DRAM buffers into stable typed arenas before ports are
    // created. Scan all non-constant buffers, not only current spills, so a
    // budget change cannot add or remove an arena type from the signature.
    llvm::SetVector<Type> scratchTypes;
    for (auto buffer : func.getOps<hls::BufferLikeInterface>())
      if (!isa<ConstBufferOp>(*buffer))
        if (Type elementType = buffer.getMemrefType().getElementType();
            elementType.isIntOrFloat())
          scratchTypes.insert(elementType);
    if (scratchTypes.empty())
      scratchTypes.insert(builder.getF64Type());
    DenseMap<Operation *, unsigned> operationOrder;
    unsigned nextOrder = 0;
    for (Operation &op : func.front())
      operationOrder[&op] = nextOrder++;
    auto interval = [&](hls::BufferLikeInterface buffer) {
      unsigned first = operationOrder.lookup(buffer.getOperation());
      unsigned last = first;
      for (OpOperand &use : buffer.getMemref().getUses()) {
        Operation *owner = use.getOwner();
        while (owner->getParentOp() != func && owner->getParentOp())
          owner = owner->getParentOp();
        if (owner->getParentOp() == func) {
          unsigned position = operationOrder.lookup(owner);
          first = std::min(first, position);
          last = std::max(last, position);
        }
      }
      return std::pair(first, last);
    };
    for (Type elementType : scratchTypes) {
      using UserDirs = DenseMap<Operation *, unsigned>; // 1=read, 2=written
      std::array<SmallVector<hls::BufferLikeInterface>, 2> banks;
      std::array<SmallVector<std::pair<unsigned, unsigned>>, 2> lifetimes;
      std::array<SmallVector<UserDirs>, 2> bankUsers;
      std::array<uint64_t, 2> payload = {0, 0};
      for (auto buffer : func.getOps<hls::BufferLikeInterface>()) {
        if (isExtBuffer(buffer.getMemref()) && !isa<ConstBufferOp>(*buffer) &&
            buffer.getMemrefType().getElementType() == elementType) {
          auto live = interval(buffer);
          // How each top-level operation touches this buffer. Two buffers
          // under one loop contend for the master every iteration, and the
          // direction decides how badly: a read stream and a write stream
          // interlock their requests and responses (II in the tens), while
          // two same-direction streams merely halve the issue rate. The
          // split-complex sweeps this backend emits read re/im pairs and
          // write re/im pairs in one loop, so the coloring keeps opposing
          // directions apart first and splits same-direction pairs second.
          UserDirs users;
          for (OpOperand &use : buffer.getMemref().getUses()) {
            Operation *owner = use.getOwner();
            Operation *leaf = owner;
            while (owner->getParentOp() != func && owner->getParentOp())
              owner = owner->getParentOp();
            if (owner->getParentOp() != func)
              continue;
            unsigned dir = isWritten(use) ? 2 : 1;
            // A call reads and writes through its own body; be pessimistic
            // only when the operand is genuinely written there.
            (void)leaf;
            users[owner] |= dir;
          }
          auto cost = [&](unsigned bank) {
            uint64_t total = 0;
            for (const auto &other : bankUsers[bank])
              for (auto [op, dir] : users) {
                auto it = other.find(op);
                if (it == other.end())
                  continue;
                unsigned both = dir | it->second;
                total += both == 3 ? 16 : 1;
              }
            return total;
          };
          auto conflicts = [&](unsigned bank) {
            return llvm::count_if(lifetimes[bank], [&](auto other) {
              return !(live.second < other.first || other.second < live.first);
            });
          };
          unsigned bank = 0;
          auto leftCost = cost(0), rightCost = cost(1);
          auto leftConflicts = conflicts(0), rightConflicts = conflicts(1);
          if (rightCost < leftCost)
            bank = 1;
          else if (rightCost == leftCost &&
                   (rightConflicts < leftConflicts ||
                    (rightConflicts == leftConflicts &&
                     payload[1] < payload[0])))
            bank = 1;
          banks[bank].push_back(buffer);
          lifetimes[bank].push_back(live);
          bankUsers[bank].push_back(std::move(users));
          payload[bank] += buffer.getMemrefType().getNumElements();
        }
      }
      for (auto &candidates : banks)
        if (failed(createScratchBuffer(func, module, builder, elementType,
                                       candidates)))
          return signalPassFailure();
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
    // time, we also directly collect all scalar arguments into "funcPorts".
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
    // Every port gets its own AXI master bundle so concurrent plane accesses
    // are not serialized by bundle arbitration. The system interconnect may
    // still concentrate those masters onto fewer memory controllers.
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
