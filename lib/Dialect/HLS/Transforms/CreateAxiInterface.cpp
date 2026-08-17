//===- CreateAxiInterface.cpp - Shape the top function's AXI ports --------===//
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
// is fixed: the kernel's declared arguments, its declared results, and one
// scratch pointer. Buffers the compiler decides to keep off chip are internal,
// so they may not appear as ports of their own -- a host would otherwise have
// to bind a different number of interfaces every time a budget, a scene size
// or a fusion decision moved a buffer across the line. They are carved out of
// the single scratch allocation instead, at offsets fixed at compile time; the
// size the caller must allocate is recorded on the top function
// (`scratch_bytes`) and is the extent of the scratch port. The port is emitted
// whether or not anything currently spills, so the signature follows the
// algorithm alone.
//
// Serialization is not a new cost: ports of one element type already share a
// single AXI bundle, and a bundle serializes its accesses whether or not each
// buffer has a port of its own.
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

static uint64_t elementBytes(MemRefType type) {
  return (type.getElementTypeBitWidth() + 7) / 8;
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
static Value createScratchBuffer(func::FuncOp func, ModuleOp module,
                                 OpBuilder &builder) {
  DenseMap<StringRef, unsigned> callerCounts;
  module.walk([&](func::CallOp call) { ++callerCounts[call.getCallee()]; });

  // A constant buffer holds data the design needs on entry, so it is not
  // scratch: it keeps its own storage and its initializer.
  SmallVector<hls::BufferLikeInterface> candidates;
  for (auto buffer : func.getOps<hls::BufferLikeInterface>())
    if (isExtBuffer(buffer.getMemref()) && !isa<ConstBufferOp>(*buffer))
      candidates.push_back(buffer);

  // One flat array cannot hold two element widths -- the emitted C++ has no
  // way to reinterpret it -- so the scratch takes the widest element among
  // the buffers to be carved. Whatever else the design allocates decides the
  // type when nothing spills, so the port's type still follows the algorithm.
  Type elementType;
  unsigned widest = 0;
  auto widen = [&](MemRefType type) {
    if (type.getElementTypeBitWidth() > widest) {
      widest = type.getElementTypeBitWidth();
      elementType = type.getElementType();
    }
  };
  if (candidates.empty()) {
    for (auto buffer : func.getOps<hls::BufferLikeInterface>())
      widen(buffer.getMemrefType());
    for (auto arg : func.getArguments())
      if (auto type = dyn_cast<MemRefType>(arg.getType()))
        widen(type);
  } else {
    for (auto buffer : candidates)
      widen(buffer.getMemrefType());
  }
  if (!elementType)
    elementType = builder.getF64Type();

  // A buffer may ask to start at a value. One allocation carries one such
  // value, so the scratch adopts it -- initializing a buffer that asked for
  // nothing is harmless, contradicting one that asked for something is not,
  // so a second, different request keeps its own storage.
  TypedAttr initValue;
  for (auto buffer : candidates)
    if (auto buf = dyn_cast<BufferOp>(*buffer))
      if (auto init = buf.getInitValueAttr()) {
        if (!initValue)
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
    if (!isCarvable(type) || type.getElementType() != elementType ||
        (init && init != initValue) ||
        failed(collectScratchUses(buffer.getMemref(), module, callerCounts,
                                  uses, visited))) {
      skipped.push_back(buffer);
      continue;
    }

    slots.push_back({buffer, elements});
    slotUses.push_back(std::move(uses));
    elements += ((int64_t)type.getNumElements() + align - 1) / align * align;
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

  // Anything left keeps a port of its own, which is exactly the behaviour the
  // fixed signature exists to prevent -- so say so rather than quietly
  // growing the interface.
  for (auto buffer : skipped)
    buffer->emitWarning("internal DRAM buffer cannot be carved into the "
                        "scratch allocation and becomes a top-level port; "
                        "the design's port count no longer follows its I/O");

  func->setAttr(
      "scratch_bytes",
      builder.getI64IntegerAttr((int64_t)elements * elementBytes(scratchType)));
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

    // Fold the internal DRAM buffers into one scratch allocation before any
    // port is made, so exactly one of them reaches the signature.
    createScratchBuffer(func, module, builder);

    // The wrapper takes the fixed name "main"; a module that already claims
    // that symbol -- including a kernel named "main" itself -- would end up
    // with two functions under one name.
    if (module.lookupSymbol("main")) {
      func.emitError("module already contains a symbol named 'main', which "
                     "the AXI wrapper function needs; rename the kernel");
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
    // Ports of the same element type share one AXI master: a design then
    // presents a handful of interfaces, which is what a device can wire up
    // and a host can bind.
    unsigned bundleIndex = 0;
    llvm::DenseMap<BundleType, AxiBundleOp> bundles;
    for (auto buffer : buffers) {
      auto bundleType = getBundleType(buffer);
      auto &bundle = bundles[bundleType];
      if (!bundle) {
        builder.setInsertionPointToStart(&func.front());
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
