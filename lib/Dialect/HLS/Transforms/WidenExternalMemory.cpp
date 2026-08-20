//===- WidenExternalMemory.cpp - pack aligned external accesses ----------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// An AXI master moves a full bus word per beat, so a loop reading one scalar
// per iteration spends a beat on a fraction of a transfer. Where a whole
// group of accesses to one port is provably contiguous and aligned, this pass
// retypes the port to `vector<factor x T>` and rewrites the loop to move one
// word per iteration, extracting the lanes in the body.
//
// The rewrite changes the port's element type, so it is only applied to a
// group whose accesses are all proven: one unrecognized access to a port
// disqualifies the whole port rather than producing a mixed representation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Support/HLSHints.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_WIDENEXTERNALMEMORY
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::sar;
using namespace mlir::sar::hls;

namespace {

struct LinearForm {
  SmallVector<int64_t> dims;
  SmallVector<int64_t> symbols;
  int64_t constant = 0;
  bool valid = true;
};

static bool addChecked(int64_t lhs, int64_t rhs, int64_t &result) {
  return !__builtin_add_overflow(lhs, rhs, &result);
}

static bool mulChecked(int64_t lhs, int64_t rhs, int64_t &result) {
  return !__builtin_mul_overflow(lhs, rhs, &result);
}

static LinearForm linearize(AffineExpr expr, unsigned numDims,
                            unsigned numSymbols) {
  LinearForm result{SmallVector<int64_t>(numDims, 0),
                    SmallVector<int64_t>(numSymbols, 0), 0, true};
  if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
    result.dims[dim.getPosition()] = 1;
    return result;
  }
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expr)) {
    result.symbols[symbol.getPosition()] = 1;
    return result;
  }
  if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
    result.constant = constant.getValue();
    return result;
  }

  auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binary) {
    result.valid = false;
    return result;
  }
  if (expr.getKind() == AffineExprKind::Add) {
    auto lhs = linearize(binary.getLHS(), numDims, numSymbols);
    auto rhs = linearize(binary.getRHS(), numDims, numSymbols);
    if (!lhs.valid || !rhs.valid)
      result.valid = false;
    for (auto [out, l, r] : llvm::zip(result.dims, lhs.dims, rhs.dims))
      if (!addChecked(l, r, out))
        result.valid = false;
    for (auto [out, l, r] : llvm::zip(result.symbols, lhs.symbols, rhs.symbols))
      if (!addChecked(l, r, out))
        result.valid = false;
    if (!addChecked(lhs.constant, rhs.constant, result.constant))
      result.valid = false;
    return result;
  }
  if (expr.getKind() != AffineExprKind::Mul) {
    result.valid = false;
    return result;
  }

  auto lhsConstant = dyn_cast<AffineConstantExpr>(binary.getLHS());
  auto rhsConstant = dyn_cast<AffineConstantExpr>(binary.getRHS());
  if (!lhsConstant && !rhsConstant) {
    result.valid = false;
    return result;
  }
  int64_t scale = lhsConstant ? lhsConstant.getValue() : rhsConstant.getValue();
  auto value = linearize(lhsConstant ? binary.getRHS() : binary.getLHS(),
                         numDims, numSymbols);
  if (!value.valid)
    result.valid = false;
  for (auto [out, coefficient] : llvm::zip(result.dims, value.dims))
    if (!mulChecked(coefficient, scale, out))
      result.valid = false;
  for (auto [out, coefficient] : llvm::zip(result.symbols, value.symbols))
    if (!mulChecked(coefficient, scale, out))
      result.valid = false;
  if (!mulChecked(value.constant, scale, result.constant))
    result.valid = false;
  return result;
}

/// One affine load or store, with its map fully composed so the index
/// arithmetic can be inspected without walking back through the operands.
struct Access {
  Operation *operation;
  Value memref;
  AffineMap map;
  SmallVector<Value> operands;
  bool isLoad;
};

static FailureOr<Access> composeAccess(Operation *operation) {
  AffineMap map;
  SmallVector<Value> operands;
  Value memref;
  bool isLoad = false;
  if (auto load = dyn_cast<AffineLoadOp>(operation)) {
    map = load.getAffineMap();
    operands.assign(load.getMapOperands().begin(), load.getMapOperands().end());
    memref = load.getMemRef();
    isLoad = true;
  } else if (auto store = dyn_cast<AffineStoreOp>(operation)) {
    map = store.getAffineMap();
    operands.assign(store.getMapOperands().begin(),
                    store.getMapOperands().end());
    memref = store.getMemRef();
  } else {
    return failure();
  }
  fullyComposeAffineMapAndOperands(&map, &operands);
  canonicalizeMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  return Access{operation, memref, map, std::move(operands), isLoad};
}

/// Whether `access` sweeps `factor` consecutive elements per iteration of
/// `loop`, starting on a `factor`-aligned boundary. That is the condition
/// under which `factor` scalar accesses are exactly one packed word.
static bool isAlignedContiguous(const Access &access, AffineForOp loop,
                                unsigned factor) {
  auto type = dyn_cast<MemRefType>(access.memref.getType());
  if (!type || access.map.getNumResults() != (unsigned)type.getRank())
    return false;

  auto laneCoefficient = [&](const LinearForm &form) {
    int64_t coefficient = 0;
    for (auto [index, value] : llvm::enumerate(access.operands)) {
      if (value != loop.getInductionVar())
        continue;
      int64_t term = index < access.map.getNumDims()
                         ? form.dims[index]
                         : form.symbols[index - access.map.getNumDims()];
      if (!addChecked(coefficient, term, coefficient))
        return std::optional<int64_t>();
    }
    return std::optional<int64_t>(coefficient);
  };

  for (auto [index, expr] : llvm::enumerate(access.map.getResults())) {
    auto form =
        linearize(expr, access.map.getNumDims(), access.map.getNumSymbols());
    if (!form.valid)
      return false;
    auto lane = laneCoefficient(form);
    if (!lane ||
        (index + 1 == access.map.getNumResults() ? *lane != 1 : *lane != 0))
      return false;
    if (index + 1 != access.map.getNumResults())
      continue;

    for (auto [operandIndex, coefficient] : llvm::enumerate(form.dims))
      if (access.operands[operandIndex] != loop.getInductionVar() &&
          coefficient % factor != 0)
        return false;
    for (auto [symbolIndex, coefficient] : llvm::enumerate(form.symbols))
      if (access.operands[access.map.getNumDims() + symbolIndex] !=
              loop.getInductionVar() &&
          coefficient % factor != 0)
        return false;
    if (form.constant % factor != 0)
      return false;
  }
  return true;
}

static AffineMap getWordMap(const Access &access, unsigned factor) {
  SmallVector<AffineExpr> results(access.map.getResults());
  results.back() = results.back().floorDiv(factor);
  return simplifyAffineMap(AffineMap::get(access.map.getNumDims(),
                                          access.map.getNumSymbols(), results,
                                          access.map.getContext()));
}

/// One port and everything reached from it: the values that alias it, the
/// accesses to be repacked, and the types before and after the rewrite. Ports
/// whose reachable sets intersect are merged, since they must agree on a
/// representation.
struct Candidate {
  SmallVector<AxiPortOp> roots;
  DenseSet<Value> values;
  SmallVector<Access> accesses;
  SmallVector<memref::LoadOp> dynamicLoads;
  MemRefType oldType;
  MemRefType newType;
};

static bool collectReachable(Value value, ModuleOp module,
                             DenseSet<Value> &values) {
  if (!values.insert(value).second)
    return true;
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (isa<AffineLoadOp>(owner))
      continue;
    if (auto load = dyn_cast<memref::LoadOp>(owner)) {
      if (use.getOperandNumber() == 0)
        continue;
      return false;
    }
    if (auto store = dyn_cast<AffineStoreOp>(owner)) {
      if (use.getOperandNumber() == 1)
        continue;
      return false;
    }
    auto call = dyn_cast<func::CallOp>(owner);
    if (!call)
      return false;
    auto callee = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
    if (!callee || callee.isExternal() ||
        use.getOperandNumber() >= callee.getNumArguments())
      return false;
    if (!collectReachable(callee.getArgument(use.getOperandNumber()), module,
                          values))
      return false;
  }
  return true;
}

static bool intersects(const DenseSet<Value> &lhs, const DenseSet<Value> &rhs) {
  return llvm::any_of(lhs, [&](Value value) { return rhs.contains(value); });
}

static void refreshFunctionTypes(ModuleOp module) {
  Builder builder(module.getContext());
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;
    auto returnOp = dyn_cast<func::ReturnOp>(func.front().getTerminator());
    if (!returnOp)
      continue;
    func.setType(builder.getFunctionType(func.front().getArgumentTypes(),
                                         returnOp.getOperandTypes()));
  }
}

static void vectorizeDynamicLoad(memref::LoadOp load, unsigned factor) {
  OpBuilder builder(load);
  Location loc = load.getLoc();
  SmallVector<Value> indices(load.getIndices());
  Value scalarIndex = indices.back();
  Value divisor = arith::ConstantIndexOp::create(builder, loc, factor);
  Value wordIndex = arith::DivUIOp::create(builder, loc, scalarIndex, divisor);
  Value lane = arith::RemUIOp::create(builder, loc, scalarIndex, divisor);
  indices.back() = wordIndex;
  Value word = memref::LoadOp::create(builder, loc, load.getMemRef(), indices);
  Value element = vector::ExtractOp::create(builder, loc, word, lane);
  load.replaceAllUsesWith(element);
  load.erase();
}

static Value laneIndex(OpBuilder &builder, Location loc, Value word,
                       unsigned factor, unsigned lane) {
  AffineExpr dim = builder.getAffineDimExpr(0);
  auto map = AffineMap::get(1, 0, dim * factor + lane);
  return AffineApplyOp::create(builder, loc, map, word);
}

static SmallVector<Value> mappedOperands(const Access &access,
                                         IRMapping &mapping) {
  SmallVector<Value> operands;
  operands.reserve(access.operands.size());
  for (Value operand : access.operands)
    operands.push_back(mapping.lookupOrDefault(operand));
  return operands;
}

static LogicalResult
vectorizeLoop(AffineForOp loop, unsigned factor,
              const DenseSet<Operation *> &selectedAccesses) {
  if (!loop.hasConstantBounds() || loop.getConstantLowerBound() != 0 ||
      loop.getStep() != 1)
    return failure();
  int64_t tripCount = loop.getConstantUpperBound();
  if (tripCount <= 0 || tripCount % factor != 0)
    return failure();
  for (Operation &operation : loop.getBody()->without_terminator())
    if (operation.getNumRegions() != 0)
      return failure();

  SmallVector<Access> loads;
  SmallVector<Access> stores;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (!selectedAccesses.contains(&operation))
      continue;
    auto access = composeAccess(&operation);
    if (failed(access) || !isAlignedContiguous(*access, loop, factor))
      return failure();
    (access->isLoad ? loads : stores).push_back(std::move(*access));
  }
  if (loads.empty() && stores.empty())
    return failure();

  OpBuilder builder(loop);
  auto newLoop =
      AffineForOp::create(builder, loop.getLoc(), 0, tripCount / factor);
  for (NamedAttribute attribute : loop->getAttrs()) {
    StringRef name = attribute.getName().getValue();
    if (name == "lowerBoundMap" || name == "upperBoundMap" || name == "step" ||
        name == "operandSegmentSizes")
      continue;
    newLoop->setAttr(attribute.getName(), attribute.getValue());
  }
  newLoop->removeAttr(kUnrollFactorAttr);
  builder.setInsertionPointToStart(newLoop.getBody());

  Value firstLane =
      laneIndex(builder, loop.getLoc(), newLoop.getInductionVar(), factor, 0);
  IRMapping firstMapping;
  firstMapping.map(loop.getInductionVar(), firstLane);

  DenseMap<Operation *, Value> vectors;
  for (const Access &load : loads) {
    auto value = AffineLoadOp::create(builder, load.operation->getLoc(),
                                      load.memref, getWordMap(load, factor),
                                      mappedOperands(load, firstMapping));
    vectors[load.operation] = value;
  }

  DenseMap<Operation *, SmallVector<Value>> storeLanes;
  for (unsigned lane = 0; lane < factor; ++lane) {
    IRMapping mapping;
    Value scalarIndex = laneIndex(builder, loop.getLoc(),
                                  newLoop.getInductionVar(), factor, lane);
    mapping.map(loop.getInductionVar(), scalarIndex);
    for (const Access &load : loads)
      mapping.map(load.operation->getResult(0),
                  vector::ExtractOp::create(builder, load.operation->getLoc(),
                                            vectors.lookup(load.operation),
                                            lane));

    for (Operation &operation : loop.getBody()->without_terminator()) {
      if (selectedAccesses.contains(&operation)) {
        if (auto store = dyn_cast<AffineStoreOp>(operation))
          storeLanes[&operation].push_back(
              mapping.lookupOrDefault(store.getValueToStore()));
        continue;
      }
      builder.clone(operation, mapping);
    }
  }

  for (const Access &store : stores) {
    auto values = storeLanes.lookup(store.operation);
    if (values.size() != factor)
      return failure();
    auto vectorType = cast<MemRefType>(store.memref.getType()).getElementType();
    Value packed = vector::FromElementsOp::create(
        builder, store.operation->getLoc(), vectorType, values);
    AffineStoreOp::create(builder, store.operation->getLoc(), packed,
                          store.memref, getWordMap(store, factor),
                          mappedOperands(store, firstMapping));
  }

  loop.erase();
  return success();
}

struct WidenExternalMemory
    : public sar::impl::WidenExternalMemoryBase<WidenExternalMemory> {
  WidenExternalMemory() = default;
  WidenExternalMemory(unsigned argBusBits, unsigned argMaxLanes,
                      unsigned argMinElements) {
    busBits = argBusBits;
    maxLanes = argMaxLanes;
    minElements = argMinElements;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<AxiPortOp> ports;
    module.walk([&](AxiPortOp port) {
      if (isa<MemRefType>(port.getElement().getType()))
        ports.push_back(port);
    });
    if (ports.empty())
      return;

    unsigned widestScalar = 0;
    for (AxiPortOp port : ports) {
      auto type = cast<MemRefType>(port.getElement().getType());
      if (type.getElementType().isIntOrFloat())
        widestScalar = std::max(widestScalar, type.getElementTypeBitWidth());
    }
    if (!widestScalar || busBits < widestScalar)
      return;
    if (maxLanes == 0)
      return;
    unsigned factor = busBits / widestScalar;
    factor = 1u << llvm::Log2_32(factor);
    factor = std::min(factor, maxLanes.getValue());
    factor = 1u << llvm::Log2_32(factor);
    if (factor <= 1)
      return;

    SmallVector<Candidate, 4> candidates;
    for (AxiPortOp port : ports) {
      Candidate candidate;
      candidate.roots.push_back(port);
      if (!collectReachable(port.getElement(), module, candidate.values))
        continue;
      candidates.push_back(std::move(candidate));
    }

    for (unsigned i = 0; i < candidates.size(); ++i)
      for (unsigned j = i + 1; j < candidates.size();) {
        if (!intersects(candidates[i].values, candidates[j].values)) {
          ++j;
          continue;
        }
        candidates[i].roots.append(candidates[j].roots);
        candidates[i].values.insert(candidates[j].values.begin(),
                                    candidates[j].values.end());
        candidates.erase(candidates.begin() + j);
      }

    SmallVector<Candidate, 4> selected;
    DenseSet<Operation *> selectedAccesses;
    DenseSet<Operation *> loops;
    for (Candidate &candidate : candidates) {
      auto type =
          dyn_cast<MemRefType>(candidate.roots.front().getElement().getType());
      if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
          type.getRank() < 2 || !type.getElementType().isIntOrFloat() ||
          type.getNumElements() < minElements ||
          type.getShape().back() % factor != 0)
        continue;
      bool compatible = llvm::all_of(candidate.values, [&](Value value) {
        return value.getType() == type;
      });
      if (!compatible)
        continue;

      bool valid = true;
      for (Value value : candidate.values)
        for (OpOperand &use : value.getUses()) {
          Operation *owner = use.getOwner();
          if (isa<func::CallOp>(owner))
            continue;
          if (auto load = dyn_cast<memref::LoadOp>(owner)) {
            if (use.getOperandNumber() != 0) {
              valid = false;
              break;
            }
            candidate.dynamicLoads.push_back(load);
            continue;
          }
          auto access = composeAccess(owner);
          auto loop = owner->getParentOfType<AffineForOp>();
          if (failed(access) || !loop || owner->getBlock() != loop.getBody() ||
              !isAlignedContiguous(*access, loop, factor)) {
            valid = false;
            break;
          }
          candidate.accesses.push_back(std::move(*access));
        }
      if (!valid ||
          (candidate.accesses.empty() && candidate.dynamicLoads.empty()))
        continue;
      // Read-only ports can change representation without introducing
      // read-modify-write traffic or making the public result ABI depend on
      // buffer privatization and placement. Write packing remains safe for
      // dedicated movers, but needs a stable adapter before it is used on a
      // general kernel result or scratch arena.
      if (llvm::any_of(candidate.accesses,
                       [](const Access &access) { return !access.isLoad; }))
        continue;

      auto shape = SmallVector<int64_t>(type.getShape());
      shape.back() /= factor;
      auto vectorType =
          VectorType::get({(int64_t)factor}, type.getElementType());
      candidate.oldType = type;
      candidate.newType = MemRefType::get(shape, vectorType, AffineMap(),
                                          type.getMemorySpace());
      for (const Access &access : candidate.accesses) {
        selectedAccesses.insert(access.operation);
        loops.insert(access.operation->getParentOfType<AffineForOp>());
      }
      selected.push_back(std::move(candidate));
    }
    if (selected.empty())
      return;

    for (Candidate &candidate : selected)
      for (Value value : candidate.values)
        value.setType(candidate.newType);

    for (Candidate &candidate : selected)
      for (memref::LoadOp load : candidate.dynamicLoads)
        vectorizeDynamicLoad(load, factor);

    for (Candidate &candidate : selected)
      for (AxiPortOp port : candidate.roots) {
        auto axiType = AxiType::get(module.getContext(), candidate.newType);
        port.getElement().setType(candidate.newType);
        port.getAxi().setType(axiType);
        port->setAttr("vector_factor",
                      IntegerAttr::get(
                          IntegerType::get(module.getContext(), 32), factor));
        auto bundle = port.getBundle().getDefiningOp<AxiBundleOp>();
        auto oldBundle = cast<BundleType>(bundle.getBundle().getType());
        bundle.getBundle().setType(BundleType::get(
            module.getContext(), candidate.newType.getElementType(),
            oldBundle.getKind()));

        auto top = port->getParentOfType<func::FuncOp>();
        unsigned argument = cast<BlockArgument>(port.getAxi()).getArgNumber();
        module.walk([&](func::CallOp call) {
          if (call.getCallee() != top.getName() ||
              argument >= call.getNumOperands())
            return;
          Value operand = call.getOperand(argument);
          operand.setType(axiType);
          if (auto pack = operand.getDefiningOp<AxiPackOp>()) {
            pack.getElement().setType(candidate.newType);
            pack.getAxi().setType(axiType);
          }
        });
      }

    SmallVector<AffineForOp> orderedLoops;
    for (Operation *operation : loops)
      orderedLoops.push_back(cast<AffineForOp>(operation));
    llvm::stable_sort(orderedLoops, [](AffineForOp lhs, AffineForOp rhs) {
      return lhs->isProperAncestor(rhs);
    });
    for (AffineForOp loop : llvm::reverse(orderedLoops))
      if (failed(vectorizeLoop(loop, factor, selectedAccesses))) {
        loop.emitError("failed to pack a proven external access group");
        return signalPassFailure();
      }

    refreshFunctionTypes(module);
  }
};

} // namespace

std::unique_ptr<Pass> sar::createWidenExternalMemoryPass(unsigned busBits,
                                                         unsigned maxLanes,
                                                         unsigned minElements) {
  return std::make_unique<WidenExternalMemory>(busBits, maxLanes, minElements);
}
