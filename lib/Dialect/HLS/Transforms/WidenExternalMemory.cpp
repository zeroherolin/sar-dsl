//===- WidenExternalMemory.cpp - pack aligned external accesses -----------===//
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

#include <iterator>
#include <numeric>

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

/// Most scalar fallback adapters a compiler-owned scratch arena may need
/// before packing it is declined and the arena stays scalar.
constexpr unsigned kMaxScratchScalarFallbacks = 256;

/// An index expression split into a linear part and the sub-expressions no
/// linear part can describe. `floordiv`, `ceildiv`, `mod` and products of two
/// non-constants become opaque terms, each paired with the constant scaling
/// it. Index arithmetic that wraps a coordinate -- an fftshift, a modular
/// stride -- is linear in the lane direction even though the wrap itself is
/// not, and keeping the wrap as an opaque term is what lets such an access
/// still be recognized as a contiguous sweep.
struct LinearForm {
  SmallVector<int64_t> dims;
  SmallVector<int64_t> symbols;
  int64_t constant = 0;
  SmallVector<std::pair<AffineExpr, int64_t>> opaque;
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
                    SmallVector<int64_t>(numSymbols, 0),
                    0,
                    {},
                    true};
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
    result.opaque = std::move(lhs.opaque);
    llvm::append_range(result.opaque, rhs.opaque);
    return result;
  }

  auto lhsConstant = expr.getKind() == AffineExprKind::Mul
                         ? dyn_cast<AffineConstantExpr>(binary.getLHS())
                         : nullptr;
  auto rhsConstant = expr.getKind() == AffineExprKind::Mul
                         ? dyn_cast<AffineConstantExpr>(binary.getRHS())
                         : nullptr;
  if (!lhsConstant && !rhsConstant) {
    // Not linear: keep the whole term, so callers can still ask whether it
    // varies with a given loop and whether its scale is a multiple of theirs.
    result.opaque.emplace_back(expr, 1);
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
  for (auto &[term, coefficient] : value.opaque) {
    int64_t scaled = 0;
    if (!mulChecked(coefficient, scale, scaled))
      result.valid = false;
    result.opaque.emplace_back(term, scaled);
  }
  return result;
}

/// Whether `expr` reads any of the given dimension or symbol positions.
static bool usesPositions(AffineExpr expr,
                          const llvm::SmallDenseSet<unsigned> &dims,
                          const llvm::SmallDenseSet<unsigned> &symbols) {
  bool used = false;
  expr.walk([&](AffineExpr sub) {
    if (auto dim = dyn_cast<AffineDimExpr>(sub))
      used |= dims.contains(dim.getPosition());
    else if (auto symbol = dyn_cast<AffineSymbolExpr>(sub))
      used |= symbols.contains(symbol.getPosition());
  });
  return used;
}

/// Positions the lane loop's induction variable occupies in an access map.
struct LanePositions {
  llvm::SmallDenseSet<unsigned> dims;
  llvm::SmallDenseSet<unsigned> symbols;
  /// Iterations of the lane loop.
  int64_t trip = 0;
};

/// Coefficient of the lane loop in `form`, or none on overflow.
static std::optional<int64_t> laneCoefficient(const LinearForm &form,
                                              const LanePositions &lane) {
  int64_t coefficient = 0;
  for (unsigned position : lane.dims)
    if (!addChecked(coefficient, form.dims[position], coefficient))
      return std::nullopt;
  for (unsigned position : lane.symbols)
    if (!addChecked(coefficient, form.symbols[position], coefficient))
      return std::nullopt;
  return coefficient;
}

static bool isLaneInvariant(AffineExpr expr, const LanePositions &lane);

/// Largest `g` such that every lane-independent part of `form` is a multiple
/// of it, or none when the lane-dependent part is not a plain linear term.
/// A value of zero means the lane-independent part is identically zero.
static std::optional<int64_t> laneFreeDivisor(const LinearForm &form,
                                              const LanePositions &lane) {
  int64_t divisor = 0;
  auto absorb = [&](int64_t value) { divisor = std::gcd(divisor, value); };
  for (auto [position, coefficient] : llvm::enumerate(form.dims))
    if (!lane.dims.contains(position))
      absorb(coefficient);
  for (auto [position, coefficient] : llvm::enumerate(form.symbols))
    if (!lane.symbols.contains(position))
      absorb(coefficient);
  absorb(form.constant);
  for (auto [term, coefficient] : form.opaque) {
    if (!isLaneInvariant(term, lane))
      return std::nullopt;
    absorb(coefficient);
  }
  return divisor;
}

/// Whether `expr` holds the same value across every iteration of the lane
/// loop. Beyond expressions that simply do not mention it, this proves the
/// index-wrapping shape `(k*i + rest) floordiv c`: when `rest` is a multiple
/// of some `g` that divides `c`, and the lane sweep `k*i` spans less than `g`,
/// the whole sweep stays inside one `c`-aligned block and the quotient is
/// constant. Row-major wrapping -- an fftshift, a modular row stride -- is
/// exactly that shape, and proving it is what keeps such an access packable.
static bool isLaneInvariant(AffineExpr expr, const LanePositions &lane) {
  if (!usesPositions(expr, lane.dims, lane.symbols))
    return true;
  auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binary || expr.getKind() != AffineExprKind::FloorDiv)
    return false;
  auto blockExpr = dyn_cast<AffineConstantExpr>(binary.getRHS());
  if (!blockExpr || blockExpr.getValue() <= 0)
    return false;
  int64_t block = blockExpr.getValue();

  unsigned numDims = 0, numSymbols = 0;
  for (unsigned position : lane.dims)
    numDims = std::max(numDims, position + 1);
  for (unsigned position : lane.symbols)
    numSymbols = std::max(numSymbols, position + 1);
  binary.getLHS().walk([&](AffineExpr sub) {
    if (auto dim = dyn_cast<AffineDimExpr>(sub))
      numDims = std::max(numDims, dim.getPosition() + 1);
    else if (auto symbol = dyn_cast<AffineSymbolExpr>(sub))
      numSymbols = std::max(numSymbols, symbol.getPosition() + 1);
  });
  auto form = linearize(binary.getLHS(), numDims, numSymbols);
  if (!form.valid)
    return false;
  auto coefficient = laneCoefficient(form, lane);
  if (!coefficient)
    return false;
  auto divisor = laneFreeDivisor(form, lane);
  if (!divisor)
    return false;
  // A zero lane-independent part sits at the origin of every block.
  int64_t span = 0;
  if (!mulChecked(std::abs(*coefficient), lane.trip - 1, span))
    return false;
  if (*divisor == 0)
    return span < block;
  return block % *divisor == 0 && span < *divisor;
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
///
/// Non-linear sub-expressions (`floordiv`, `mod`, and products of two
/// non-constants) are admitted when they are provably lane-invariant and do
/// not shift the word boundary: their scale must be a multiple of `factor`.
/// A wrapped row base such as `(i + c) floordiv n * -n` is exactly that, so
/// an fftshift or a modular row stride does not force a port to stay scalar.
static bool isAlignedContiguous(const Access &access, AffineForOp loop,
                                unsigned factor) {
  auto type = dyn_cast<MemRefType>(access.memref.getType());
  if (!type || access.map.getNumResults() != (unsigned)type.getRank())
    return false;
  if (!loop.hasConstantBounds())
    return false;

  LanePositions lane;
  lane.trip = loop.getConstantUpperBound() - loop.getConstantLowerBound();
  if (lane.trip < factor || lane.trip % factor != 0)
    return false;
  for (auto [index, value] : llvm::enumerate(access.operands)) {
    if (value != loop.getInductionVar())
      continue;
    if (index < access.map.getNumDims())
      lane.dims.insert(index);
    else
      lane.symbols.insert(index - access.map.getNumDims());
  }

  for (auto [index, expr] : llvm::enumerate(access.map.getResults())) {
    auto form =
        linearize(expr, access.map.getNumDims(), access.map.getNumSymbols());
    if (!form.valid)
      return false;
    bool isLast = index + 1 == access.map.getNumResults();
    // An opaque term must be lane-invariant everywhere, and on the swept
    // dimension it must additionally leave the word boundary intact.
    for (auto [term, coefficient] : form.opaque) {
      if (!isLaneInvariant(term, lane))
        return false;
      if (isLast && coefficient % factor != 0)
        return false;
    }
    auto coefficient = laneCoefficient(form, lane);
    if (!coefficient || (isLast ? *coefficient != 1 : *coefficient != 0))
      return false;
    if (!isLast)
      continue;

    for (auto [operandIndex, scale] : llvm::enumerate(form.dims))
      if (!lane.dims.contains(operandIndex) && scale % factor != 0)
        return false;
    for (auto [symbolIndex, scale] : llvm::enumerate(form.symbols))
      if (!lane.symbols.contains(symbolIndex) && scale % factor != 0)
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

/// Word map for the word immediately following `access`'s aligned word.
static AffineMap getNextWordMap(const Access &access, unsigned factor) {
  SmallVector<AffineExpr> results(access.map.getResults());
  results.back() = results.back().floorDiv(factor) + 1;
  return simplifyAffineMap(AffineMap::get(access.map.getNumDims(),
                                          access.map.getNumSymbols(), results,
                                          access.map.getContext()));
}

/// Returns the constant difference `rhs - lhs` when two affine expressions
/// have identical dynamic terms. This is the shape of a sliding tap window:
/// every tap shares the same row/base expression and differs only by a small
/// compile-time offset.
static std::optional<int64_t> constantDifference(AffineExpr lhs, AffineExpr rhs,
                                                 unsigned numDims,
                                                 unsigned numSymbols) {
  LinearForm left = linearize(lhs, numDims, numSymbols);
  LinearForm right = linearize(rhs, numDims, numSymbols);
  if (!left.valid || !right.valid || left.dims != right.dims ||
      left.symbols != right.symbols ||
      left.opaque.size() != right.opaque.size())
    return std::nullopt;
  for (auto [l, r] : llvm::zip(left.opaque, right.opaque))
    if (l.first != r.first || l.second != r.second)
      return std::nullopt;
  return right.constant - left.constant;
}

/// Builds a scalar affine index for the last dimension of an access.
static Value scalarIndex(OpBuilder &builder, const Access &access) {
  AffineMap map =
      AffineMap::get(access.map.getNumDims(), access.map.getNumSymbols(),
                     access.map.getResults().back(), access.map.getContext());
  return AffineApplyOp::create(builder, access.operation->getLoc(), map,
                               access.operands);
}

/// One port and everything reached from it: the values that alias it, the
/// accesses to be repacked, and the types before and after the rewrite. Ports
/// whose reachable sets intersect are merged, since they must agree on a
/// representation.
struct Candidate {
  SmallVector<AxiPortOp> roots;
  DenseSet<Value> values;
  SmallVector<Access> accesses;
  SmallVector<Access> scalarAccesses;
  SmallVector<memref::LoadOp> dynamicLoads;
  SmallVector<memref::StoreOp> dynamicStores;
  MemRefType oldType;
  MemRefType newType;
  bool compilerOwnedScratch = false;
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
    if (auto store = dyn_cast<memref::StoreOp>(owner)) {
      if (use.getOperandNumber() == 1)
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

static bool hasOtherMemrefUse(Block *block, Value memref,
                              const DenseSet<Operation *> &ignored);

struct DynamicIndexForm {
  Value root;
  int64_t offset = 0;
  std::optional<int64_t> lower;
  std::optional<int64_t> upper;
  bool valid = true;
};

static std::optional<int64_t> constantInteger(Value value) {
  if (auto index = value.getDefiningOp<arith::ConstantIndexOp>())
    return index.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  return std::nullopt;
}

/// Normalizes the bounded integer-index shape emitted by interpolation and
/// SVA: clamp(base + offset, lo, hi), optionally followed by index_cast. The
/// root remains a runtime value, while tap offsets become compile-time data
/// that can be packed into one or two adjacent vector words.
static DynamicIndexForm parseDynamicIndex(Value value) {
  if (auto constant = constantInteger(value))
    return {Value(), *constant, std::nullopt, std::nullopt, true};
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return parseDynamicIndex(cast.getIn());
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    if (auto rhs = constantInteger(add.getRhs())) {
      auto form = parseDynamicIndex(add.getLhs());
      form.offset += *rhs;
      return form;
    }
    if (auto lhs = constantInteger(add.getLhs())) {
      auto form = parseDynamicIndex(add.getRhs());
      form.offset += *lhs;
      return form;
    }
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    if (auto rhs = constantInteger(sub.getRhs())) {
      auto form = parseDynamicIndex(sub.getLhs());
      form.offset -= *rhs;
      return form;
    }
  }
  auto applyClamp = [&](Value dynamic, bool isLower, int64_t bound) {
    auto form = parseDynamicIndex(dynamic);
    if (isLower)
      form.lower = form.lower ? std::max(*form.lower, bound) : bound;
    else
      form.upper = form.upper ? std::min(*form.upper, bound) : bound;
    return form;
  };
  if (auto max = value.getDefiningOp<arith::MaxSIOp>()) {
    if (auto rhs = constantInteger(max.getRhs()))
      return applyClamp(max.getLhs(), true, *rhs);
    if (auto lhs = constantInteger(max.getLhs()))
      return applyClamp(max.getRhs(), true, *lhs);
  }
  if (auto min = value.getDefiningOp<arith::MinSIOp>()) {
    if (auto rhs = constantInteger(min.getRhs()))
      return applyClamp(min.getLhs(), false, *rhs);
    if (auto lhs = constantInteger(min.getLhs()))
      return applyClamp(min.getRhs(), false, *lhs);
  }
  return {value, 0, std::nullopt, std::nullopt, true};
}

static bool sameDynamicIndexFamily(const DynamicIndexForm &lhs,
                                   const DynamicIndexForm &rhs) {
  return lhs.valid && rhs.valid && lhs.root == rhs.root &&
         lhs.lower == rhs.lower && lhs.upper == rhs.upper;
}

static void vectorizeDynamicWindowGroup(ArrayRef<memref::LoadOp> group,
                                        unsigned factor) {
  assert(group.size() >= 2 &&
         "dynamic window group must contain multiple loads");
  unsigned baseIndex = 0;
  SmallVector<DynamicIndexForm> forms;
  forms.reserve(group.size());
  for (memref::LoadOp load : group)
    forms.push_back(parseDynamicIndex(load.getIndices().back()));
  for (unsigned i = 1; i < forms.size(); ++i)
    if (forms[i].offset < forms[baseIndex].offset)
      baseIndex = i;

  OpBuilder builder(group.front());
  memref::LoadOp first = group.front();
  Location loc = first.getLoc();
  memref::LoadOp baseLoad = group[baseIndex];
  Value baseScalar = baseLoad.getIndices().back();
  Value divisor = arith::ConstantIndexOp::create(builder, loc, factor);
  Value baseWord = arith::DivUIOp::create(builder, loc, baseScalar, divisor);
  SmallVector<Value> firstIndices(baseLoad.getIndices());
  firstIndices.back() = baseWord;
  Value firstWord =
      memref::LoadOp::create(builder, loc, baseLoad.getMemRef(), firstIndices);
  SmallVector<Value> nextIndices(firstIndices);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  nextIndices.back() = arith::AddIOp::create(builder, loc, baseWord, one);
  Value secondWord =
      memref::LoadOp::create(builder, loc, baseLoad.getMemRef(), nextIndices);

  for (memref::LoadOp load : group) {
    builder.setInsertionPoint(load);
    Value scalar = load.getIndices().back();
    Value word = arith::DivUIOp::create(builder, loc, scalar, divisor);
    Value lane = arith::RemUIOp::create(builder, loc, scalar, divisor);
    Value useSecond = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ne, word, baseWord);
    Value selected =
        arith::SelectOp::create(builder, loc, useSecond, secondWord, firstWord);
    Value element = vector::ExtractOp::create(builder, loc, selected, lane);
    load.replaceAllUsesWith(element);
  }
  for (memref::LoadOp load : group)
    load.erase();
}

static bool isDynamicWindowGroup(ArrayRef<memref::LoadOp> group,
                                 unsigned factor) {
  if (group.size() < 2)
    return false;
  memref::LoadOp first = group.front();
  auto firstForm = parseDynamicIndex(first.getIndices().back());
  int64_t minOffset = firstForm.offset, maxOffset = firstForm.offset;
  for (memref::LoadOp load : group) {
    if (load->getBlock() != first->getBlock() ||
        load.getMemRef() != first.getMemRef() ||
        load.getIndices().size() != first.getIndices().size() ||
        !llvm::equal(load.getIndices().drop_back(),
                     first.getIndices().drop_back()))
      return false;
    auto form = parseDynamicIndex(load.getIndices().back());
    if (!sameDynamicIndexFamily(firstForm, form))
      return false;
    minOffset = std::min(minOffset, form.offset);
    maxOffset = std::max(maxOffset, form.offset);
  }
  return maxOffset - minOffset <= static_cast<int64_t>(factor);
}

static SmallVector<SmallVector<memref::LoadOp>>
groupDynamicLoads(ArrayRef<memref::LoadOp> loads, unsigned factor) {
  SmallVector<SmallVector<memref::LoadOp>> groups;
  SmallVector<std::pair<std::pair<Block *, Value>, SmallVector<memref::LoadOp>>>
      buckets;
  DenseMap<std::pair<Block *, Value>, unsigned> bucketIndices;
  for (memref::LoadOp load : loads) {
    auto key = std::make_pair(load->getBlock(), load.getMemRef());
    auto it = bucketIndices.find(key);
    if (it == bucketIndices.end()) {
      bucketIndices[key] = buckets.size();
      buckets.push_back({key, {load}});
    } else {
      buckets[it->second].second.push_back(load);
    }
  }

  for (auto &entry : buckets) {
    auto &ordered = entry.second;
    llvm::stable_sort(
        ordered, [](auto lhs, auto rhs) { return lhs->isBeforeInBlock(rhs); });
    for (unsigned start = 0; start < ordered.size();) {
      unsigned end = start + 1;
      while (end < ordered.size()) {
        SmallVector<memref::LoadOp> candidate(ordered.begin() + start,
                                              ordered.begin() + end + 1);
        if (!isDynamicWindowGroup(candidate, factor))
          break;
        ++end;
      }
      if (end - start >= 2) {
        SmallVector<memref::LoadOp> candidate(ordered.begin() + start,
                                              ordered.begin() + end);
        DenseSet<Operation *> ignored;
        for (auto load : candidate)
          ignored.insert(load.getOperation());
        if (!hasOtherMemrefUse(candidate.front()->getBlock(),
                               candidate.front().getMemRef(), ignored)) {
          groups.push_back(std::move(candidate));
          start = end;
          continue;
        }
      }
      groups.push_back({ordered[start]});
      ++start;
    }
  }
  return groups;
}

static void vectorizeDynamicStore(memref::StoreOp store, unsigned factor) {
  OpBuilder builder(store);
  Location loc = store.getLoc();
  SmallVector<Value> indices(store.getIndices());
  Value scalarIndex = indices.back();
  Value divisor = arith::ConstantIndexOp::create(builder, loc, factor);
  Value wordIndex = arith::DivUIOp::create(builder, loc, scalarIndex, divisor);
  Value lane = arith::RemUIOp::create(builder, loc, scalarIndex, divisor);
  indices.back() = wordIndex;
  Value word = memref::LoadOp::create(builder, loc, store.getMemRef(), indices);
  Value updated = vector::InsertOp::create(
      builder, loc, store.getValueToStore(), word, OpFoldResult(lane));
  memref::StoreOp::create(builder, loc, updated, store.getMemRef(), indices);
  store.erase();
}

static AffineMap getLaneMap(const Access &access, unsigned factor) {
  AffineExpr lane = access.map.getResults().back() % factor;
  return simplifyAffineMap(AffineMap::get(access.map.getNumDims(),
                                          access.map.getNumSymbols(), lane,
                                          access.map.getContext()));
}

/// Scalar fallback for a compiler-owned packed arena. It keeps unusual
/// gathers/scatters correct through word extraction or read-modify-write while
/// the regular movers on the same physical port still transfer whole words.
static void vectorizeScalarAccess(const Access &access, unsigned factor) {
  OpBuilder builder(access.operation);
  Location loc = access.operation->getLoc();
  Value lane = AffineApplyOp::create(builder, loc, getLaneMap(access, factor),
                                     access.operands);
  Value word = AffineLoadOp::create(
      builder, loc, access.memref, getWordMap(access, factor), access.operands);
  if (access.isLoad) {
    auto load = cast<AffineLoadOp>(access.operation);
    Value element = vector::ExtractOp::create(builder, loc, word, lane);
    load.replaceAllUsesWith(element);
    load.erase();
    return;
  }
  auto store = cast<AffineStoreOp>(access.operation);
  Value updated = vector::InsertOp::create(
      builder, loc, store.getValueToStore(), word, OpFoldResult(lane));
  AffineStoreOp::create(builder, loc, updated, access.memref,
                        getWordMap(access, factor), access.operands);
  store.erase();
}

/// Rewrites a run of scalar accesses that all address one packed word as a
/// single read-modify-write: one word load, the lane updates threaded through
/// it, and one word store. Done per access instead, each lane would reload and
/// rewrite the same address, which serializes a pipelined body on one port.
static void vectorizeScalarWordGroup(ArrayRef<Access> group, unsigned factor) {
  const Access &first = group.front();
  OpBuilder builder(first.operation);
  Location loc = first.operation->getLoc();
  AffineMap wordMap = getWordMap(first, factor);
  Value word =
      AffineLoadOp::create(builder, loc, first.memref, wordMap, first.operands);

  Operation *lastStore = nullptr;
  for (const Access &access : group) {
    builder.setInsertionPoint(access.operation);
    Value lane =
        AffineApplyOp::create(builder, access.operation->getLoc(),
                              getLaneMap(access, factor), access.operands);
    if (access.isLoad) {
      auto load = cast<AffineLoadOp>(access.operation);
      Value element = vector::ExtractOp::create(
          builder, access.operation->getLoc(), word, lane);
      load.replaceAllUsesWith(element);
      continue;
    }
    auto store = cast<AffineStoreOp>(access.operation);
    word = vector::InsertOp::create(builder, access.operation->getLoc(),
                                    store.getValueToStore(), word,
                                    OpFoldResult(lane));
    lastStore = access.operation;
  }

  if (lastStore) {
    builder.setInsertionPointAfter(word.getDefiningOp());
    AffineStoreOp::create(builder, loc, word, first.memref, wordMap,
                          first.operands);
  }
  for (const Access &access : group)
    access.operation->erase();
}

/// Packs scalar loads from a two-word sliding window. The first word is read
/// once, the immediately following word is read once, and every tap selects
/// the correct word before extracting its lane. This handles non-aligned
/// offsets such as `base`, `base + 2`, and `base + 4` without eight separate
/// vector loads or a read-modify-write sequence.
static void vectorizeScalarWindowGroup(ArrayRef<Access> group,
                                       unsigned factor) {
  assert(group.size() >= 2 && "window group must contain multiple loads");
  const Access *base = &group.front();
  int64_t minOffset = 0;
  for (const Access &access : group) {
    auto offset = constantDifference(
        group.front().map.getResults().back(), access.map.getResults().back(),
        group.front().map.getNumDims(), group.front().map.getNumSymbols());
    assert(offset && "window group contains incompatible affine indices");
    if (*offset < minOffset) {
      minOffset = *offset;
      base = &access;
    }
  }

  OpBuilder builder(group.front().operation);
  Location loc = group.front().operation->getLoc();
  Value baseIndex = scalarIndex(builder, *base);
  Value divisor = arith::ConstantIndexOp::create(builder, loc, factor);
  Value baseWord = arith::DivUIOp::create(builder, loc, baseIndex, divisor);
  Value firstWord = AffineLoadOp::create(
      builder, loc, base->memref, getWordMap(*base, factor), base->operands);
  Value secondWord =
      AffineLoadOp::create(builder, loc, base->memref,
                           getNextWordMap(*base, factor), base->operands);

  for (const Access &access : group) {
    builder.setInsertionPoint(access.operation);
    Value index = scalarIndex(builder, access);
    Value word = arith::DivUIOp::create(builder, loc, index, divisor);
    Value lane = arith::RemUIOp::create(builder, loc, index, divisor);
    Value next = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                       word, baseWord);
    Value selected =
        arith::SelectOp::create(builder, loc, next, secondWord, firstWord);
    Value element = vector::ExtractOp::create(builder, loc, selected, lane);
    auto load = cast<AffineLoadOp>(access.operation);
    load.replaceAllUsesWith(element);
  }
  for (const Access &access : group)
    access.operation->erase();
}

/// Whether every access in `group` addresses the same word of the same memref
/// inside one block, so that a single read-modify-write can serve all of them.
static bool isOneWordGroup(ArrayRef<Access> group, unsigned factor) {
  if (group.size() < 2)
    return false;
  const Access &first = group.front();
  Block *block = first.operation->getBlock();
  AffineMap wordMap = getWordMap(first, factor);
  for (const Access &access : group) {
    if (access.operation->getBlock() != block || access.memref != first.memref)
      return false;
    if (getWordMap(access, factor) != wordMap ||
        access.operands != first.operands)
      return false;
  }
  return true;
}

/// Whether loads form a bounded two-word sliding window. The dynamic part of
/// each last index must match, and all static offsets must fit in one word
/// span. A window is deliberately restricted to loads: stores need a
/// read-modify-write ordering proof that is separate from tap gathering.
static bool isTwoWordWindowGroup(ArrayRef<Access> group, unsigned factor) {
  if (group.size() < 2 ||
      llvm::any_of(group, [](const Access &access) { return !access.isLoad; }))
    return false;
  const Access &first = group.front();
  int64_t minOffset = 0, maxOffset = 0;
  for (const Access &access : group) {
    if (access.operation->getBlock() != first.operation->getBlock() ||
        access.memref != first.memref ||
        access.map.getNumDims() != first.map.getNumDims() ||
        access.map.getNumSymbols() != first.map.getNumSymbols() ||
        access.operands != first.operands ||
        access.map.getNumResults() != first.map.getNumResults())
      return false;
    for (unsigned index = 0; index + 1 < access.map.getNumResults(); ++index)
      if (access.map.getResult(index) != first.map.getResult(index))
        return false;
    auto offset = constantDifference(
        first.map.getResults().back(), access.map.getResults().back(),
        first.map.getNumDims(), first.map.getNumSymbols());
    if (!offset)
      return false;
    minOffset = std::min(minOffset, *offset);
    maxOffset = std::max(maxOffset, *offset);
  }
  return minOffset >= -static_cast<int64_t>(factor) &&
         maxOffset - minOffset <= static_cast<int64_t>(factor);
}

static bool hasOtherMemrefUse(Block *block, Value memref,
                              const DenseSet<Operation *> &ignored) {
  return llvm::any_of(*block, [&](Operation &operation) {
    return !ignored.contains(&operation) &&
           llvm::is_contained(operation.getOperands(), memref);
  });
}

/// Groups `accesses` into maximal runs that a single read-modify-write can
/// serve. A run is only formed when every scalar access to that memref in the
/// block lands on the same word: otherwise a neighboring word access could be
/// reordered across the merged load or store.
static SmallVector<SmallVector<Access>>
groupScalarAccesses(ArrayRef<Access> accesses, unsigned factor) {
  // Bucket by (block, memref), preserving program order within a bucket.
  SmallVector<std::pair<std::pair<Block *, Value>, SmallVector<Access>>> order;
  DenseMap<std::pair<Block *, Value>, unsigned> index;
  for (const Access &access : accesses) {
    auto key = std::make_pair(access.operation->getBlock(), access.memref);
    auto it = index.find(key);
    if (it == index.end()) {
      index[key] = order.size();
      order.push_back({key, {access}});
      continue;
    }
    order[it->second].second.push_back(access);
  }

  SmallVector<SmallVector<Access>> groups;
  for (auto &entry : order) {
    SmallVector<Access> &bucket = entry.second;
    llvm::sort(bucket, [](const Access &lhs, const Access &rhs) {
      return lhs.operation->isBeforeInBlock(rhs.operation);
    });
    DenseSet<Operation *> bucketOps;
    for (const Access &access : bucket)
      bucketOps.insert(access.operation);
    if (isOneWordGroup(bucket, factor) &&
        !hasOtherMemrefUse(entry.first.first, entry.first.second, bucketOps)) {
      groups.push_back(std::move(bucket));
      continue;
    }

    // Find maximal tap runs before falling back to one-access adapters. The
    // run may cover two adjacent packed words but must not cross another
    // memory use of the same arena.
    for (unsigned start = 0; start < bucket.size();) {
      unsigned end = start + 1;
      while (end < bucket.size()) {
        SmallVector<Access> candidate(bucket.begin() + start,
                                      bucket.begin() + end + 1);
        if (!isTwoWordWindowGroup(candidate, factor))
          break;
        ++end;
      }
      if (end - start >= 2) {
        SmallVector<Access> candidate(bucket.begin() + start,
                                      bucket.begin() + end);
        DenseSet<Operation *> candidateOps;
        for (const Access &access : candidate)
          candidateOps.insert(access.operation);
        if (!hasOtherMemrefUse(entry.first.first, entry.first.second,
                               candidateOps)) {
          groups.push_back(std::move(candidate));
          start = end;
          continue;
        }
      }
      groups.push_back({bucket[start]});
      ++start;
    }
  }
  return groups;
}

static Value laneIndex(OpBuilder &builder, Location loc, Value word,
                       unsigned factor, unsigned lane) {
  AffineExpr dim = builder.getAffineDimExpr(0);
  auto map = AffineMap::get(1, 0, dim * factor + lane);
  return AffineApplyOp::create(builder, loc, map, word);
}

static Value laneIndex(OpBuilder &builder, Location loc, Value word,
                       unsigned factor, Value lane) {
  AffineExpr wordExpr = builder.getAffineDimExpr(0);
  AffineExpr laneExpr = builder.getAffineDimExpr(1);
  auto map = AffineMap::get(2, 0, wordExpr * factor + laneExpr);
  return AffineApplyOp::create(builder, loc, map, ValueRange{word, lane});
}

static SmallVector<Value> mappedOperands(const Access &access,
                                         IRMapping &mapping) {
  SmallVector<Value> operands;
  operands.reserve(access.operands.size());
  for (Value operand : access.operands)
    operands.push_back(mapping.lookupOrDefault(operand));
  return operands;
}

/// Carries a loop's initiation-interval target across packing.
///
/// An II target records a structural hazard: the banded gather, for instance,
/// cannot beat one iteration per tap because its taps contend for the window
/// buffer's banks. Packing puts `factor` original iterations in one body, so
/// that many hazards now fall in one interval; the banks are dual-ported, so
/// two of them still retire per cycle. Leaving the original target in place
/// asks Vitis for an interval no binding can reach, and the modulo scheduler
/// spends minutes -- twenty, on the production omega-K gather -- proving that
/// before settling anyway. Both the pipeline directive (already attached by
/// this point) and the floor the directive was derived from are rescaled.
static void scaleMinimumII(AffineForOp from, AffineForOp to, unsigned factor) {
  auto scale = [&](int64_t value) {
    return std::max<int64_t>(1, (value * factor + 1) / 2);
  };
  if (auto minimum = from->getAttrOfType<IntegerAttr>(kMinIIAttr))
    to->setAttr(kMinIIAttr,
                IntegerAttr::get(minimum.getType(), scale(minimum.getInt())));
  if (auto directive = getLoopDirective(from))
    if (directive.getPipeline() && directive.getTargetII() > 1)
      setLoopDirective(to, /*pipeline=*/true, scale(directive.getTargetII()));
}

/// Packs a loop by fully unrolling its lanes in MLIR: each selected load
/// becomes one word load, the body is cloned once per lane with the induction
/// variable and the extracted lane values substituted, and each selected store
/// gathers its per-lane values into a single packed word store.
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
  scaleMinimumII(loop, newLoop, factor);
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

/// Packs a loop's aligned ports without cloning a data-dependent body in MLIR.
/// A compact lane loop is marked for HLS unrolling and writes completely
/// partitioned lane buffers; the outer loop keeps its pipeline directive. The
/// generated source stays small while Vitis sees the same parallel lanes.
static LogicalResult
vectorizeLoopCompact(AffineForOp loop, unsigned factor, unsigned computeFactor,
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
  SmallVector<Value> laneBuffers;
  {
    OpBuilder::InsertionGuard guard(builder);
    auto func = loop->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();
    builder.setInsertionPointToStart(&func.front());
    for (const Access &store : stores) {
      auto vectorType = cast<VectorType>(
          cast<MemRefType>(store.memref.getType()).getElementType());
      Value buffer = BufferOp::create(
          builder, store.operation->getLoc(),
          MemRefType::get({(int64_t)factor}, vectorType.getElementType()),
          /*depth=*/1, /*initValue=*/TypedAttr());
      buffer.getDefiningOp()->setAttr(kPartitionKindsAttr,
                                      builder.getStrArrayAttr({"complete"}));
      buffer.getDefiningOp()->setAttr(
          kPartitionFactorsAttr, builder.getI64ArrayAttr({(int64_t)factor}));
      laneBuffers.push_back(buffer);
    }
  }
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
  scaleMinimumII(loop, newLoop, factor);
  builder.setInsertionPointToStart(newLoop.getBody());

  Value firstLane =
      laneIndex(builder, loop.getLoc(), newLoop.getInductionVar(), factor, 0);
  IRMapping firstMapping;
  firstMapping.map(loop.getInductionVar(), firstLane);

  DenseMap<Operation *, Value> vectors;
  for (const Access &load : loads)
    vectors[load.operation] = AffineLoadOp::create(
        builder, load.operation->getLoc(), load.memref,
        getWordMap(load, factor), mappedOperands(load, firstMapping));

  auto laneLoop = AffineForOp::create(builder, loop.getLoc(), 0, factor);
  computeFactor = std::max(1u, std::min(computeFactor, factor));
  laneLoop->setAttr(
      kUnrollFactorAttr,
      IntegerAttr::get(IntegerType::get(loop.getContext(), 64), computeFactor));
  builder.setInsertionPointToStart(laneLoop.getBody());

  IRMapping mapping;
  Value scalarIndex =
      laneIndex(builder, loop.getLoc(), newLoop.getInductionVar(), factor,
                laneLoop.getInductionVar());
  mapping.map(loop.getInductionVar(), scalarIndex);
  for (const Access &load : loads)
    mapping.map(load.operation->getResult(0),
                vector::ExtractOp::create(builder, load.operation->getLoc(),
                                          vectors.lookup(load.operation),
                                          laneLoop.getInductionVar()));

  DenseMap<Operation *, unsigned> storeIndices;
  for (auto [index, store] : llvm::enumerate(stores))
    storeIndices[store.operation] = index;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (selectedAccesses.contains(&operation)) {
      if (auto store = dyn_cast<AffineStoreOp>(operation)) {
        unsigned index = storeIndices.lookup(&operation);
        AffineStoreOp::create(builder, store.getLoc(),
                              mapping.lookupOrDefault(store.getValueToStore()),
                              laneBuffers[index],
                              ValueRange{laneLoop.getInductionVar()});
      }
      continue;
    }
    builder.clone(operation, mapping);
  }
  builder.setInsertionPointAfter(laneLoop);
  for (auto [index, store] : llvm::enumerate(stores)) {
    SmallVector<Value> lanes;
    for (unsigned lane = 0; lane < factor; ++lane) {
      Value position = arith::ConstantIndexOp::create(
          builder, store.operation->getLoc(), lane);
      lanes.push_back(memref::LoadOp::create(builder, store.operation->getLoc(),
                                             laneBuffers[index], position));
    }
    auto vectorType = cast<VectorType>(
        cast<MemRefType>(store.memref.getType()).getElementType());
    Value packed = vector::FromElementsOp::create(
        builder, store.operation->getLoc(), vectorType, lanes);
    AffineStoreOp::create(builder, store.operation->getLoc(), packed,
                          store.memref, getWordMap(store, factor),
                          mappedOperands(store, firstMapping));
  }

  loop.erase();
  return success();
}

/// Coalesce vector read-modify-write updates to one word. Scalar fallback
/// stores are first lowered to `load -> vector.insert -> store`; when
/// several lanes target the same word, retaining that form makes Vitis issue
/// one memory transaction per lane and raises II by the lane count. Updates may
/// have unrelated source loads between them, so group them across the block,
/// thread the vector value through the inserts, and leave one final store.
static void mergeVectorReadModifyWriteGroups(ModuleOp module) {
  module.walk([&](Block *block) {
    struct Triple {
      AffineLoadOp load;
      vector::InsertOp insert;
      AffineStoreOp store;
    };
    SmallVector<SmallVector<Triple, 8>> groups;
    for (auto it = block->begin(); it != block->end(); ++it) {
      auto load = dyn_cast<AffineLoadOp>(&*it);
      if (!load || !isa<VectorType>(load.getResult().getType()))
        continue;
      auto insertPos = std::next(it);
      if (insertPos == block->end())
        continue;
      auto insert = dyn_cast<vector::InsertOp>(&*insertPos);
      auto storePos = std::next(insertPos);
      if (!insert || storePos == block->end())
        continue;
      auto store = dyn_cast<AffineStoreOp>(&*storePos);
      if (!store || insert.getDest() != load.getResult() ||
          store.getValueToStore() != insert.getResult() ||
          store.getMemRef() != load.getMemRef() ||
          store.getAffineMap() != load.getAffineMap() ||
          store.getMapOperands() != load.getMapOperands())
        continue;

      auto sameWord = [&](Triple triple) {
        auto candidate = triple.load;
        return MemRefAccess(candidate) == MemRefAccess(load);
      };
      auto group = llvm::find_if(groups, [&](const auto &candidate) {
        return sameWord(candidate.front());
      });
      if (group == groups.end())
        groups.push_back({{load, insert, store}});
      else
        group->push_back({load, insert, store});
    }

    for (auto &group : groups) {
      if (group.size() < 2 || llvm::any_of(group, [](const Triple &triple) {
            return !triple.load->getResult(0).hasOneUse() ||
                   !triple.insert->getResult(0).hasOneUse();
          }))
        continue;

      DenseSet<Operation *> groupOps;
      for (const Triple &triple : group) {
        groupOps.insert(triple.load);
        groupOps.insert(triple.store);
      }
      Value memref = group.front().load.getMemRef();
      if (hasOtherMemrefUse(block, memref, groupOps))
        continue;

      Value current = group.front().insert.getResult();
      group.front().store.erase();
      for (unsigned index = 1; index < group.size(); ++index) {
        auto &triple = group[index];
        triple.insert.getDestMutable().assign(current);
        current = triple.insert.getResult();
        triple.load.erase();
        if (index + 1 != group.size())
          triple.store.erase();
      }
    }
  });
}

struct WidenExternalMemory
    : public sar::impl::WidenExternalMemoryBase<WidenExternalMemory> {
  WidenExternalMemory() = default;
  WidenExternalMemory(unsigned argBusBits, unsigned argMaxLanes,
                      unsigned argMinElements, bool argPackPublicOutputs,
                      unsigned argComputeMaxLanes) {
    busBits = argBusBits;
    maxLanes = argMaxLanes;
    minElements = argMinElements;
    packPublicOutputs = argPackPublicOutputs;
    computeMaxLanes = argComputeMaxLanes;
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

    // Candidates that share a value describe one access group and must be
    // merged into one, or the same value is rewritten twice. Absorbing a
    // later candidate grows `values`, which can bring an *earlier* one into
    // the group -- so the scan restarts rather than continuing forward.
    // Without that, whether the group closes depends on port order: with
    // ports (a), (b), (a+b) the bridging entry comes last and closes it;
    // with (a), (a+b), (b) it does not.
    for (unsigned i = 0; i < candidates.size(); ++i) {
      bool merged = true;
      while (merged) {
        merged = false;
        for (unsigned j = i + 1; j < candidates.size();) {
          if (!intersects(candidates[i].values, candidates[j].values)) {
            ++j;
            continue;
          }
          candidates[i].roots.append(candidates[j].roots);
          candidates[i].values.insert(candidates[j].values.begin(),
                                      candidates[j].values.end());
          candidates.erase(candidates.begin() + j);
          merged = true;
        }
      }
    }

    SmallVector<Candidate, 4> selected;
    DenseSet<Operation *> selectedAccesses;
    DenseSet<Operation *> loops;
    for (Candidate &candidate : candidates) {
      auto type =
          dyn_cast<MemRefType>(candidate.roots.front().getElement().getType());
      bool compilerOwnedScratch =
          llvm::all_of(candidate.roots, [](AxiPortOp port) {
            return port->hasAttr("hls.scratch");
          });
      if (!type)
        continue;
      if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
          (type.getRank() < 2 && !compilerOwnedScratch) ||
          !type.getElementType().isIntOrFloat() ||
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
          if (auto store = dyn_cast<memref::StoreOp>(owner)) {
            if (use.getOperandNumber() != 1) {
              valid = false;
              break;
            }
            candidate.dynamicStores.push_back(store);
            continue;
          }
          auto access = composeAccess(owner);
          auto loop = owner->getParentOfType<AffineForOp>();
          bool flatLoopBody =
              loop && llvm::all_of(loop.getBody()->without_terminator(),
                                   [](Operation &operation) {
                                     return operation.getNumRegions() == 0;
                                   });
          if (failed(access)) {
            valid = false;
            break;
          }
          if (flatLoopBody && owner->getBlock() == loop.getBody() &&
              isAlignedContiguous(*access, loop, factor))
            candidate.accesses.push_back(std::move(*access));
          else if (compilerOwnedScratch)
            candidate.scalarAccesses.push_back(std::move(*access));
          else {
            valid = false;
            break;
          }
        }
      if (!valid ||
          (candidate.accesses.empty() && candidate.scalarAccesses.empty() &&
           candidate.dynamicLoads.empty() && candidate.dynamicStores.empty()))
        continue;

      // A loop containing even one scalar fallback stays scalar as a unit.
      // Mixing read-modify-write adapters with a simultaneous whole-loop
      // rewrite makes the selected operation set unstable and rarely improves
      // throughput; clean mover loops on the same arena are still packed.
      DenseSet<Operation *> fallbackLoops;
      for (const Access &access : candidate.scalarAccesses)
        if (auto loop = access.operation->getParentOfType<AffineForOp>())
          fallbackLoops.insert(loop);
      for (memref::StoreOp store : candidate.dynamicStores)
        if (auto loop = store->getParentOfType<AffineForOp>())
          fallbackLoops.insert(loop);
      SmallVector<Access> packedAccesses;
      for (Access &access : candidate.accesses) {
        auto loop = access.operation->getParentOfType<AffineForOp>();
        if (loop && fallbackLoops.contains(loop))
          candidate.scalarAccesses.push_back(std::move(access));
        else
          packedAccesses.push_back(std::move(access));
      }
      candidate.accesses = std::move(packedAccesses);
      bool nestedRewriteConflict = false;
      for (auto [lhsIndex, lhs] : llvm::enumerate(candidate.accesses))
        for (const Access &rhs :
             ArrayRef(candidate.accesses).drop_front(lhsIndex + 1)) {
          auto lhsLoop = lhs.operation->getParentOfType<AffineForOp>();
          auto rhsLoop = rhs.operation->getParentOfType<AffineForOp>();
          nestedRewriteConflict |= lhsLoop && rhsLoop &&
                                   (lhsLoop->isProperAncestor(rhsLoop) ||
                                    rhsLoop->isProperAncestor(lhsLoop));
        }
      if (nestedRewriteConflict) {
        if (!compilerOwnedScratch)
          continue;
        candidate.scalarAccesses.append(
            std::make_move_iterator(candidate.accesses.begin()),
            std::make_move_iterator(candidate.accesses.end()));
        candidate.accesses.clear();
      }
      if (compilerOwnedScratch &&
          candidate.scalarAccesses.size() + candidate.dynamicStores.size() >
              kMaxScratchScalarFallbacks)
        continue;
      if (compilerOwnedScratch && type.getElementTypeBitWidth() > 32) {
        unsigned adapters = candidate.dynamicLoads.size() +
                            2 * candidate.dynamicStores.size() +
                            candidate.scalarAccesses.size();
        bool scalarWrite =
            llvm::any_of(candidate.scalarAccesses,
                         [](const Access &access) { return !access.isLoad; });
        // Double-word read-modify-write adapters have a disproportionate
        // binding cost. Pack f64 scratch only when all irregular accesses are
        // reads and the adapter body remains bounded; aligned movers may still
        // read and write complete words.
        if (scalarWrite || !candidate.dynamicStores.empty() || adapters > 64)
          continue;
      }
      if (!compilerOwnedScratch && !candidate.scalarAccesses.empty())
        continue;

      // A public port may change representation when it is purely read or
      // purely written and every affine access is a complete aligned group.
      // Read/write ports retain their scalar ABI: packing them would require
      // public read-modify-write semantics. Compiler-owned scratch has no
      // host-visible element layout and may use the scalar adapters above.
      bool hasReads =
          !candidate.dynamicLoads.empty() ||
          llvm::any_of(candidate.accesses,
                       [](const Access &access) { return access.isLoad; });
      bool hasWrites =
          !candidate.dynamicStores.empty() ||
          llvm::any_of(candidate.accesses,
                       [](const Access &access) { return !access.isLoad; });
      if (!compilerOwnedScratch && hasReads && hasWrites)
        continue;
      if (!compilerOwnedScratch && hasWrites && !packPublicOutputs)
        continue;
      if ((!candidate.dynamicStores.empty() ||
           llvm::any_of(candidate.scalarAccesses,
                        [](const Access &access) { return !access.isLoad; })) &&
          !compilerOwnedScratch)
        continue;

      auto shape = SmallVector<int64_t>(type.getShape());
      shape.back() /= factor;
      auto vectorType =
          VectorType::get({(int64_t)factor}, type.getElementType());
      candidate.oldType = type;
      candidate.newType = MemRefType::get(shape, vectorType, AffineMap(),
                                          type.getMemorySpace());
      candidate.compilerOwnedScratch = compilerOwnedScratch;
      selected.push_back(std::move(candidate));
    }
    if (selected.empty())
      return;

    // Rewriting both an outer loop and one of its descendants is not
    // composable: the inner rewrite replaces a region the outer rewrite would
    // subsequently try to clone. Keep the descendant packed; an outer scratch
    // access falls back to scalar word extraction, while a public port keeps
    // its original ABI.
    SmallVector<AffineForOp> candidateLoops;
    for (Candidate &candidate : selected)
      for (const Access &access : candidate.accesses)
        candidateLoops.push_back(
            access.operation->getParentOfType<AffineForOp>());
    DenseSet<Operation *> conflictingOuterLoops;
    for (AffineForOp lhs : candidateLoops)
      for (AffineForOp rhs : candidateLoops)
        if (lhs && rhs && lhs != rhs && lhs->isProperAncestor(rhs))
          conflictingOuterLoops.insert(lhs);
    SmallVector<Candidate, 4> filtered;
    for (Candidate &candidate : selected) {
      // Packing retypes the port *and its bundle*. A bundle shared by
      // several read-only inputs carries one element type for all of them,
      // so retyping it for one would leave the others disagreeing with it.
      // Those ports are small tables by construction -- that is why they
      // share -- so the packing they give up is worth less than the master
      // the sharing saves.
      if (llvm::any_of(candidate.roots, [&](AxiPortOp port) {
            auto bundle = port.getBundle().getDefiningOp<AxiBundleOp>();
            if (!bundle)
              return false;
            unsigned users = 0;
            for (Operation *user : bundle.getBundle().getUsers())
              users += isa<AxiPortOp>(user);
            return users > 1;
          }))
        continue;
      bool rejectsPublic = false;
      SmallVector<Access> packed;
      for (Access &access : candidate.accesses) {
        auto loop = access.operation->getParentOfType<AffineForOp>();
        bool mustStayScalar = loop && conflictingOuterLoops.contains(loop);
        if (!mustStayScalar) {
          packed.push_back(std::move(access));
          continue;
        }
        if (!candidate.compilerOwnedScratch) {
          rejectsPublic = true;
          break;
        }
        candidate.scalarAccesses.push_back(std::move(access));
      }
      if (rejectsPublic)
        continue;
      candidate.accesses = std::move(packed);
      if (candidate.compilerOwnedScratch &&
          candidate.scalarAccesses.size() + candidate.dynamicStores.size() >
              kMaxScratchScalarFallbacks)
        continue;
      filtered.push_back(std::move(candidate));
    }
    selected = std::move(filtered);
    if (selected.empty())
      return;
    DenseSet<Operation *> compactLoops;
    for (Candidate &candidate : selected) {
      for (memref::LoadOp load : candidate.dynamicLoads)
        if (auto loop = load->getParentOfType<AffineForOp>())
          compactLoops.insert(loop);
      for (memref::StoreOp store : candidate.dynamicStores)
        if (auto loop = store->getParentOfType<AffineForOp>())
          compactLoops.insert(loop);
    }
    for (Candidate &candidate : selected)
      for (const Access &access : candidate.accesses) {
        selectedAccesses.insert(access.operation);
        loops.insert(access.operation->getParentOfType<AffineForOp>());
      }

    for (Candidate &candidate : selected)
      for (Value value : candidate.values)
        value.setType(candidate.newType);

    for (Candidate &candidate : selected)
      for (auto &group : groupDynamicLoads(candidate.dynamicLoads, factor))
        if (group.size() >= 2)
          vectorizeDynamicWindowGroup(group, factor);
        else
          vectorizeDynamicLoad(group.front(), factor);
    for (Candidate &candidate : selected)
      for (memref::StoreOp store : candidate.dynamicStores)
        vectorizeDynamicStore(store, factor);
    for (Candidate &candidate : selected)
      for (auto &group : groupScalarAccesses(candidate.scalarAccesses, factor))
        if (group.size() == 1)
          vectorizeScalarAccess(group.front(), factor);
        else if (isTwoWordWindowGroup(group, factor))
          vectorizeScalarWindowGroup(group, factor);
        else
          vectorizeScalarWordGroup(group, factor);

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
      if (failed(compactLoops.contains(loop)
                     ? vectorizeLoopCompact(loop, factor,
                                            computeMaxLanes ? computeMaxLanes
                                                            : factor,
                                            selectedAccesses)
                     : vectorizeLoop(loop, factor, selectedAccesses))) {
        loop.emitError("failed to pack a proven external access group");
        return signalPassFailure();
      }

    mergeVectorReadModifyWriteGroups(module);

    refreshFunctionTypes(module);
  }
};

} // namespace

std::unique_ptr<Pass>
sar::createWidenExternalMemoryPass(unsigned busBits, unsigned maxLanes,
                                   unsigned minElements, bool packPublicOutputs,
                                   unsigned computeMaxLanes) {
  return std::make_unique<WidenExternalMemory>(
      busBits, maxLanes, minElements, packPublicOutputs, computeMaxLanes);
}
