//===- FuseSiblingLoops.cpp - fuse independent affine sweeps --------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Splitting complex values leaves the real and imaginary halves of one phase
// expression in two adjacent sweeps over the same iteration space. Each
// recomputes the geometry the phase is built from -- the same coordinates,
// the same trigonometry. Merging the pair puts those expressions in one body,
// where CSE removes the duplicate.
//
// Two nests fuse when they iterate identically and neither writes what the
// other touches, which is what makes running them interleaved equivalent to
// running them in sequence.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "sar/Dialect/HLS/IR/Utils.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_FUSESIBLINGLOOPS
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::sar;

namespace {
struct Accesses {
  DenseSet<Value> reads;
  DenseSet<Value> writes;
  bool supported = true;
};

static bool isPlainMemref(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.hasStaticShape() && type.getLayout().isIdentity() &&
         !value.getDefiningOp<memref::SubViewOp>();
}

static Accesses collectAccesses(AffineForOp loop) {
  Accesses accesses;
  loop.walk([&](Operation *op) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      accesses.supported &= isPlainMemref(load.getMemRef());
      accesses.reads.insert(load.getMemRef());
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      accesses.supported &= isPlainMemref(store.getMemRef());
      accesses.writes.insert(store.getMemRef());
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      accesses.supported &= isPlainMemref(load.getMemRef());
      accesses.reads.insert(load.getMemRef());
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      accesses.supported &= isPlainMemref(store.getMemRef());
      accesses.writes.insert(store.getMemRef());
    } else if (isa<func::CallOp, memref::CopyOp>(op)) {
      accesses.supported = false;
    }
  });
  return accesses;
}

static bool intersects(const DenseSet<Value> &lhs, const DenseSet<Value> &rhs) {
  return llvm::any_of(lhs, [&](Value value) { return rhs.contains(value); });
}

static bool isResultArgument(Value value, func::FuncOp func) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &func.front())
    return false;
  auto names = func->getAttrOfType<ArrayAttr>("sar.arg_names");
  return names && argument.getArgNumber() >= names.size();
}

static bool hasReadOutsidePair(Value value, AffineForOp first,
                               AffineForOp second) {
  return llvm::any_of(value.getUses(), [&](OpOperand &use) {
    Operation *owner = use.getOwner();
    return isRead(use) && !first->isAncestor(owner) &&
           !second->isAncestor(owner);
  });
}

static bool writesResult(const Accesses &accesses, func::FuncOp func) {
  return func && llvm::any_of(accesses.writes, [&](Value value) {
           return isResultArgument(value, func);
         });
}

static bool writesLiveInternal(const Accesses &accesses, func::FuncOp func,
                               AffineForOp first, AffineForOp second) {
  return llvm::any_of(accesses.writes, [&](Value value) {
    return !isResultArgument(value, func) &&
           hasReadOutsidePair(value, first, second);
  });
}

/// The chain of loops under `root` that nest perfectly, outermost first.
/// Empty when `root` holds anything beside a single child loop.
static SmallVector<AffineForOp> getPerfectNest(AffineForOp root) {
  SmallVector<AffineForOp> nest{root};
  while (true) {
    AffineForOp child;
    for (Operation &op : root.getBody()->without_terminator()) {
      auto loop = dyn_cast<AffineForOp>(op);
      if (!loop || child)
        return nest;
      child = loop;
    }
    if (!child)
      return nest;
    nest.push_back(child);
    root = child;
  }
}

/// Whether two nests have identical depth, bounds and steps at every level.
static bool sameIterationSpace(ArrayRef<AffineForOp> lhs,
                               ArrayRef<AffineForOp> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    AffineForOp a = lhs[i];
    AffineForOp b = rhs[i];
    if (a.getLowerBoundMap() != b.getLowerBoundMap() ||
        a.getLowerBoundOperands() != b.getLowerBoundOperands() ||
        a.getUpperBoundMap() != b.getUpperBoundMap() ||
        a.getUpperBoundOperands() != b.getUpperBoundOperands() ||
        a.getStep() != b.getStep() || a.getNumIterOperands() != 0 ||
        b.getNumIterOperands() != 0)
      return false;
  }
  return true;
}

static bool pointwiseProducerConsumer(AffineForOp first, AffineForOp second,
                                      const Accesses &producer,
                                      const Accesses &consumer,
                                      const IRMapping &mapping) {
  SmallVector<Value> shared;
  for (Value value : producer.writes)
    if (consumer.reads.contains(value))
      shared.push_back(value);
  if (shared.empty())
    return false;

  for (Value value : shared) {
    SmallVector<Operation *> stores;
    SmallVector<Operation *> loads;
    first.walk([&](Operation *op) {
      if (auto store = dyn_cast<AffineStoreOp>(op)) {
        if (store.getMemRef() == value)
          stores.push_back(op);
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        if (store.getMemRef() == value)
          stores.push_back(op);
      }
    });
    second.walk([&](Operation *op) {
      if (auto load = dyn_cast<AffineLoadOp>(op)) {
        if (load.getMemRef() == value)
          loads.push_back(op);
      } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
        if (load.getMemRef() == value)
          loads.push_back(op);
      }
    });
    if (!llvm::hasSingleElement(stores) || !llvm::hasSingleElement(loads))
      return false;
    auto store = dyn_cast<AffineStoreOp>(stores.front());
    auto load = dyn_cast<AffineLoadOp>(loads.front());
    if (!store || !load || store.getAffineMap() != load.getAffineMap())
      return false;
    auto storeOperands = store.getMapOperands();
    auto loadOperands = load.getMapOperands();
    if (storeOperands.size() != loadOperands.size())
      return false;
    for (auto [storeOperand, loadOperand] :
         llvm::zip(storeOperands, loadOperands))
      if (storeOperand != mapping.lookupOrDefault(loadOperand))
        return false;
  }
  return true;
}

static bool fusePair(AffineForOp first, AffineForOp second,
                     ArrayRef<Operation *> hoistableBetween,
                     bool preserveDataflowRoles, IRRewriter &rewriter) {
  auto firstNest = getPerfectNest(first);
  auto secondNest = getPerfectNest(second);
  if (!sameIterationSpace(firstNest, secondNest))
    return false;

  uint64_t operationCount = 0;
  first.walk([&](Operation *) { ++operationCount; });
  second.walk([&](Operation *) { ++operationCount; });
  if (operationCount > 1024)
    return false;

  Accesses a = collectAccesses(first), b = collectAccesses(second);
  if (!a.supported || !b.supported || intersects(a.reads, b.writes) ||
      intersects(a.writes, b.writes))
    return false;

  IRMapping mapping;
  for (auto [from, to] : llvm::zip(secondNest, firstNest))
    mapping.map(from.getInductionVar(), to.getInductionVar());
  auto func = first->getParentOfType<func::FuncOp>();
  // Vitis dataflow processes should either write caller outputs or feed a
  // successor channel. Combining both roles produces HLS 200-1450 and can
  // serialize the successor behind the output handshake.
  if (preserveDataflowRoles &&
      ((writesResult(a, func) && writesLiveInternal(b, func, first, second)) ||
       (writesResult(b, func) && writesLiveInternal(a, func, first, second))))
    return false;
  bool hasProducerConsumer = intersects(a.writes, b.reads);
  if (hasProducerConsumer) {
    if (!pointwiseProducerConsumer(first, second, a, b, mapping))
      return false;
  }

  for (Operation *operation : hoistableBetween)
    operation->moveBefore(first);
  Block *source = secondNest.back().getBody();
  Block *target = firstNest.back().getBody();
  rewriter.setInsertionPoint(target->getTerminator());
  for (Operation &op : source->without_terminator())
    rewriter.clone(op, mapping);
  rewriter.eraseOp(second);
  return true;
}

struct FuseSiblingLoops
    : public sar::impl::FuseSiblingLoopsBase<FuseSiblingLoops> {
  FuseSiblingLoops() = default;
  explicit FuseSiblingLoops(bool preserveRoles) {
    preserveDataflowRoles = preserveRoles;
  }

  void runOnOperation() override {
    auto func = getOperation();
    bool changed = true;
    IRRewriter rewriter(func.getContext());
    while (changed) {
      changed = false;
      AffineForOp previous;
      SmallVector<Operation *> hoistableBetween;
      for (Operation &op : llvm::make_early_inc_range(func.front())) {
        auto loop = dyn_cast<AffineForOp>(op);
        if (!loop) {
          if (previous && isa<memref::AllocOp, memref::AllocaOp,
                              memref::GetGlobalOp, arith::ConstantOp>(op)) {
            hoistableBetween.push_back(&op);
            continue;
          }
          previous = {};
          hoistableBetween.clear();
          continue;
        }
        if (previous) {
          if (fusePair(previous, loop, hoistableBetween, preserveDataflowRoles,
                       rewriter)) {
            changed = true;
            break;
          }
        }
        previous = loop;
        hoistableBetween.clear();
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass>
sar::createFuseSiblingLoopsPass(bool preserveDataflowRoles) {
  return std::make_unique<FuseSiblingLoops>(preserveDataflowRoles);
}
