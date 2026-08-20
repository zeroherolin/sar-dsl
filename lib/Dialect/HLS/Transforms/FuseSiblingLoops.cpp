//===- FuseSiblingLoops.cpp - fuse independent affine sweeps -------------===//
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
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

static bool fusePair(AffineForOp first, AffineForOp second,
                     IRRewriter &rewriter) {
  if (first->hasAttr("sar.sibling_fused") ||
      second->hasAttr("sar.sibling_fused"))
    return false;
  auto firstNest = getPerfectNest(first);
  auto secondNest = getPerfectNest(second);
  if (!sameIterationSpace(firstNest, secondNest))
    return false;

  Accesses a = collectAccesses(first), b = collectAccesses(second);
  if (!a.supported || !b.supported || intersects(a.writes, b.reads) ||
      intersects(a.reads, b.writes) || intersects(a.writes, b.writes))
    return false;

  IRMapping mapping;
  for (auto [from, to] : llvm::zip(secondNest, firstNest))
    mapping.map(from.getInductionVar(), to.getInductionVar());

  Block *source = secondNest.back().getBody();
  Block *target = firstNest.back().getBody();
  rewriter.setInsertionPoint(target->getTerminator());
  for (Operation &op : source->without_terminator())
    rewriter.clone(op, mapping);
  first->setAttr("sar.sibling_fused", rewriter.getUnitAttr());
  rewriter.eraseOp(second);
  return true;
}

struct FuseSiblingLoops
    : public sar::impl::FuseSiblingLoopsBase<FuseSiblingLoops> {
  void runOnOperation() override {
    auto func = getOperation();
    bool changed = true;
    IRRewriter rewriter(func.getContext());
    while (changed) {
      changed = false;
      AffineForOp previous;
      for (Operation &op : llvm::make_early_inc_range(func.front())) {
        auto loop = dyn_cast<AffineForOp>(op);
        if (!loop) {
          previous = {};
          continue;
        }
        if (previous && fusePair(previous, loop, rewriter)) {
          changed = true;
          break;
        }
        previous = loop;
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> sar::createFuseSiblingLoopsPass() {
  return std::make_unique<FuseSiblingLoops>();
}
