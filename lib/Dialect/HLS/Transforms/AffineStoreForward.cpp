//===- AffineStoreForward.cpp - forward stores to loads across ifs --------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "sar/Dialect/HLS/IR/Utils.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include <algorithm>

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_AFFINESTOREFORWARD
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace sar::hls;

/// Proves that no potentially aliasing `EffectType` occurs between `start` and
/// `memOp`. Unknown effects and aliases are handled conservatively. This
/// extends the upstream affine helper to other effect/op combinations.
template <typename EffectType>
static bool
hasNoInterveningEffect(Operation *start, Operation *memOp, Value memref,
                       llvm::function_ref<bool(Value, Value)> mayAlias) {
  bool hasSideEffect = false;

  std::function<void(Operation *)> checkOperation = [&](Operation *op) {
    if (hasSideEffect)
      return;

    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffects(effects);

      bool opMayHaveEffect = false;
      for (auto effect : effects) {
        if (isa<EffectType>(effect.getEffect())) {
          if (effect.getValue() && effect.getValue() != memref &&
              !mayAlias(effect.getValue(), memref))
            continue;
          opMayHaveEffect = true;
          break;
        }
      }

      if (!opMayHaveEffect)
        return;

      if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(
              op) &&
          isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(
              memOp)) {
        affine::MemRefAccess srcAccess(op);
        affine::MemRefAccess destAccess(memOp);

        if (srcAccess.memref == destAccess.memref &&
            affine::getAffineScope(op) == affine::getAffineScope(memOp) &&
            affine::getAffineScope(op) == affine::getAffineScope(start)) {
          unsigned minSurroundingLoops =
              affine::getNumCommonSurroundingLoops(*start, *memOp);

          unsigned nsLoops = affine::getNumCommonSurroundingLoops(*op, *memOp);

          unsigned d;
          affine::FlatAffineValueConstraints dependenceConstraints;
          for (d = nsLoops + 1; d > minSurroundingLoops; d--) {
            affine::DependenceResult result =
                affine::checkMemrefAccessDependence(
                    srcAccess, destAccess, d, &dependenceConstraints,
                    /*dependenceComponents=*/nullptr);
            if (!noDependence(result)) {
              hasSideEffect = true;
              return;
            }
          }

          return;
        }
      }
      hasSideEffect = true;
      return;
    }

    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Operation &op : block)
            checkOperation(&op);
      return;
    }

    hasSideEffect = true;
  };

  auto until = [&](Operation *parent, Operation *to) {
    assert(parent->isAncestor(to));
    checkOperation(parent);
  };

  std::function<void(Operation *, Operation *)> recur =
      [&](Operation *from, Operation *untilOp) {
        assert(
            from->getParentRegion()->isAncestor(untilOp->getParentRegion()) &&
            "Checking for side effect between two operations without a common "
            "ancestor");

        if (from->getParentRegion() != untilOp->getParentRegion()) {
          recur(from, untilOp->getParentOp());
          until(untilOp->getParentOp(), untilOp);
          return;
        }

        SmallVector<Block *, 2> todoBlocks;
        {
          for (auto iter = ++from->getIterator(), end = from->getBlock()->end();
               iter != end && &*iter != untilOp; ++iter) {
            checkOperation(&*iter);
          }

          if (untilOp->getBlock() != from->getBlock())
            for (Block *succ : from->getBlock()->getSuccessors())
              todoBlocks.push_back(succ);
        }

        llvm::SmallDenseSet<Block *, 4> done;
        while (!todoBlocks.empty()) {
          Block *blk = todoBlocks.pop_back_val();
          if (done.count(blk))
            continue;
          done.insert(blk);
          for (auto &op : *blk) {
            if (&op == untilOp)
              break;
            checkOperation(&op);
            if (&op == blk->getTerminator())
              for (Block *succ : blk->getSuccessors())
                todoBlocks.push_back(succ);
          }
        }
      };

  recur(start, memOp);
  return !hasSideEffect;
}

/// A helper to check whether an ifOp is valid to be replaced with select.
bool isValid(AffineIfOp ifOp, Operation *targetOp) {
  return !ifOp.hasElse() && ifOp.getThenBlock()->getOperations().size() == 2 &&
         ifOp->getParentRegion()->isAncestor(targetOp->getParentRegion());
}

/// Attempt to eliminate loadOp by replacing it with a value stored into memory
/// which the load is guaranteed to retrieve. This check involves three
/// components: 1) The store and load must be on the same location 2) The store
/// must dominate (and therefore must always occur prior to) the load 3) No
/// other operations will overwrite the memory loaded between the given load
/// and store.  If such a value exists, the replaced `loadOp` will be added to
/// `loadOpsToErase` and its memref will be added to `memrefsToErase`.
static mlir::affine::AffineReadOpInterface
forwardStoreToLoad(mlir::affine::AffineReadOpInterface loadOp,
                   SmallVectorImpl<Operation *> &loadOpsToErase,
                   SmallPtrSetImpl<Value> &memrefsToErase,
                   DominanceInfo &domInfo,
                   llvm::function_ref<bool(Value, Value)> mayAlias) {

  // The store op candidate for forwarding that satisfies all conditions
  // to replace the load, if any.
  mlir::affine::AffineWriteOpInterface lastWriteStoreOp = nullptr;

  for (auto *user : loadOp.getMemRef().getUsers()) {
    auto storeOp = dyn_cast<mlir::affine::AffineWriteOpInterface>(user);
    if (!storeOp)
      continue;
    MemRefAccess srcAccess(storeOp);
    MemRefAccess destAccess(loadOp);

    // 1. Check if the store and the load have mathematically equivalent
    // affine access functions; this implies that they statically refer to the
    // same single memref element. As an example this filters out cases like:
    //     store %A[%i0 + 1]
    //     load %A[%i0]
    //     store %A[%M]
    //     load %A[%N]
    // Use the AffineValueMap difference based memref access equality checking.
    if (srcAccess != destAccess)
      continue;

    // A store that is the sole operation inside an `affine.if` is the
    // special case: the if statement then stands in
    // as start for intervening effect searching.
    Operation *startOp = storeOp;
    if (startOp->getParentRegion() != loadOp->getParentRegion()) {
      if (auto ifOp =
              dyn_cast<mlir::affine::AffineIfOp>(storeOp->getParentOp()))
        if (isValid(ifOp, loadOp) && !ifAlwaysTrueOrFalse(ifOp).second)
          startOp = ifOp;
    }

    // 2. The store has to dominate the load op to be candidate.
    if (!domInfo.dominates(startOp, loadOp))
      continue;

    // 3. Ensure there is no intermediate operation which could replace the
    // value in memory.
    if (!hasNoInterveningEffect<MemoryEffects::Write>(
            startOp, loadOp, loadOp.getMemRef(), mayAlias))
      continue;

    // This is a forwarding candidate. Two candidates that both
    // dominate the load with no intervening write should be impossible; if
    // the analysis ever disagrees, forwarding either would be a guess, so
    // bail out rather than pick one.
    if (lastWriteStoreOp)
      return loadOp;
    lastWriteStoreOp = storeOp;
  }

  if (!lastWriteStoreOp)
    return loadOp;

  // Perform the actual store to load forwarding.
  Value storeVal = lastWriteStoreOp.getValueToStore();
  // Check if 2 values have the same shape. This is needed for affine vector
  // loads and stores.
  if (storeVal.getType() != loadOp.getValue().getType())
    return loadOp;

  if (!domInfo.dominates(lastWriteStoreOp, loadOp)) {
    // Special case when the store is inside of an if statement.
    auto ifOp = lastWriteStoreOp->getParentOfType<mlir::affine::AffineIfOp>();
    lastWriteStoreOp->moveBefore(ifOp);

    // Create a load and select op as the new value to write.
    auto builder = OpBuilder(ifOp);
    builder.setInsertionPoint(lastWriteStoreOp);
    auto newLoad =
        cast<mlir::affine::AffineReadOpInterface>(builder.clone(*loadOp));
    auto select = hls::AffineSelectOp::create(
        builder, ifOp.getLoc(), ifOp.getIntegerSet(), ifOp.getOperands(),
        storeVal, newLoad.getValue());
    ifOp->erase();

    auto valueIdx = llvm::find(lastWriteStoreOp->getOperands(), storeVal) -
                    lastWriteStoreOp->operand_begin();
    lastWriteStoreOp->getOpOperand(valueIdx).set(select);
    loadOp.getValue().replaceAllUsesWith(select);

    loadOpsToErase.push_back(loadOp);
    return newLoad;
  }

  // Normal case for direct forwarding.
  loadOp.getValue().replaceAllUsesWith(storeVal);
  memrefsToErase.insert(loadOp.getMemRef());
  loadOpsToErase.push_back(loadOp);
  return mlir::affine::AffineReadOpInterface();
}

// This attempts to find stores which have no impact on the final result.
// A writing op writeA will be eliminated if there exists an op writeB if
// 1) writeA and writeB have mathematically equivalent affine access functions.
// 2) writeB postdominates writeA.
// 3) There is no potential read between writeA and writeB.
static void findUnusedStore(mlir::affine::AffineWriteOpInterface writeA,
                            SmallVectorImpl<Operation *> &opsToErase,
                            SmallPtrSetImpl<Value> &memrefsToErase,
                            PostDominanceInfo &postDominanceInfo,
                            llvm::function_ref<bool(Value, Value)> mayAlias) {
  auto memref = writeA.getMemRef();
  for (Operation *user : writeA.getMemRef().getUsers()) {
    // Only consider writing operations.
    auto writeB = dyn_cast<mlir::affine::AffineWriteOpInterface>(user);
    if (!writeB)
      continue;

    // The operations must be distinct.
    if (writeB == writeA)
      continue;

    // Both operations must write to the same memory.
    MemRefAccess srcAccess(writeB);
    MemRefAccess destAccess(writeA);

    if (srcAccess != destAccess)
      continue;

    // Both operations must lie in the same region. Likewise a
    // special case that when write A is the sole operation in an if statement,
    // where write B is possible to be unused.
    Operation *targetA = writeA;
    Operation *targetB = writeB;
    if (targetA->getParentRegion() != targetB->getParentRegion()) {
      if (auto ifOpA =
              dyn_cast<mlir::affine::AffineIfOp>(writeA->getParentOp())) {
        if (isValid(ifOpA, writeB))
          targetA = ifOpA;
        if (auto ifOpB =
                dyn_cast<mlir::affine::AffineIfOp>(writeB->getParentOp()))
          if (checkSameIfStatement(ifOpA, ifOpB))
            targetB = ifOpB;
      } else if (auto ifOpB = dyn_cast<mlir::affine::AffineIfOp>(
                     writeB->getParentOp())) {
        if (isValid(ifOpB, writeA))
          targetB = ifOpB;
      }
      if (targetA->getParentRegion() != targetB->getParentRegion())
        continue;
    }

    // writeB must postdominate writeA.
    if (!postDominanceInfo.postDominates(targetB, targetA))
      continue;

    // There cannot be an operation which reads from memory between
    // the two writes.
    if (!hasNoInterveningEffect<MemoryEffects::Read>(
            targetA, writeB, writeB.getMemRef(), mayAlias))
      continue;

    if (targetA == writeA && targetB != writeB) {
      auto ifOp = cast<AffineIfOp>(targetB);
      writeB->moveBefore(ifOp);

      auto builder = OpBuilder(ifOp);
      builder.setInsertionPoint(writeB);
      auto select = hls::AffineSelectOp::create(
          builder, ifOp.getLoc(), ifOp.getIntegerSet(), ifOp.getOperands(),
          writeB.getValueToStore(), writeA.getValueToStore());
      ifOp->erase();

      writeB.getValueToStore().replaceUsesWithIf(
          select, [&](OpOperand &use) { return use.getOwner() == writeB; });
    }
    opsToErase.push_back(targetA);
    break;
  }

  if (llvm::all_of(memref.getUsers(), [&](Operation *ownerOp) {
        return isa<mlir::affine::AffineWriteOpInterface>(ownerOp) ||
               hasSingleEffect<MemoryEffects::Free>(ownerOp, memref);
      }))
    memrefsToErase.insert(memref);
}

// The load to load forwarding / redundant load elimination is similar to the
// store to load forwarding.
// loadA will be be replaced with loadB if:
// 1) loadA and loadB have mathematically equivalent affine access functions.
// 2) loadB dominates loadA.
// 3) There is no write between loadA and loadB.
static void loadCSE(mlir::affine::AffineReadOpInterface loadA,
                    SmallVectorImpl<Operation *> &loadOpsToErase,
                    DominanceInfo &domInfo,
                    llvm::function_ref<bool(Value, Value)> mayAlias) {
  SmallVector<mlir::affine::AffineReadOpInterface, 4> loadCandidates;
  for (auto *user : loadA.getMemRef().getUsers()) {
    auto loadB = dyn_cast<mlir::affine::AffineReadOpInterface>(user);
    if (!loadB || loadB == loadA)
      continue;

    MemRefAccess srcAccess(loadB);
    MemRefAccess destAccess(loadA);

    // 1. The accesses have to be to the same location.
    if (srcAccess != destAccess) {
      continue;
    }

    // 2. The store has to dominate the load op to be candidate.
    if (!domInfo.dominates(loadB, loadA))
      continue;

    // 3. There is no write between loadA and loadB.
    if (!hasNoInterveningEffect<MemoryEffects::Write>(
            loadB.getOperation(), loadA, loadA.getMemRef(), mayAlias))
      continue;

    // Check if two values have the same shape. This is needed for affine vector
    // loads.
    if (loadB.getValue().getType() != loadA.getValue().getType())
      continue;

    loadCandidates.push_back(loadB);
  }

  // Of the legal load candidates, use the one that dominates all others
  // to minimize the subsequent need to loadCSE
  Value loadB;
  for (mlir::affine::AffineReadOpInterface option : loadCandidates) {
    if (llvm::all_of(loadCandidates,
                     [&](mlir::affine::AffineReadOpInterface depStore) {
                       return depStore == option ||
                              domInfo.dominates(option.getOperation(),
                                                depStore.getOperation());
                     })) {
      loadB = option.getValue();
      break;
    }
  }

  if (loadB) {
    loadA.getValue().replaceAllUsesWith(loadB);
    loadOpsToErase.push_back(loadA);
  }
}

// Forward only equivalent affine accesses when the source dominates and no
// intervening aliasing write exists. Store removal is limited to buffers with
// no remaining use other than deallocation.
static bool applyAffineStoreForward(func::FuncOp func) {
  DominanceInfo domInfo(func);
  PostDominanceInfo postDomInfo(func);

  // Answers may-alias queries for the intervening-effect checks; anything
  // the analysis cannot rule out is treated as aliasing.
  LocalAliasAnalysis aliasAnalysis;
  auto mayAlias = [&](Value val1, Value val2) -> bool {
    return !aliasAnalysis.alias(val1, val2).isNo();
  };

  // Load op's whose results were replaced by those forwarded from stores.
  SmallVector<Operation *, 8> opsToErase;

  // A list of memref's that are potentially dead / could be eliminated.
  SmallPtrSet<Value, 4> memrefsToErase;

  // Walk all load's and perform store to load forwarding.
  func.walk([&](mlir::affine::AffineReadOpInterface loadOp) {
    auto currentLoadOp = loadOp;
    auto newLoadOp = mlir::affine::AffineReadOpInterface();
    while (1) {
      newLoadOp = forwardStoreToLoad(currentLoadOp, opsToErase, memrefsToErase,
                                     domInfo, mayAlias);
      // If the current load op is erased or failed to transform, break.
      if (!newLoadOp || newLoadOp == currentLoadOp)
        break;
      currentLoadOp = newLoadOp;
    }
    if (newLoadOp)
      loadCSE(newLoadOp, opsToErase, domInfo, mayAlias);
  });

  // Erase all load op's whose results were replaced with store fwd'ed ones.
  for (auto *op : opsToErase)
    op->erase();
  opsToErase.clear();

  // Walk all store's and perform unused store elimination
  func.walk([&](mlir::affine::AffineWriteOpInterface storeOp) {
    findUnusedStore(storeOp, opsToErase, memrefsToErase, postDomInfo, mayAlias);
  });
  // Erase all store op's which don't impact the program
  for (auto *op : opsToErase)
    op->erase();

  // Check if the store fwd'ed memrefs are now left with only stores and
  // deallocs and can thus be completely deleted. Note: the canonicalize pass
  // could do this as well; it happens here because the pass already collected
  // these anyway.
  for (auto memref : memrefsToErase) {
    // If the memref hasn't been locally alloc'ed, skip. (A memref returned
    // by a side-effect-free call could also qualify, but none of the
    // pipelines produce one here.)
    Operation *defOp = memref.getDefiningOp();
    if (!defOp || !hasSingleEffect<MemoryEffects::Allocate>(defOp, memref))
      continue;
    if (llvm::any_of(memref.getUsers(), [&](Operation *ownerOp) {
          return !isa<mlir::affine::AffineWriteOpInterface>(ownerOp) &&
                 !hasSingleEffect<MemoryEffects::Free>(ownerOp, memref);
        }))
      continue;

    // Erase all stores, the dealloc, and the alloc on the memref.
    for (auto *user : llvm::make_early_inc_range(memref.getUsers()))
      user->erase();
    defOp->erase();
  }
  return true;
}

namespace {
struct AffineStoreForward
    : public sar::impl::AffineStoreForwardBase<AffineStoreForward> {
  void runOnOperation() override { applyAffineStoreForward(getOperation()); }
};
} // namespace

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
std::unique_ptr<Pass> sar::createAffineStoreForwardPass() {
  return std::make_unique<AffineStoreForward>();
}
