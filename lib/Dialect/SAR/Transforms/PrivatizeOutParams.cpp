//===- PrivatizeOutParams.cpp - privatize written-through result ports ----===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "sar/Dialect/SAR/Transforms/Passes.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SARPRIVATIZEOUTPARAMS
#include "sar/Dialect/SAR/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace sar;

/// Whether any use of `value` reads it (transitively through region ops'
/// block arguments is not needed here: out-params at this level are used
/// directly by affine/memref load/store, copies, or passed to scf.for as
/// iter operands -- treat anything but a direct store target as a read).
static bool hasReadUse(Value value) {
  for (auto &use : value.getUses()) {
    auto *owner = use.getOwner();
    if (auto store = dyn_cast<memref::StoreOp>(owner)) {
      if (store.getMemRef() == value && use.getOperandNumber() == 1)
        continue;
      return true;
    }
    if (owner->hasTrait<OpTrait::IsTerminator>())
      continue;
    if (auto copy = dyn_cast<memref::CopyOp>(owner)) {
      if (copy.getTarget() == value)
        continue;
      return true;
    }
    if (auto store = dyn_cast<affine::AffineStoreOp>(owner))
      if (&use == &store.getMemrefMutable())
        continue;
    return true;
  }
  return false;
}

/// The number of distinct top-level operations of `block` that write
/// `value` (each becomes its own dataflow node later).
static unsigned countTopLevelWriters(Value value, Block &block) {
  DenseSet<Operation *> writers;
  for (auto &use : value.getUses()) {
    Operation *op = use.getOwner();
    while (op->getBlock() != &block)
      op = op->getParentOp();
    writers.insert(op);
  }
  return writers.size();
}

namespace {
/// On the HLS path a kernel's result planes are out-parameters
/// (destination-passing). When the working precision equals the result
/// precision, bufferization happily computes *through* the result buffer:
/// several pipeline stages write and read it in place. On the CPU that is
/// an optimization; in a dataflow design it makes the top-level port a
/// multi-producer buffer, which forfeits `#pragma HLS dataflow` for the
/// whole design and sends synthesis down a single flat schedule.
///
/// This pass restores the single-writer contract: an out-param that is
/// read, or written by more than one top-level operation, is replaced by
/// a local buffer, and one final copy writes the port. The copy is a
/// full-plane sweep, but it turns the port back into a one-writer channel
/// and buys top-level dataflow for the rest of the design.
struct SARPrivatizeOutParams
    : public sar::impl::SARPrivatizeOutParamsBase<SARPrivatizeOutParams> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal() || func.getBody().getBlocks().size() != 1)
      return;

    // `sar.arg_names` names the kernel's own inputs; everything after them
    // was appended by buffer-results-to-out-params. Without the attribute
    // there is nothing to identify, so leave the function alone.
    auto argNames = func->getAttrOfType<ArrayAttr>("sar.arg_names");
    if (!argNames)
      return;
    unsigned numInputs = argNames.size();
    if (func.getNumArguments() <= numInputs)
      return;

    Block &entry = func.front();
    OpBuilder builder(func.getContext());
    for (auto arg : entry.getArguments().drop_front(numInputs)) {
      auto type = dyn_cast<MemRefType>(arg.getType());
      if (!type || arg.use_empty())
        continue;
      if (!hasReadUse(arg) && countTopLevelWriters(arg, entry) <= 1)
        continue;

      builder.setInsertionPointToStart(&entry);
      Value local =
          memref::AllocOp::create(builder, func.getLoc(), type).getResult();
      arg.replaceAllUsesWith(local);
      builder.setInsertionPoint(entry.getTerminator());
      memref::CopyOp::create(builder, func.getLoc(), local, arg);
    }
  }
};
} // namespace
