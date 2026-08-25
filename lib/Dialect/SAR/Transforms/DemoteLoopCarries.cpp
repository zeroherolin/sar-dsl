//===- DemoteLoopCarries.cpp - Buffer-carried loops to side effects -------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// After bufferization a compiled loop (`sar.iterate`) is an `scf.for`
// carrying memrefs through `iter_args`. The HLS dataflow machinery only
// understands buffers written by side effect -- a task may not yield
// values -- so the carry is demoted: the body reads and writes the init
// buffer directly, a copy at the end of each iteration makes the yielded
// buffer current, and the loop stops carrying anything. The dead
// `iter_args` are then folded away by canonicalization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include "sar/Dialect/SAR/Transforms/Passes.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SARDEMOTELOOPCARRIES
#include "sar/Dialect/SAR/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;

namespace {

static LogicalResult demote(scf::ForOp loop) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  OpBuilder builder(yield);

  // A yield naming another carry's argument would need its value copied
  // before the first write-back clobbers it; stage such a swap through an
  // explicit buffer in the kernel instead.
  //
  // Every carry is checked before any is rewritten. The rewrite below
  // replaces a carry's argument with its buffer, so a later carry naming an
  // earlier one would no longer look like a region argument by the time its
  // own turn came -- the check would pass and the copies would silently be
  // ordered wrong, which is a wrong image rather than an error.
  for (auto [init, arg, index] :
       llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs(),
                 llvm::seq<unsigned>(0, loop.getNumResults()))) {
    if (!isa<MemRefType>(init.getType()))
      continue;
    Value yielded = yield.getOperand(index);
    if (yielded != arg && llvm::is_contained(loop.getRegionIterArgs(), yielded))
      return loop.emitOpError(
          "chained or swapped buffer carries are not supported; copy "
          "through a scratch buffer inside the loop instead");
  }

  for (auto [init, arg, result, index] : llvm::zip(
           loop.getInitArgs(), loop.getRegionIterArgs(), loop.getResults(),
           llvm::seq<unsigned>(0, loop.getNumResults()))) {
    if (!isa<MemRefType>(init.getType()))
      continue;
    Value yielded = yield.getOperand(index);
    // A function argument is a top-level port of the design, and the HLS
    // dataflow checker forbids a node that both reads and writes a port
    // ("cannot read as well as write over function parameter"). Iterate
    // in a local buffer primed from the port instead; an internal alloc
    // carries no such restriction and stays the zero-copy path.
    Value buffer = init;
    if (isa<BlockArgument>(init)) {
      OpBuilder prologue(loop);
      buffer = memref::AllocOp::create(prologue, loop.getLoc(),
                                       cast<MemRefType>(init.getType()));
      memref::CopyOp::create(prologue, loop.getLoc(), init, buffer);
    }
    if (yielded != arg)
      memref::CopyOp::create(builder, yield.getLoc(), yielded, buffer);
    // The body iterates in the (possibly localized) init buffer itself;
    // making the yield a pass-through lets canonicalization erase the
    // now-dead carry.
    arg.replaceAllUsesWith(buffer);
    result.replaceAllUsesWith(buffer);
    yield.setOperand(index, arg);
  }
  return success();
}

struct SARDemoteLoopCarries
    : public sar::impl::SARDemoteLoopCarriesBase<SARDemoteLoopCarries> {
  void runOnOperation() override {
    auto result = getOperation().walk([&](scf::ForOp loop) {
      return failed(demote(loop)) ? WalkResult::interrupt()
                                  : WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
