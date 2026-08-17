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

  for (auto [init, arg, result, index] : llvm::zip(
           loop.getInitArgs(), loop.getRegionIterArgs(), loop.getResults(),
           llvm::seq<unsigned>(0, loop.getNumResults()))) {
    if (!isa<MemRefType>(init.getType()))
      continue;
    Value yielded = yield.getOperand(index);
    // A yield naming another carry's argument would need its value copied
    // before the first write-back clobbers it; stage such a swap through
    // an explicit buffer in the kernel instead.
    if (yielded != arg && llvm::is_contained(loop.getRegionIterArgs(), yielded))
      return loop.emitOpError(
          "swapped buffer carries are not supported; copy through a "
          "scratch buffer inside the loop instead");
    if (yielded != arg)
      memref::CopyOp::create(builder, yield.getLoc(), yielded, init);
    // The body iterates in the init buffer itself; making the yield a
    // pass-through lets canonicalization erase the now-dead carry.
    arg.replaceAllUsesWith(init);
    result.replaceAllUsesWith(init);
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
