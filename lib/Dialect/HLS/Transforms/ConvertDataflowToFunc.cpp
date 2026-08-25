//===- ConvertDataflowToFunc.cpp - outline dataflow nodes as functions ----===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_CONVERTDATAFLOWTOFUNC
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

namespace {
struct InlineSchedule : public OpRewritePattern<ScheduleOp> {
  using OpRewritePattern<ScheduleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScheduleOp schedule,
                                PatternRewriter &rewriter) const override {
    // A schedule still sitting inside a node waits for the node to become
    // a function: inlining it now would leave its legality nowhere to go
    // (the dataflow directive attaches to functions), and the node keeps
    // its hierarchy so it stays a function rather than being marked for
    // emitter inlining.
    if (isa<NodeOp>(schedule->getParentOp()))
      return failure();

    auto &scheduleOps = schedule.getBody().front().getOperations();
    auto &parentOps = schedule->getBlock()->getOperations();
    parentOps.splice(schedule->getIterator(), scheduleOps);

    for (auto t :
         llvm::zip(schedule.getBody().getArguments(), schedule.getOperands()))
      std::get<0>(t).replaceAllUsesWith(std::get<1>(t));

    if (schedule.getIsLegal()) {
      if (auto func = dyn_cast<func::FuncOp>(schedule->getParentOp()))
        setFuncDirective(func, /*pipeline=*/false, /*targetInterval=*/1,
                         /*dataflow=*/true);
      // A schedule inside a loop is not marked dataflow: conditional or
      // iterated dataflow regions violate the synthesis model and obstruct the
      // outer loop pipeline. Their nodes retain their own directives.
    }
    rewriter.eraseOp(schedule);
    return success();
  }
};
} // namespace

namespace {
struct ConvertNodeToFunc : public OpRewritePattern<NodeOp> {
  ConvertNodeToFunc(MLIRContext *context, StringRef prefix, unsigned &nodeIdx)
      : OpRewritePattern<NodeOp>(context), prefix(prefix), nodeIdx(nodeIdx) {}

  LogicalResult matchAndRewrite(NodeOp node,
                                PatternRewriter &rewriter) const override {
    // Create a new sub-function.
    rewriter.setInsertionPoint(node->getParentOfType<func::FuncOp>());
    auto subFunc = func::FuncOp::create(
        rewriter, node.getLoc(),
        prefix.str() + "_node" + std::to_string(nodeIdx++),
        rewriter.getFunctionType(node.getOperandTypes(), TypeRange()));

    // A node that is a single flat loop nest carries no dataflow region of
    // its own, so keeping it a function call would only add call overhead
    // in csim and another hierarchy level in the reports; mark it for the
    // emitter to inline. Nodes with hierarchy (nested schedules) stay
    // functions -- their region structure is what the pragma attaches to.
    bool hasCallsOrTokens =
        node.walk([&](Operation *operation) {
              return isa<func::CallOp, StreamReadOp, StreamWriteOp>(operation)
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            })
            .wasInterrupted();
    auto parentSchedule = node->getParentOfType<ScheduleOp>();
    bool isDataflowProcess = parentSchedule && parentSchedule.getIsLegal() &&
                             isa<func::FuncOp>(parentSchedule->getParentOp());
    if (!isDataflowProcess) {
      auto parentFunc = node->getParentOfType<func::FuncOp>();
      auto directive =
          parentFunc ? getFuncDirective(parentFunc) : FuncDirectiveAttr();
      isDataflowProcess = parentFunc && node->getParentOp() == parentFunc &&
                          directive && directive.getDataflow();
    }
    if (!isDataflowProcess && !node.hasHierarchy() && !hasCallsOrTokens &&
        llvm::hasSingleElement(node.getOps<AffineForOp>()))
      subFunc->setAttr("inline", rewriter.getUnitAttr());

    // Inline the contents of the dataflow node.
    rewriter.inlineRegionBefore(node.getBodyRegion(), subFunc.getBody(),
                                subFunc.end());
    rewriter.setInsertionPointToEnd(&subFunc.front());
    func::ReturnOp::create(rewriter, rewriter.getUnknownLoc());

    // Replace original with a function call.
    rewriter.setInsertionPoint(node);
    rewriter.replaceOpWithNewOp<func::CallOp>(node, subFunc,
                                              node.getOperands());
    return success();
  }

private:
  StringRef prefix;
  unsigned &nodeIdx;
};
} // namespace

namespace {
struct ConvertDataflowToFunc
    : public sar::impl::ConvertDataflowToFuncBase<ConvertDataflowToFunc> {
  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    // Converting a node creates a new function holding its body, and a
    // nested schedule waits in that body for its turn (the deferral
    // above). One sweep over the module therefore may leave work in the
    // functions it created; sweep until the dataflow ops are gone.
    auto countWork = [&] {
      unsigned count = 0;
      module.walk([&](Operation *op) { count += isa<ScheduleOp, NodeOp>(op); });
      return count;
    };
    while (unsigned before = countWork()) {
      for (auto func :
           llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
        bool hasWork = false;
        func.walk([&](Operation *op) {
          if (isa<ScheduleOp, NodeOp>(op))
            hasWork = true;
        });
        if (!hasWork)
          continue;
        unsigned nodeIdx = 0;
        mlir::RewritePatternSet patterns(context);
        patterns.add<InlineSchedule>(context);
        patterns.add<ConvertNodeToFunc>(context, func.getName(), nodeIdx);
        if (failed(applyPatternsGreedily(func, std::move(patterns))))
          return signalPassFailure();
      }
      unsigned after = countWork();
      if (after >= before) {
        module.emitError("dataflow-to-function conversion made no progress");
        return signalPassFailure();
      }
    }

    // Constant buffers materialize globals inside dataflow functions. Remove
    // only symbols with no remaining references; unrelated globals stay valid.
    for (auto global :
         llvm::make_early_inc_range(module.getOps<memref::GlobalOp>()))
      if (SymbolTable::symbolKnownUseEmpty(global, module))
        global.erase();
  }
};
} // namespace

std::unique_ptr<Pass> sar::createConvertDataflowToFuncPass() {
  return std::make_unique<ConvertDataflowToFunc>();
}
