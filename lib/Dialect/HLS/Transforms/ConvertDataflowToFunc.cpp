//===- ConvertDataflowToFunc.cpp - convert dataflow to func ---------------===//
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
struct SplitScheduleExternalBufferAccess : public OpRewritePattern<ScheduleOp> {
  using OpRewritePattern<ScheduleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScheduleOp schedule,
                                PatternRewriter &rewriter) const override {
    auto &scheduleBody = schedule.getBody();
    SmallVector<Value, 16> newOperands(schedule.getOperands());
    bool hasChanged = false;

    SmallVector<BlockArgument, 16> args(scheduleBody.getArguments());
    for (auto arg : args) {
      // Only external buffers shared by multiple nodes need splitting.
      auto uses = llvm::make_filter_range(arg.getUses(), [&](auto &use) {
        return isa<NodeOp>(use.getOwner());
      });
      if (!isExtBuffer(arg) || uses.empty() || llvm::hasSingleElement(uses))
        continue;

      // Add an argument and input for each additional use.
      for (auto &use : llvm::make_early_inc_range(llvm::drop_begin(uses))) {
        newOperands.push_back(newOperands[arg.getArgNumber()]);
        auto newArg = scheduleBody.addArgument(arg.getType(), arg.getLoc());
        use.set(newArg);
        hasChanged = true;
      }
    }

    if (hasChanged) {
      auto newSchedule = ScheduleOp::create(
          rewriter, schedule.getLoc(), newOperands, schedule.getIsLegalAttr());
      rewriter.inlineRegionBefore(scheduleBody, newSchedule.getBody(),
                                  newSchedule.getBody().end());
      rewriter.eraseOp(schedule);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
struct SplitNodeExternalBufferAccess : public OpRewritePattern<NodeOp> {
  using OpRewritePattern<NodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NodeOp node,
                                PatternRewriter &rewriter) const override {
    auto &nodeBody = node.getBody();
    SmallVector<Value, 16> newInputs(node.getInputs());
    SmallVector<Value, 16> newOutputs(node.getOutputs());
    auto numInputs = node.getNumInputs();
    auto numOutputs = node.getNumOutputs();
    bool hasChanged = false;

    SmallVector<BlockArgument, 16> inputArgs(node.getInputArgs());
    SmallVector<BlockArgument, 16> outputArgs(node.getOutputArgs());
    for (auto arg : inputArgs) {
      // If the buffer is not an external buffer or has zero or one schedule
      // users, we have nothing to do.
      auto uses = llvm::make_filter_range(arg.getUses(), [&](auto &use) {
        return isa<ScheduleOp>(use.getOwner());
      });
      if (!isExtBuffer(arg) || uses.empty() || llvm::hasSingleElement(uses))
        continue;

      // Add a new argument and new input for each additional uses.
      for (auto &use : llvm::make_early_inc_range(llvm::drop_begin(uses))) {
        newInputs.push_back(newInputs[arg.getArgNumber()]);
        auto newArg =
            nodeBody.insertArgument(numInputs++, arg.getType(), arg.getLoc());
        use.set(newArg);
        hasChanged = true;
      }
    }

    for (auto arg : llvm::enumerate(outputArgs)) {
      // If the buffer is not an external buffer or has zero or one schedule
      // users, we have nothing to do.
      auto uses =
          llvm::make_filter_range(arg.value().getUses(), [&](auto &use) {
            return isa<ScheduleOp>(use.getOwner());
          });
      if (!isExtBuffer(arg.value()) || uses.empty() ||
          llvm::hasSingleElement(uses))
        continue;

      // Add a new argument and new input or output for each additional uses
      // apart from the first written use.
      bool outputFlag = false;
      for (auto &use : llvm::make_early_inc_range(uses)) {
        auto useIsWritten = isWritten(use);
        if (useIsWritten && !outputFlag) {
          outputFlag = true;
          continue;
        }
        if (useIsWritten) {
          newOutputs.push_back(newOutputs[arg.index()]);
          auto newArg = nodeBody.insertArgument(numInputs + numOutputs++,
                                                arg.value().getType(),
                                                arg.value().getLoc());
          use.set(newArg);
        } else {
          newInputs.push_back(newOutputs[arg.index()]);
          auto newArg = nodeBody.insertArgument(
              numInputs++, arg.value().getType(), arg.value().getLoc());
          use.set(newArg);
        }
        hasChanged = true;
      }
    }

    if (hasChanged) {
      auto newNode = NodeOp::create(rewriter, node.getLoc(), newInputs,
                                    newOutputs, node.getParams());
      rewriter.inlineRegionBefore(nodeBody, newNode.getBody(),
                                  newNode.getBody().end());
      rewriter.eraseOp(node);
      return success();
    }
    return failure();
  }
};
} // namespace

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
    if (!node.hasHierarchy() &&
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
  ConvertDataflowToFunc() = default;
  explicit ConvertDataflowToFunc(bool argSplitExternalAccess) {
    splitExternalAccess = argSplitExternalAccess;
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    if (splitExternalAccess.getValue()) {
      mlir::RewritePatternSet patterns(context);
      patterns.add<SplitScheduleExternalBufferAccess>(context);
      patterns.add<SplitNodeExternalBufferAccess>(context);
      (void)applyPatternsGreedily(module, std::move(patterns));
    }

    // Converting a node creates a new function holding its body, and a
    // nested schedule waits in that body for its turn (the deferral
    // above). One sweep over the module therefore may leave work in the
    // functions it created; sweep until the dataflow ops are gone.
    bool changed = true;
    while (changed) {
      changed = false;
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
        (void)applyPatternsGreedily(func, std::move(patterns));
        changed = true;
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

std::unique_ptr<Pass>
sar::createConvertDataflowToFuncPass(bool splitExternalAccess) {
  return std::make_unique<ConvertDataflowToFunc>(splitExternalAccess);
}
