//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "sar/Dialect/HLS/IR/Utils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

//===----------------------------------------------------------------------===//
// HLS dialect
//===----------------------------------------------------------------------===//

void HLSDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "sar/Dialect/HLS/IR/HLSOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "sar/Dialect/HLS/IR/HLSOpsAttributes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "sar/Dialect/HLS/IR/HLSOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DispatchOp
//===----------------------------------------------------------------------===//

namespace {
template <typename OpType>
struct SimplifyDispatchOrTaskOutputs : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto yield = op.getYieldOp();
    bool hasUnusedPort = false;

    // Identify output values that are used.
    SmallVector<Value, 4> usedOutputs;
    SmallVector<Value, 4> usedResults;
    for (auto result : op.getResults())
      if (result.use_empty()) {
        hasUnusedPort = true;
      } else {
        usedOutputs.push_back(yield.getOperand(result.getResultNumber()));
        usedResults.push_back(result);
      }

    // Construct new op with only used outputs.
    if (hasUnusedPort) {
      rewriter.setInsertionPoint(yield);
      rewriter.replaceOpWithNewOp<YieldOp>(yield, usedOutputs);

      rewriter.setInsertionPoint(op);
      auto newTask =
          OpType::create(rewriter, op.getLoc(), ValueRange(usedOutputs));
      rewriter.inlineRegionBefore(op.getBody(), newTask.getBody(),
                                  newTask.getBody().end());
      for (auto t : llvm::zip(usedResults, newTask.getResults()))
        std::get<0>(t).replaceAllUsesWith(std::get<1>(t));

      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
template <typename OpType>
struct InlineDispatchOrTask : public OpRewritePattern<OpType> {
  InlineDispatchOrTask(MLIRContext *context,
                       llvm::function_ref<bool(OpType)> condition)
      : OpRewritePattern<OpType>(context), condition(condition) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (condition(op)) {
      auto &ops = op.getBody().front().getOperations();
      auto &parentOps = op->getBlock()->getOperations();
      parentOps.splice(op->getIterator(), ops, ops.begin(),
                       std::prev(ops.end()));
      rewriter.replaceOp(op, op.getYieldOp()->getOperands());
      return success();
    }
    return failure();
  }

private:
  llvm::function_ref<bool(OpType)> condition;
};
} // namespace

void DispatchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<SimplifyDispatchOrTaskOutputs<DispatchOp>>(context);
  results.add<InlineDispatchOrTask<DispatchOp>>(context, [](DispatchOp op) {
    return op.getOps<TaskOp>().empty() || llvm::hasSingleElement(op.getOps());
  });
}

LogicalResult DispatchOp::verify() {
  if (getResultTypes() != getYieldOp().getOperandTypes())
    return emitOpError("yield type doesn't align with result type");
  return success();
}

/// Get the terminator yield op.
YieldOp DispatchOp::getYieldOp() {
  return cast<YieldOp>(getBody().front().getTerminator());
}

//===----------------------------------------------------------------------===//
// TaskOp
//===----------------------------------------------------------------------===//

void TaskOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<SimplifyDispatchOrTaskOutputs<TaskOp>>(context);
  results.add<InlineDispatchOrTask<TaskOp>>(
      context, [](TaskOp op) { return llvm::hasSingleElement(op.getOps()); });
}

LogicalResult TaskOp::verify() {
  if (getResultTypes() != getYieldOp().getOperandTypes())
    return emitOpError("yield type doesn't align with result type");
  return success();
}

/// Get the parent dispatch op.
DispatchOp TaskOp::getDispatchOp() {
  return (*this)->getParentOfType<DispatchOp>();
}

/// Get the terminator yield op.
YieldOp TaskOp::getYieldOp() {
  return cast<YieldOp>(getBody().front().getTerminator());
}

bool TaskOp::isLivein(Value value) {
  auto liveins = Liveness(*this).getLiveIn(&(*this).getBody().front());
  return liveins.count(value);
}

SmallVector<Value> TaskOp::getLiveins() {
  auto liveins = Liveness(*this).getLiveIn(&(*this).getBody().front());
  // The liveness set iterates by pointer, which varies from run to run;
  // order by first use inside the body so callers see a stable sequence.
  SmallVector<Value> ordered;
  llvm::SmallPtrSet<Value, 16> seen;
  (*this).getBody().front().walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      if (liveins.contains(operand) && seen.insert(operand).second)
        ordered.push_back(operand);
  });
  for (Value livein : liveins)
    if (!seen.contains(livein))
      ordered.push_back(livein);
  return ordered;
}

SmallVector<Operation *> TaskOp::getLiveinUsers(Value livein) {
  assert(isLivein(livein) && "invalid livein");
  auto users = llvm::make_filter_range(livein.getUsers(), [&](Operation *user) {
    return (*this)->isAncestor(user);
  });
  return {users.begin(), users.end()};
}

//===----------------------------------------------------------------------===//
// ToStreamOp and ToValueOp
//===----------------------------------------------------------------------===//

LogicalResult ToStreamOp::verify() {
  if (getValue().getType() !=
      cast<StreamType>(getStream().getType()).getElementType())
    return emitOpError("value and stream type doesn't match");
  return success();
}

OpFoldResult ToStreamOp::fold(FoldAdaptor) {
  if (auto toValue = getValue().getDefiningOp<ToValueOp>())
    if (toValue.getStream().getType() == getType())
      return toValue.getStream();
  return {};
}

LogicalResult ToValueOp::verify() {
  if (getValue().getType() !=
      cast<StreamType>(getStream().getType()).getElementType())
    return emitOpError("value and stream type doesn't match");
  return success();
}

OpFoldResult ToValueOp::fold(FoldAdaptor) {
  if (auto toStream = getStream().getDefiningOp<ToStreamOp>())
    if (toStream.getValue().getType() == getType())
      return toStream.getValue();
  return {};
}

//===----------------------------------------------------------------------===//
// ScheduleOp
//===----------------------------------------------------------------------===//

namespace {
struct SimplifyScheduleOperands : public OpRewritePattern<ScheduleOp> {
  using OpRewritePattern<ScheduleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScheduleOp schedule,
                                PatternRewriter &rewriter) const override {
    bool hasUnusedPort = false;

    // Identify input values that are used.
    llvm::SmallDenseSet<BlockArgument, 4> unusedArgs;
    SmallVector<Value, 4> usedOperands;
    for (auto arg : schedule.getBody().getArguments())
      if (arg.use_empty()) {
        hasUnusedPort = true;
        unusedArgs.insert(arg);
      } else {
        usedOperands.push_back(schedule.getOperand(arg.getArgNumber()));
      }
    schedule.getBody().front().eraseArguments(
        [&](BlockArgument arg) { return unusedArgs.count(arg); });

    // Construct new schedule.
    if (hasUnusedPort) {
      rewriter.setInsertionPoint(schedule);
      auto newSchedule =
          ScheduleOp::create(rewriter, schedule.getLoc(), usedOperands);
      rewriter.inlineRegionBefore(schedule.getBody(), newSchedule.getBody(),
                                  newSchedule.getBody().end());
      rewriter.eraseOp(schedule);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
template <typename OpType>
struct InlineScheduleOrNode : public OpRewritePattern<OpType> {
  InlineScheduleOrNode(MLIRContext *context,
                       llvm::function_ref<bool(OpType)> condition)
      : OpRewritePattern<OpType>(context), condition(condition) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (condition(op)) {
      auto &ops = op.getBody().front().getOperations();
      auto &parentOps = op->getBlock()->getOperations();
      parentOps.splice(op->getIterator(), ops);

      for (auto t : llvm::zip(op.getBody().getArguments(), op.getOperands()))
        std::get<0>(t).replaceAllUsesWith(std::get<1>(t));
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }

private:
  llvm::function_ref<bool(OpType)> condition;
};
} // namespace

void ScheduleOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<SimplifyScheduleOperands>(context);
  results.add<InlineScheduleOrNode<ScheduleOp>>(
      context, [](ScheduleOp op) { return op.getOps<NodeOp>().empty(); });
}

LogicalResult ScheduleOp::verify() {
  if (getOperandTypes() != getBody().getArgumentTypes())
    return emitOpError("operand type doesn't align with argument type");

  if (getIsLegal())
    for (auto &op : getOps())
      if (!isa<NodeOp, BufferOp, ConstBufferOp, StreamOp>(op)) {
        auto diag = emitOpError("legal schedule has illegal ops:\n");
        diag.attachNote(op.getLoc())
            .append("see current op: ")
            .appendOp(op, OpPrintingFlags().printGenericOpForm());
        return diag;
      }
  return success();
}

void ScheduleOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto &operand : getOperation()->getOpOperands())
    if (isa<MemRefType, StreamType>(operand.get().getType())) {
      effects.emplace_back(MemoryEffects::Read::get(), &operand,
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &operand,
                           SideEffects::DefaultResource::get());
    }
}

/// FIXME: Check whether the schedule is dependence free.
bool ScheduleOp::isDependenceFree() {
  if (auto loop = dyn_cast<mlir::affine::AffineForOp>((*this)->getParentOp()))
    return hasParallelAttr(loop);
  return isa<func::FuncOp>((*this)->getParentOp());
}

/// Update the signature of the schedule op recursively.
void ScheduleOp::updateSignatureRecursively() {
  for (auto [operand, arg] : llvm::zip(getOperands(), getBody().getArguments()))
    arg.setType(operand.getType());
  for (auto node : getOps<NodeOp>())
    node.updateSignatureRecursively();
}

//===----------------------------------------------------------------------===//
// NodeOp
//===----------------------------------------------------------------------===//

namespace {
struct SimplifyNodeIOs : public OpRewritePattern<NodeOp> {
  using OpRewritePattern<NodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NodeOp node,
                                PatternRewriter &rewriter) const override {
    bool hasUnusedPort = false;

    // Identify input values that are used.
    llvm::SmallDenseSet<BlockArgument, 4> unusedArgs;
    SmallVector<Value, 4> usedInputs;
    SmallVector<int32_t, 4> usedInputTaps;
    SmallVector<Value, 4> usedOutputs;
    SmallVector<Value, 4> usedParams;
    for (auto arg : node.getBody().getArguments())
      if (arg.use_empty()) {
        hasUnusedPort = true;
        unusedArgs.insert(arg);
      } else {
        auto idx = arg.getArgNumber();
        if (node.getOperandKind(idx) == OperandKind::INPUT) {
          usedInputs.push_back(node.getOperand(idx));
          usedInputTaps.push_back(node.getInputTap(idx));
        } else if (node.getOperandKind(idx) == OperandKind::OUTPUT)
          usedOutputs.push_back(node.getOperand(idx));
        else
          usedParams.push_back(node.getOperand(idx));
      }
    node.getBody().front().eraseArguments(
        [&](BlockArgument arg) { return unusedArgs.count(arg); });

    // Construct new dataflow node.
    if (hasUnusedPort) {
      rewriter.setInsertionPoint(node);
      auto newNode = NodeOp::create(
          rewriter, node.getLoc(), usedInputs, usedOutputs, usedParams,
          rewriter.getI32ArrayAttr(usedInputTaps), node.getLevelAttr());
      rewriter.inlineRegionBefore(node.getBody(), newNode.getBody(),
                                  newNode.getBody().end());
      rewriter.eraseOp(node);
      return success();
    }
    return failure();
  }
};
} // namespace

void NodeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<SimplifyNodeIOs>(context);
  results.add<InlineScheduleOrNode<NodeOp>>(context,
                                            [](NodeOp op) { return false; });
}

LogicalResult NodeOp::verify() {
  if (getOperandTypes() != getBody().getArgumentTypes())
    return emitOpError("operand type doesn't align with argument type");

  if (llvm::any_of(getParams(), [](Value param) {
        return isa<MemRefType, StreamType>(param.getType());
      }))
    return emitOpError("node params should not be memref or stream typed");

  if (getInputs().size() != getInputTaps().size())
    return emitOpError("number of node inputs and input taps are not aligned");
  for (auto t : llvm::zip(getInputs(), getInputTapsAsInt())) {
    auto depth = getBufferDepth(std::get<0>(t));
    auto inputTap = (unsigned)std::get<1>(t);
    if (depth <= inputTap) {
      auto diag = emitOpError("node input tap is larger than buffer depth, ");
      diag << "input tap: " << inputTap << ", depth: " << depth;
      return diag;
    }
  }

  for (auto inputArg : getInputArgs())
    if (llvm::any_of(inputArg.getUses(), isWritten)) {
      auto diag = emitOpError("input operand ");
      diag << inputArg << " is written";
      return diag;
    }

  for (auto outputArg : getOutputArgs())
    if (!llvm::any_of(outputArg.getUses(), isWritten)) {
      auto diag = emitOpError("output operand ");
      diag << outputArg << " is not written";
      return diag;
    }

  if (getScheduleOp().getIsLegal()) {
    if (!getLevel())
      return emitOpError("node is not scheduled");

    for (auto output : getOutputs()) {
      // DRAM buffer is not considered - the dependencies associated with them
      // are handled later by tokens.
      if (isExtBuffer(output))
        continue;

      if (getDependentConsumers(output, *this).size() > 1 ||
          getProducers(output).size() > 1) {
        auto diag = emitOpError(
            "legal schedule violates single-consumer or single-producer, ");
        diag << "see current buffer: " << output << "\n";
        for (auto user : output.getUsers())
          diag.attachNote(user->getLoc())
              .append("see current buffer user: ")
              .appendOp(*user, OpPrintingFlags().printGenericOpForm());
        return diag;
      }
    }
  }
  return success();
}

void NodeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto numInputs = getInputs().size();
  for (auto &operand : getOperation()->getOpOperands()) {
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
    if (operand.getOperandNumber() >= numInputs)
      effects.emplace_back(MemoryEffects::Write::get(), &operand,
                           SideEffects::DefaultResource::get());
  }
}

/// Get the parent schedule op.
ScheduleOp NodeOp::getScheduleOp() {
  return (*this)->getParentOfType<ScheduleOp>();
}

/// Get input taps.
void NodeOp::setInputTap(unsigned idx, unsigned tap) {
  SmallVector<int32_t> newInputTaps(llvm::map_range(
      getInputTapsAsInt(), [](unsigned a) { return (int32_t)a; }));
  newInputTaps[idx] = tap;
  Builder builder(getContext());
  setInputTapsAttr(builder.getI32ArrayAttr(newInputTaps));
}
unsigned NodeOp::getInputTap(unsigned idx) {
  return cast<IntegerAttr>(getInputTaps()[idx]).getInt();
}
SmallVector<unsigned> NodeOp::getInputTapsAsInt() {
  auto array = llvm::map_range(getInputTaps(), [](Attribute attr) {
    return cast<IntegerAttr>(attr).getInt();
  });
  return {array.begin(), array.end()};
}

/// Return the number of inputs, outputs, and params.
unsigned NodeOp::getNumInputs() {
  return getODSOperandIndexAndLength(0).second;
}
unsigned NodeOp::getNumOutputs() {
  return getODSOperandIndexAndLength(1).second;
}
unsigned NodeOp::getNumParams() {
  return getODSOperandIndexAndLength(2).second;
}

/// Get the type of operand: input, output, or param.
OperandKind NodeOp::getOperandKind(OpOperand &operand) {
  assert(operand.getOwner() == *this && "invalid operand");
  return getOperandKind(operand.getOperandNumber());
}
OperandKind NodeOp::getOperandKind(unsigned operandIdx) {
  if (operandIdx >= getODSOperandIndexAndLength(2).first)
    return OperandKind::PARAM;
  else if (operandIdx >= getODSOperandIndexAndLength(1).first)
    return OperandKind::OUTPUT;
  else
    return OperandKind::INPUT;
}

/// Get the input, output, and param arguments.
iterator_range<Block::args_iterator> NodeOp::getInputArgs() {
  auto range = getODSOperandIndexAndLength(0);
  return {std::next(getBody().args_begin(), range.first),
          std::next(getBody().args_begin(), range.first + range.second)};
}
iterator_range<Block::args_iterator> NodeOp::getOutputArgs() {
  auto range = getODSOperandIndexAndLength(1);
  return {std::next(getBody().args_begin(), range.first),
          std::next(getBody().args_begin(), range.first + range.second)};
}
iterator_range<Block::args_iterator> NodeOp::getParamArgs() {
  auto range = getODSOperandIndexAndLength(2);
  return {std::next(getBody().args_begin(), range.first),
          std::next(getBody().args_begin(), range.first + range.second)};
}

bool NodeOp::isLivein(Value value) {
  return isa<BlockArgument>(value) &&
         value.getParentRegion() == &(*this).getBody();
}

SmallVector<Value> NodeOp::getLiveins() {
  auto args = (*this).getBody().getArguments();
  return {args.begin(), args.end()};
}

SmallVector<Operation *> NodeOp::getLiveinUsers(Value livein) {
  assert(isLivein(livein) && "invalid livein");
  auto users = livein.getUsers();
  return {users.begin(), users.end()};
}

/// Update the signature of the node op recursively.
void NodeOp::updateSignatureRecursively() {
  llvm::SmallDenseSet<ScheduleOp> schedules;
  for (auto [operand, arg] :
       llvm::zip(getOperands(), getBody().getArguments())) {
    arg.setType(operand.getType());
    for (auto user : arg.getUsers())
      if (auto schedule = dyn_cast<ScheduleOp>(user))
        schedules.insert(schedule);
  }
  // TODO: How to traverse all schedule ops?
  for (auto schedule : schedules)
    schedule.updateSignatureRecursively();
}

//===----------------------------------------------------------------------===//
// BufferOp and ConstBufferOp
//===----------------------------------------------------------------------===//

namespace {
struct FlattenReadOnlyBuffer : public OpRewritePattern<BufferOp> {
  using OpRewritePattern<BufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferOp buffer,
                                PatternRewriter &rewriter) const override {
    if (buffer.getInitValue() &&
        llvm::all_of(buffer->getUsers(), [](Operation *user) {
          return isa<mlir::affine::AffineLoadOp>(user);
        })) {
      auto initValue = buffer.getInitValue().value();
      auto constant =
          arith::ConstantOp::create(rewriter, buffer.getLoc(), initValue);
      for (auto user : buffer->getUsers())
        rewriter.replaceOp(user, constant.getResult());
      rewriter.eraseOp(buffer);
      return success();
    }
    return failure();
  }
};
} // namespace

static NodeOp sinkBufferIntoNode(NodeOp node, BufferOp buffer,
                                 PatternRewriter &rewriter) {
  assert(node->getParentRegion() == buffer->getParentRegion() &&
         "node and buffer is not at the same region");
  SmallVector<Value> inputs;
  SmallVector<unsigned, 8> inputTaps;
  SmallVector<Value> outputs;
  llvm::BitVector eraseIndices;

  for (auto input : llvm::enumerate(node.getInputs())) {
    if (input.value() != buffer) {
      inputs.push_back(input.value());
      inputTaps.push_back(node.getInputTap(input.index()));
      eraseIndices.push_back(false);
    } else {
      auto arg = node.getBody().getArgument(input.index());
      arg.replaceAllUsesWith(buffer);
      eraseIndices.push_back(true);
    }
  }
  for (auto output : llvm::enumerate(node.getOutputs())) {
    if (output.value() != buffer) {
      outputs.push_back(output.value());
      eraseIndices.push_back(false);
    } else {
      auto arg =
          node.getBody().getArgument(node.getNumInputs() + output.index());
      arg.replaceAllUsesWith(buffer);
      eraseIndices.push_back(true);
    }
  }
  for (unsigned i = 0; i < node.getNumParams(); ++i)
    eraseIndices.push_back(false);

  auto &nodeBlock = node.getBody().front();
  buffer->moveBefore(&nodeBlock.front());
  nodeBlock.eraseArguments(eraseIndices);

  rewriter.setInsertionPointAfter(node);
  auto newNode =
      NodeOp::create(rewriter, node.getLoc(), inputs, outputs, node.getParams(),
                     inputTaps, node.getLevelAttr());
  rewriter.inlineRegionBefore(node.getBody(), newNode.getBody(),
                              newNode.getBody().begin());
  rewriter.eraseOp(node);
  return newNode;
}

static ScheduleOp sinkBufferIntoSchedule(ScheduleOp schedule, BufferOp buffer,
                                         PatternRewriter &rewriter) {
  assert(schedule->getParentRegion() == buffer->getParentRegion() &&
         "node and buffer is not at the same region");
  SmallVector<Value> operands;
  llvm::BitVector eraseIndices;

  for (auto operand : llvm::enumerate(schedule.getOperands())) {
    if (operand.value() != buffer) {
      operands.push_back(operand.value());
      eraseIndices.push_back(false);
    } else
      eraseIndices.push_back(true);
  }

  auto &scheduleBlock = schedule.getBody().front();
  buffer->moveBefore(&scheduleBlock.front());
  scheduleBlock.eraseArguments(eraseIndices);

  rewriter.setInsertionPointAfter(schedule);
  auto newSchedule = ScheduleOp::create(rewriter, schedule.getLoc(), operands,
                                        schedule.getIsLegalAttr());
  rewriter.inlineRegionBefore(schedule.getBody(), newSchedule.getBody(),
                              newSchedule.getBody().begin());
  rewriter.eraseOp(schedule);
  return newSchedule;
}

namespace {
struct SinkInternalBuffer : public OpRewritePattern<BufferOp> {
  using OpRewritePattern<BufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferOp buffer,
                                PatternRewriter &rewriter) const override {
    if (!isExtBuffer(buffer) && llvm::hasSingleElement(buffer->getUsers())) {
      auto user = *buffer->getUsers().begin();

      // Sink the buffer into the node or schedule user.
      if (user->getParentRegion() == buffer->getParentRegion() &&
          isa<NodeOp, ScheduleOp>(user)) {
        if (auto node = dyn_cast<NodeOp>(user))
          sinkBufferIntoNode(node, buffer, rewriter);
        else if (auto schedule = dyn_cast<ScheduleOp>(user))
          sinkBufferIntoSchedule(schedule, buffer, rewriter);
        return success();
      }
    }
    return failure();
  }
};
} // namespace

void BufferOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<FlattenReadOnlyBuffer>(context);
  results.add<SinkInternalBuffer>(context);
}

LogicalResult BufferOp::verify() {
  if (auto initValue = getInitValue())
    if (initValue.value().getType() != getType().getElementType())
      return emitOpError("initial value's type doesn't align with memref type");

  if (isExtBuffer(*this)) {
    if (auto node = dyn_cast<NodeOp>((*this)->getParentOp()))
      return emitOpError("external buffer should not be placed in node");
    if (auto schedule = dyn_cast<ScheduleOp>((*this)->getParentOp()))
      if (!isa<func::FuncOp>(schedule->getParentOp()))
        return emitOpError("external buffer must be placed in top schedule");
  }
  return success();
}

int32_t BufferOp::getBufferDepth() { return getDepth(); }
std::optional<TypedAttr> BufferOp::getBufferInitValue() {
  return getInitValue();
}

void BufferOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       cast<OpResult>(getMemref()),
                       SideEffects::DefaultResource::get());
}

int32_t ConstBufferOp::getBufferDepth() { return 1; }
std::optional<TypedAttr> ConstBufferOp::getBufferInitValue() {
  return std::optional<TypedAttr>();
}

LogicalResult ConstBufferOp::verify() {
  if (llvm::any_of((*this)->getUses(), isWritten))
    return emitOpError("const buffer cannot be written");

  auto memrefType = getType();
  auto attrType = cast<TensorType>(getValue().getType());
  if (memrefType.getElementType() != attrType.getElementType())
    return emitOpError("element type mismatch");
  if (memrefType.getShape() != attrType.getShape())
    return emitOpError("shape mismatch");
  return success();
}

void ConstBufferOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       cast<OpResult>(getMemref()),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// StreamOp, StreamReadOp, and StreamWriteOp
//===----------------------------------------------------------------------===//

LogicalResult StreamOp::verify() {
  if (getDepth() != cast<StreamType>(getChannel().getType()).getDepth())
    return emitOpError("stream channel depth is not aligned");
  return success();
}

void StreamOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       cast<OpResult>(getChannel()),
                       SideEffects::DefaultResource::get());
}

LogicalResult StreamReadOp::verify() {
  if (getResult())
    if (cast<StreamType>(getChannel().getType()).getElementType() !=
        getResult().getType())
      return emitOpError("result type doesn't align with channel type");
  return success();
}

LogicalResult StreamWriteOp::verify() {
  if (cast<StreamType>(getChannel().getType()).getElementType() !=
      getValue().getType())
    return emitOpError("value type doesn't align with channel type");
  return success();
}

//===----------------------------------------------------------------------===//
// AxiBundleOp, AxiPortOp, and AxiPackOp
//===----------------------------------------------------------------------===//

LogicalResult AxiType::verify(function_ref<InFlightDiagnostic()> emitError,
                              Type elementType) {
  // TODO: Support AxiLite type for scalar values.
  if (!isa<MemRefType, StreamType>(elementType))
    return emitError() << "AXI element type must be a memref or stream";
  return success();
}

Type AxiType::getDataType() {
  if (auto memrefType = dyn_cast<MemRefType>(getElementType()))
    return memrefType.getElementType();
  else if (auto streamType = dyn_cast<StreamType>(getElementType()))
    return streamType.getElementType();
  else
    llvm_unreachable("AXI element type must be a memref or stream");
  return Type();
}

LogicalResult AxiPortOp::verify() {
  if (!isa<BlockArgument>(getAxi()))
    return emitOpError("axi must be block arguments");

  if (getAxiType().getElementType() != getElement().getType())
    return emitOpError("axi type doesn't align with element type");

  if (getAxiType().getDataType() != getBundleType().getDataType())
    return emitOpError("axi type doesn't align with bundle type");

  return TypeSwitch<Type, LogicalResult>(getAxiType().getElementType())
      .Case<MemRefType>([&](auto type) -> LogicalResult {
        if (getBundleType().getKind() != AxiKind::MM)
          return emitOpError("memref type must be mapped to AXI-MM");
        return success();
      })
      .Case<StreamType>([&](auto type) -> LogicalResult {
        if (getBundleType().getKind() != AxiKind::STREAM)
          return emitOpError("stream type must be mapped to AXI-STREAM");
        return success();
      })
      .Default([&](auto type) -> LogicalResult {
        llvm_unreachable("AXI element type must be a memref or stream");
      });
}

LogicalResult AxiPackOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// Affine SelectOp
//===----------------------------------------------------------------------===//

void AffineSelectOp::build(OpBuilder &builder, OperationState &result,
                           IntegerSet set, ValueRange args, Value trueValue,
                           Value falseValue) {
  assert(trueValue.getType() == falseValue.getType() &&
         "true and false must have the same type");
  result.addTypes(trueValue.getType());
  result.addOperands(args);
  result.addOperands(trueValue);
  result.addOperands(falseValue);
  result.addAttribute(getConditionAttrStrName(), IntegerSetAttr::get(set));
}

namespace {
struct AlwaysTrueOrFalseSelect : public OpRewritePattern<AffineSelectOp> {
  using OpRewritePattern<AffineSelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineSelectOp op,
                                PatternRewriter &rewriter) const override {
    auto set = op.getIntegerSet();
    bool alwaysTrue = false;
    bool alwaysFalse = false;

    if (set.isEmptyIntegerSet())
      alwaysFalse = true;

    else if (set.getNumInputs() == 0) {
      SmallVector<bool, 4> flagList;
      for (auto expr : llvm::enumerate(set.getConstraints())) {
        auto constValue = cast<AffineConstantExpr>(expr.value()).getValue();
        flagList.push_back(set.isEq(expr.index()) ? constValue == 0
                                                  : constValue >= 0);
      }
      alwaysTrue = llvm::all_of(flagList, [](bool flag) { return flag; });
      alwaysFalse = !alwaysTrue;

    } else {
      // Create the base constraints from the integer set attached to SelectOp.
      FlatAffineValueConstraints constrs(set);

      // Bind vars in the constraints to SelectOp args.
      auto args = SmallVector<Value, 4>(op.getArgs());
      constrs.setValues(0, constrs.getNumDimAndSymbolVars(), args);

      // Add induction variable constraints.
      for (auto arg : args)
        if (isAffineForInductionVar(arg))
          (void)constrs.addAffineForOpDomain(getForInductionVarOwner(arg));

      // Always false if there's no known solution for the constraints.
      alwaysFalse = constrs.isEmpty();
    }

    // Replace uses if always-false or true is proved.
    if (alwaysFalse)
      rewriter.replaceOp(op, op.getFalseValue());
    else if (alwaysTrue)
      rewriter.replaceOp(op, op.getTrueValue());
    else
      return failure();
    return success();
  }
};
} // namespace

void AffineSelectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<AlwaysTrueOrFalseSelect>(context);
}

/// Canonicalize an affine if op's conditional (integer set + operands).
OpFoldResult AffineSelectOp::fold(FoldAdaptor) {
  auto set = getIntegerSet();
  SmallVector<Value, 4> operands(getArgs());
  // Compose affine.apply chains into the set via the map-composition
  // API (composeSetAndOperands is no longer public upstream).
  if (llvm::any_of(operands, [](Value v) {
        return isa_and_present<affine::AffineApplyOp>(v.getDefiningOp());
      })) {
    auto map = AffineMap::get(set.getNumDims(), set.getNumSymbols(),
                              set.getConstraints(), set.getContext());
    affine::fullyComposeAffineMapAndOperands(&map, &operands);
    set = IntegerSet::get(map.getNumDims(), map.getNumSymbols(),
                          map.getResults(), set.getEqFlags());
  }
  canonicalizeSetAndOperands(&set, &operands);
  return {};
}

LogicalResult AffineSelectOp::verify() {
  // Verify that we have a condition attribute.
  // FIXME: This should be specified in the arguments list in ODS.
  auto conditionAttr =
      (*this)->getAttrOfType<IntegerSetAttr>(getConditionAttrStrName());
  if (!conditionAttr)
    return emitOpError("requires an integer set attribute named 'condition'");

  // Verify that there are enough operands for the condition.
  IntegerSet condition = conditionAttr.getValue();
  if (getArgs().size() != condition.getNumInputs())
    return emitOpError("operand count and condition integer set dimension and "
                       "symbol count must match");

  // Verify that the operands are valid dimension/symbols.
  unsigned opIt = 0;
  for (auto operand : getArgs()) {
    if (opIt++ < condition.getNumDims()) {
      if (!isValidDim(operand, getAffineScope(*this)))
        return emitOpError("operand cannot be used as a dimension id");
    } else if (!isValidSymbol(operand, getAffineScope(*this))) {
      return emitOpError("operand cannot be used as a symbol");
    }
  }
  return success();
}

ParseResult AffineSelectOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the condition attribute set.
  IntegerSetAttr conditionAttr;
  unsigned numDims;
  if (parser.parseAttribute(conditionAttr,
                            AffineSelectOp::getConditionAttrStrName(),
                            result.attributes) ||
      parseDimAndSymbolList(parser, result.operands, numDims))
    return failure();

  // Verify the condition operands.
  auto set = conditionAttr.getValue();
  if (set.getNumDims() != numDims)
    return parser.emitError(
        parser.getNameLoc(),
        "dim operand count and integer set dim count must match");
  if (numDims + set.getNumSymbols() != result.operands.size())
    return parser.emitError(
        parser.getNameLoc(),
        "symbol operand count and integer set symbol count must match");

  SmallVector<OpAsmParser::UnresolvedOperand, 4> values;
  SMLoc valuesLoc = parser.getCurrentLocation();
  Type resultType;
  if (parser.parseOperandList(values) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType))
    return failure();
  result.types.push_back(resultType);

  if (values.size() != 2)
    return parser.emitError(valuesLoc, "should only have two input values");
  if (parser.resolveOperands(values, {resultType, resultType}, valuesLoc,
                             result.operands))
    return failure();
  return success();
}

/// Prints `(dims)[symbols]`; the upstream helper is no longer public.
static void printDimAndSymbolList(Operation::operand_iterator begin,
                                  Operation::operand_iterator end,
                                  unsigned numDims, OpAsmPrinter &p) {
  OperandRange operands(begin, end);
  p << '(' << operands.take_front(numDims) << ')';
  if (operands.size() > numDims)
    p << '[' << operands.drop_front(numDims) << ']';
}

void AffineSelectOp::print(OpAsmPrinter &p) {
  auto conditionAttr =
      (*this)->getAttrOfType<IntegerSetAttr>(getConditionAttrStrName());
  p << " " << conditionAttr;
  printDimAndSymbolList(getArgs().begin(), getArgs().end(),
                        conditionAttr.getValue().getNumDims(), p);

  p << " ";
  p.printOperand(getTrueValue());
  p << ", ";
  p.printOperand(getFalseValue());
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/getConditionAttrStrName());
  p << " : ";
  p.printType(getType());
}

IntegerSet AffineSelectOp::getIntegerSet() {
  return (*this)
      ->getAttrOfType<IntegerSetAttr>(getConditionAttrStrName())
      .getValue();
}

void AffineSelectOp::setIntegerSet(IntegerSet newSet) {
  (*this)->setAttr(getConditionAttrStrName(), IntegerSetAttr::get(newSet));
}

void AffineSelectOp::setConditional(IntegerSet set, ValueRange operands) {
  setIntegerSet(set);
  getArgsMutable().assign(operands);
}

//===----------------------------------------------------------------------===//
// PartitionLayoutAttr
//===----------------------------------------------------------------------===//

AffineMap PartitionLayoutAttr::getAffineMap() const {
  auto b = Builder(getContext());
  assert(getKinds().size() == getFactors().size() &&
         "invalid partition layout");
  auto rank = getKinds().size();

  SmallVector<AffineExpr, 4> partitionIndices;
  SmallVector<AffineExpr, 4> addressIndices;

  // Compute the partition and address indices.
  for (unsigned dim = 0; dim < rank; ++dim) {
    auto kind = getKinds()[dim];
    auto factor = getFactors()[dim];

    if (kind == PartitionKind::NONE) {
      partitionIndices.push_back(b.getAffineConstantExpr(0));
      addressIndices.push_back(b.getAffineDimExpr(dim));

    } else if (kind == PartitionKind::CYCLIC) {
      partitionIndices.push_back(b.getAffineDimExpr(dim) % factor);
      addressIndices.push_back(b.getAffineDimExpr(dim).floorDiv(factor));

    } else if (kind == PartitionKind::BLOCK) {
      partitionIndices.push_back(b.getAffineDimExpr(dim).floorDiv(factor));
      addressIndices.push_back(b.getAffineDimExpr(dim) % factor);

    } else {
      partitionIndices.push_back(b.getAffineDimExpr(dim));
      addressIndices.push_back(b.getAffineConstantExpr(0));
    }
  }

  // Construct layout affine map.
  partitionIndices.append(addressIndices);
  return AffineMap::get(rank, 0, partitionIndices, getContext());
}

LogicalResult PartitionLayoutAttr::verifyLayout(
    ArrayRef<int64_t> shape,
    function_ref<InFlightDiagnostic()> emitError) const {
  if (shape.size() != getKinds().size() || shape.size() != getFactors().size())
    return emitError() << "number of memref dimensions must be equal to "
                          "number of partition kinds and factors";

  for (auto [size, kind, factor] : llvm::zip(shape, getKinds(), getFactors())) {
    if ((kind == PartitionKind::CYCLIC && size % factor != 0) ||
        (kind == PartitionKind::BLOCK && size % factor != 0))
      return emitError() << "cyclic or block partition factor must be a "
                            "divisor of memref dimension size";
    if (kind == PartitionKind::NONE && factor != 1)
      return emitError() << "none partition factor must be 1";
    if (kind == PartitionKind::COMPLETE && factor != size)
      return emitError() << "complete partition factor must be equal to "
                            "memref dimension size";
  }
  return success();
}

LogicalResult
PartitionLayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            ArrayRef<PartitionKind> kinds,
                            ArrayRef<int64_t> factors) {
  if (kinds.size() != factors.size())
    return emitError() << "number of partition kinds and factors must be equal";
  if (llvm::any_of(factors, [](int64_t factor) { return factor <= 0; }))
    return emitError() << "partition factor must be positive";
  return success();
}

/// The affine map of "block" partition needs array shape to be inferenced. For
/// example, if the partition factor is [2] and the shape of the array is [16],
/// the affine map should be (d0) -> (d0 / 8, d0 % 8), where 8 is equal to 16
/// / 2. However, as the shape information is not known at the time of attribute
/// construction, we can only encode factor [8] in the attribute instead of the
/// actual factor [2]. This method returns the actual partition factor with the
/// given array shape.
SmallVector<int64_t>
PartitionLayoutAttr::getActualFactors(ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 4> actualFactors;
  for (auto [size, kind, factor] : llvm::zip(shape, getKinds(), getFactors())) {
    if (kind == PartitionKind::BLOCK)
      actualFactors.push_back((size + factor - 1) / factor);
    else
      actualFactors.push_back(factor);
  }
  return actualFactors;
}

/// This method construct a PartitionLayoutAttr with the given partition kinds,
/// actual partition factors, and array shape.
PartitionLayoutAttr PartitionLayoutAttr::getWithActualFactors(
    MLIRContext *context, ArrayRef<PartitionKind> kinds,
    ArrayRef<int64_t> actualFactors, ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 4> factors;
  for (auto [size, kind, factor] : llvm::zip(shape, kinds, actualFactors)) {
    if (kind == PartitionKind::BLOCK)
      factors.push_back((size + factor - 1) / factor);
    else
      factors.push_back(factor);
  }
  return get(context, kinds, factors);
}

//===----------------------------------------------------------------------===//
// TileLayoutAttr
//===----------------------------------------------------------------------===//

AffineMap TileLayoutAttr::getAffineMap() const {
  auto b = Builder(getContext());

  SmallVector<AffineExpr> exprs;
  for (auto tileSize : llvm::enumerate(getTileShape()))
    exprs.push_back(
        b.getAffineDimExpr(tileSize.index()).floorDiv(tileSize.value()));

  for (auto [tileSize, vectorSize] :
       llvm::zip(llvm::enumerate(getTileShape()), getVectorShape()))
    exprs.push_back((b.getAffineDimExpr(tileSize.index()) % tileSize.value())
                        .floorDiv(vectorSize));

  for (auto vectorSize : llvm::enumerate(getVectorShape()))
    exprs.push_back(b.getAffineDimExpr(vectorSize.index()) %
                    vectorSize.value());

  // The verifier should have made sure that the number of memref dimensions is
  // equal to the number of tile shape dimensions.
  return AffineMap::get(getTileShape().size(), 0, exprs, getContext());
}

LogicalResult TileLayoutAttr::verifyLayout(
    ArrayRef<int64_t> shape,
    function_ref<InFlightDiagnostic()> emitError) const {
  if (shape.size() != getTileShape().size())
    return emitError() << "number of memref dimensions must be equal to number "
                          "of vector shape dimensions";
  for (auto [size, tileSize] : llvm::zip(shape, getTileShape()))
    if (size % tileSize != 0)
      return emitError() << "memref dimension size must be a multiple of "
                            "vector shape dimension size";
  return success();
}

LogicalResult
TileLayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<int64_t> tileShape,
                       ArrayRef<int64_t> vectorShape) {
  if (tileShape.size() != vectorShape.size())
    return emitError() << "number of dimensions in affine map must be equal to "
                          "number of vector shape dimensions";
  for (auto [tileSize, vectorSize] : llvm::zip(tileShape, vectorShape))
    if (tileSize % vectorSize != 0)
      return emitError() << "tile shape dimension size must be a multiple of "
                            "vector shape dimension size";
  return success();
}

//===----------------------------------------------------------------------===//
// Tile layout attribute utils.
//===----------------------------------------------------------------------===//

TileLayoutAttr hls::getTileLayout(Operation *op) {
  return op->getAttrOfType<TileLayoutAttr>("tile_layout");
}
void hls::setTileLayout(Operation *op, TileLayoutAttr tileLayout) {
  op->setAttr("tile_layout", tileLayout);
}
void hls::setTileLayout(Operation *op, ArrayRef<int64_t> tileShape,
                        ArrayRef<int64_t> vectorShape) {
  auto tileLayout =
      TileLayoutAttr::get(op->getContext(), tileShape, vectorShape);
  setTileLayout(op, tileLayout);
}
void hls::setTileLayout(Operation *op, ArrayRef<int64_t> tileShape) {
  auto tileLayout = TileLayoutAttr::get(op->getContext(), tileShape);
  setTileLayout(op, tileLayout);
}

TileLayoutAttr hls::getTileLayout(Value memref) {
  if (auto buffer = findBuffer(memref)) {
    if (auto bufferArg = dyn_cast<BlockArgument>(buffer)) {
      if (auto func =
              dyn_cast<func::FuncOp>(bufferArg.getOwner()->getParentOp()))
        return func.getArgAttrOfType<TileLayoutAttr>(bufferArg.getArgNumber(),
                                                     "hls.tile_layout");
    } else if (auto bufferOp = buffer.getDefiningOp<hls::BufferLikeInterface>())
      return getTileLayout(bufferOp);
  }
  return TileLayoutAttr();
}
void hls::setTileLayout(Value memref, TileLayoutAttr tileLayout) {
  if (auto buffer = findBuffer(memref)) {
    if (auto bufferArg = dyn_cast<BlockArgument>(buffer)) {
      if (auto func =
              dyn_cast<func::FuncOp>(bufferArg.getOwner()->getParentOp()))
        func.setArgAttr(bufferArg.getArgNumber(), "hls.tile_layout",
                        tileLayout);
    } else if (auto bufferOp = buffer.getDefiningOp<hls::BufferLikeInterface>())
      setTileLayout(bufferOp, tileLayout);
  }
}
void hls::setTileLayout(Value memref, ArrayRef<int64_t> tileShape,
                        ArrayRef<int64_t> vectorShape) {
  auto tileLayout =
      TileLayoutAttr::get(memref.getContext(), tileShape, vectorShape);
  setTileLayout(memref, tileLayout);
}
void hls::setTileLayout(Value memref, ArrayRef<int64_t> tileShape) {
  auto tileLayout = TileLayoutAttr::get(memref.getContext(), tileShape);
  setTileLayout(memref, tileLayout);
}

//===----------------------------------------------------------------------===//
// HLS resource and timing attributes
//===----------------------------------------------------------------------===//

/// Timing attribute utils.
TimingAttr hls::getTiming(Operation *op) {
  return op->getAttrOfType<TimingAttr>("timing");
}
void hls::setTiming(Operation *op, TimingAttr timing) {
  assert(timing.getBegin() <= timing.getEnd() && "invalid timing attribute");
  op->setAttr("timing", timing);
}
void hls::setTiming(Operation *op, int64_t begin, int64_t end, int64_t latency,
                    int64_t minII) {
  auto timing = TimingAttr::get(op->getContext(), begin, end, latency, minII);
  setTiming(op, timing);
}

/// Resource attribute utils.
ResourceAttr hls::getResource(Operation *op) {
  return op->getAttrOfType<ResourceAttr>("resource");
}
void hls::setResource(Operation *op, ResourceAttr resource) {
  op->setAttr("resource", resource);
}
void hls::setResource(Operation *op, int64_t lut, int64_t dsp, int64_t bram) {
  auto resource = ResourceAttr::get(op->getContext(), lut, dsp, bram);
  setResource(op, resource);
}

/// Loop information attribute utils.
LoopInfoAttr hls::getLoopInfo(Operation *op) {
  return op->getAttrOfType<LoopInfoAttr>("loop_info");
}
void hls::setLoopInfo(Operation *op, LoopInfoAttr loopInfo) {
  op->setAttr("loop_info", loopInfo);
}
void hls::setLoopInfo(Operation *op, int64_t flattenTripCount,
                      int64_t iterLatency, int64_t minII) {
  auto loopInfo =
      LoopInfoAttr::get(op->getContext(), flattenTripCount, iterLatency, minII);
  setLoopInfo(op, loopInfo);
}

//===----------------------------------------------------------------------===//
// HLS directive attributes
//===----------------------------------------------------------------------===//

/// Loop directives attribute utils.
LoopDirectiveAttr hls::getLoopDirective(Operation *op) {
  return op->getAttrOfType<LoopDirectiveAttr>("loop_directive");
}
void hls::setLoopDirective(Operation *op, LoopDirectiveAttr loopDirective) {
  op->setAttr("loop_directive", loopDirective);
}
void hls::setLoopDirective(Operation *op, bool pipeline, int64_t targetII,
                           bool dataflow, bool flatten) {
  auto loopDirective = LoopDirectiveAttr::get(op->getContext(), pipeline,
                                              targetII, dataflow, flatten);
  setLoopDirective(op, loopDirective);
}

/// Parrallel and point loop attribute utils.
void hls::setParallelAttr(Operation *op) {
  op->setAttr("parallel", UnitAttr::get(op->getContext()));
}
bool hls::hasParallelAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("parallel");
}
void hls::setPointAttr(Operation *op) {
  op->setAttr("point", UnitAttr::get(op->getContext()));
}
bool hls::hasPointAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("point");
}

/// Function directives attribute utils.
FuncDirectiveAttr hls::getFuncDirective(Operation *op) {
  return op->getAttrOfType<FuncDirectiveAttr>("func_directive");
}
void hls::setFuncDirective(Operation *op, FuncDirectiveAttr funcDirective) {
  op->setAttr("func_directive", funcDirective);
}
void hls::setFuncDirective(Operation *op, bool pipeline, int64_t targetInterval,
                           bool dataflow) {
  auto funcDirective = FuncDirectiveAttr::get(op->getContext(), pipeline,
                                              targetInterval, dataflow);
  setFuncDirective(op, funcDirective);
}

/// Top and runtime function attribute utils.
void hls::setTopFuncAttr(Operation *op) {
  op->setAttr("top_func", UnitAttr::get(op->getContext()));
}
bool hls::hasTopFuncAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("top_func");
}
void hls::setRuntimeAttr(Operation *op) {
  op->setAttr("runtime", UnitAttr::get(op->getContext()));
}
bool hls::hasRuntimeAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("runtime");
}

//===----------------------------------------------------------------------===//
// Include tablegen classes
//===----------------------------------------------------------------------===//

#include "sar/Dialect/HLS/IR/HLSOpsDialect.cpp.inc"
#include "sar/Dialect/HLS/IR/HLSOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "sar/Dialect/HLS/IR/HLSOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "sar/Dialect/HLS/IR/HLSOpsTypes.cpp.inc"

#include "sar/Dialect/HLS/IR/HLSOpsInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "sar/Dialect/HLS/IR/HLSOps.cpp.inc"
