//===- Utils.cpp - HLS dialect utilities ----------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "sar/Dialect/HLS/IR/Utils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "sar/Support/HLSHints.h"

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

//===----------------------------------------------------------------------===//
// HLS dialect utils
//===----------------------------------------------------------------------===//

MemoryKind sar::getMemoryKind(MemRefType type) {
  if (auto memorySpace = type.getMemorySpace())
    if (auto kindAttr = dyn_cast<MemoryKindAttr>(memorySpace))
      return kindAttr.getValue();
  return MemoryKind::UNKNOWN;
}

static bool isDram(MemRefType type) {
  auto kind = getMemoryKind(type);
  return kind == MemoryKind::DRAM;
}
//===----------------------------------------------------------------------===//
// Dataflow utils
//===----------------------------------------------------------------------===//

/// Wrap the operations in the block with dispatch op.
DispatchOp sar::dispatchBlock(Block *block) {
  if (!block->getOps<DispatchOp>().empty() ||
      !isa<func::FuncOp, mlir::affine::AffineForOp>(block->getParentOp()))
    return DispatchOp();

  OpBuilder builder(block, block->begin());
  ValueRange returnValues(block->getTerminator()->getOperands());
  auto loc = builder.getUnknownLoc();
  auto dispatch = DispatchOp::create(builder, loc, returnValues);

  auto &dispatchBlock = dispatch.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&dispatchBlock);
  YieldOp::create(builder, loc, returnValues);

  auto &dispatchOps = dispatchBlock.getOperations();
  auto &parentOps = block->getOperations();
  dispatchOps.splice(dispatchBlock.begin(), parentOps,
                     std::next(parentOps.begin()), std::prev(parentOps.end()));
  block->getTerminator()->setOperands(dispatch.getResults());
  return dispatch;
}

/// Fuse the given operations into a new task, created before the first of
/// them and holding them in order. Always succeeds, even when the resulting
/// IR is invalid.
TaskOp sar::fuseOpsIntoTask(ArrayRef<Operation *> ops,
                            PatternRewriter &rewriter) {
  assert(!ops.empty() && "must fuse at least one op");
  llvm::SmallDenseSet<Operation *, 4> opsSet(ops.begin(), ops.end());

  // Collect output values. This is not sufficient and may lead to empty-used
  // outputs, which will be removed during canonicalization.
  llvm::SetVector<Value> outputValues;
  for (auto op : ops)
    for (auto result : op->getResults())
      if (llvm::any_of(result.getUsers(),
                       [&](Operation *user) { return !opsSet.count(user); }))
        outputValues.insert(result);

  auto loc = rewriter.getUnknownLoc();
  rewriter.setInsertionPoint(ops.front());
  auto task =
      TaskOp::create(rewriter, loc, ValueRange(outputValues.getArrayRef()));
  auto taskBlock = rewriter.createBlock(&task.getBody());

  rewriter.setInsertionPointToEnd(taskBlock);
  auto yield = YieldOp::create(rewriter, loc, outputValues.getArrayRef());
  for (auto op : ops)
    op->moveBefore(yield);

  // Replace external output uses with the task results.
  unsigned idx = 0;
  for (auto output : outputValues)
    output.replaceUsesWithIf(task.getResult(idx++), [&](OpOperand &use) {
      return !task->isProperAncestor(use.getOwner());
    });

  // Inline all sub-tasks.
  for (auto subTask : llvm::make_early_inc_range(task.getOps<TaskOp>())) {
    auto &subTaskOps = subTask.getBody().front().getOperations();
    auto &taskOps = task.getBody().front().getOperations();
    taskOps.splice(subTask->getIterator(), subTaskOps, subTaskOps.begin(),
                   std::prev(subTaskOps.end()));
    rewriter.replaceOp(subTask, subTask.getYieldOp()->getOperands());
  }
  return task;
}

/// Fuse multiple nodes into a new node.
NodeOp sar::fuseNodeOps(ArrayRef<NodeOp> nodes, PatternRewriter &rewriter) {
  assert((nodes.size() > 1) && "must fuse at least two nodes");

  llvm::SetVector<Value> inputs;
  llvm::SmallVector<unsigned, 8> inputTaps;
  llvm::SmallVector<Location, 8> inputLocs;
  llvm::SetVector<Value> outputs;
  llvm::SmallVector<Location, 8> outputLocs;
  llvm::SetVector<Value> params;
  llvm::SmallVector<Location, 8> paramLocs;

  for (auto node : nodes) {
    for (auto output : node.getOutputs())
      if (outputs.insert(output))
        outputLocs.push_back(output.getLoc());
    for (auto param : node.getParams())
      if (params.insert(param))
        paramLocs.push_back(param.getLoc());
  }
  for (auto node : nodes)
    for (auto input : llvm::enumerate(node.getInputs())) {
      if (outputs.count(input.value()))
        continue;
      if (inputs.insert(input.value())) {
        inputLocs.push_back(input.value().getLoc());
        inputTaps.push_back(node.getInputTap(input.index()));
      }
    }

  rewriter.setInsertionPointAfter(nodes.back());
  auto newNode =
      NodeOp::create(rewriter, rewriter.getUnknownLoc(), inputs.getArrayRef(),
                     outputs.getArrayRef(), params.getArrayRef(), inputTaps);
  auto block = rewriter.createBlock(&newNode.getBody());
  block->addArguments(ValueRange(inputs.getArrayRef()), inputLocs);
  block->addArguments(ValueRange(outputs.getArrayRef()), outputLocs);
  block->addArguments(ValueRange(params.getArrayRef()), paramLocs);

  for (auto node : nodes) {
    auto &nodeOps = node.getBody().front().getOperations();
    auto &newNodeOps = newNode.getBody().front().getOperations();
    newNodeOps.splice(newNode.end(), nodeOps);
    for (auto t : llvm::zip(node.getBody().getArguments(), node.getOperands()))
      std::get<0>(t).replaceAllUsesWith(std::get<1>(t));
    rewriter.eraseOp(node);
  }

  for (auto t : llvm::zip(newNode.getOperands(), block->getArguments()))
    std::get<0>(t).replaceUsesWithIf(std::get<1>(t), [&](OpOperand &use) {
      return newNode->isProperAncestor(use.getOwner());
    });
  return newNode;
}

/// A helper to get all users of a buffer except the given node and with the
/// given kind (producer or consumer).
static auto getUsersExcept(Value buffer, OperandKind kind, NodeOp except) {
  SmallVector<NodeOp> nodes;
  for (auto &use : buffer.getUses())
    if (auto node = dyn_cast<NodeOp>(use.getOwner()))
      if (node != except && node.getOperandKind(use) == kind)
        nodes.push_back(node);
  return nodes;
}

/// Get the consumer/producer nodes of the given buffer expect the given node.
SmallVector<NodeOp> sar::getConsumersExcept(Value buffer, NodeOp except) {
  return getUsersExcept(buffer, OperandKind::INPUT, except);
}
SmallVector<NodeOp> sar::getProducersExcept(Value buffer, NodeOp except) {
  return getUsersExcept(buffer, OperandKind::OUTPUT, except);
}
SmallVector<NodeOp> sar::getProducers(Value buffer) {
  return getProducersExcept(buffer, NodeOp());
}
SmallVector<NodeOp> sar::getDependentConsumers(Value buffer, NodeOp node) {
  // A buffer defined outside a dependence-free schedule op can
  // ignore back dependences.
  bool ignoreBackDependence =
      isa<BlockArgument>(buffer) && node.getScheduleOp().isDependenceFree();

  DominanceInfo domInfo;
  SmallVector<NodeOp> nodes;
  for (auto consumer : getConsumersExcept(buffer, node))
    if (!ignoreBackDependence || domInfo.properlyDominates(node, consumer))
      nodes.push_back(consumer);
  return nodes;
}
Value sar::findBuffer(Value memref) {
  if (auto arg = dyn_cast<BlockArgument>(memref)) {
    if (auto node = dyn_cast<NodeOp>(arg.getParentBlock()->getParentOp()))
      return findBuffer(node->getOperand(arg.getArgNumber()));
    else if (auto schedule =
                 dyn_cast<ScheduleOp>(arg.getParentBlock()->getParentOp()))
      return findBuffer(schedule->getOperand(arg.getArgNumber()));
    return memref;
  } else if (auto viewOp = memref.getDefiningOp<ViewLikeOpInterface>())
    return findBuffer(viewOp.getViewSource());
  else if (auto buffer = memref.getDefiningOp<hls::BufferLikeInterface>())
    return buffer.getMemref();
  return Value();
}
static hls::BufferLikeInterface findBufferOp(Value memref) {
  if (auto buffer = findBuffer(memref))
    return buffer.getDefiningOp<hls::BufferLikeInterface>();
  return hls::BufferLikeInterface();
}

/// Get the depth of a buffer or stream channel. Note that only if the defining
/// operation of the buffer is not a BufferOp or stream types, the returned
/// result will be 1.
unsigned sar::getBufferDepth(Value memref) {
  if (auto streamType = dyn_cast<StreamType>(memref.getType())) {
    return streamType.getDepth();
  } else if (auto bufferOp = findBufferOp(memref))
    return bufferOp.getBufferDepth();
  return 1;
}

bool sar::isExtBuffer(Value memref) {
  if (auto type = dyn_cast<MemRefType>(memref.getType()))
    return isDram(type);
  return false;
}

/// Check whether the given use has read/write semantics.
bool sar::isRead(OpOperand &use) {
  // NodeOp and ScheduleOp carry no usable memory-effect interface; the
  // effect comes from walking their region instead.
  if (auto node = dyn_cast<NodeOp>(use.getOwner()))
    return llvm::any_of(
        node.getBody().getArgument(use.getOperandNumber()).getUses(),
        [](OpOperand &argUse) { return isRead(argUse); });
  else if (auto schedule = dyn_cast<ScheduleOp>(use.getOwner()))
    return llvm::any_of(
        schedule.getBody().getArgument(use.getOperandNumber()).getUses(),
        [](OpOperand &argUse) { return isRead(argUse); });
  else if (auto view = dyn_cast<ViewLikeOpInterface>(use.getOwner()))
    return llvm::any_of(view->getUses(),
                        [](OpOperand &viewUse) { return isRead(viewUse); });
  else if (auto call = dyn_cast<func::CallOp>(use.getOwner())) {
    auto callee = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
    if (!callee || callee.isExternal())
      return true;
    return llvm::any_of(callee.getArgument(use.getOperandNumber()).getUses(),
                        [](OpOperand &argUse) { return isRead(argUse); });
  }
  return hasEffect<MemoryEffects::Read>(use.getOwner(), use.get()) ||
         isa<StreamReadOp>(use.getOwner());
}
bool sar::isWritten(OpOperand &use) {
  // A NodeOp records the kind directly; a ScheduleOp carries no usable
  // memory-effect interface, so its region is walked instead.
  if (auto node = dyn_cast<NodeOp>(use.getOwner()))
    return node.getOperandKind(use) == OperandKind::OUTPUT;
  else if (auto schedule = dyn_cast<ScheduleOp>(use.getOwner()))
    return llvm::any_of(
        schedule.getBody().getArgument(use.getOperandNumber()).getUses(),
        [](OpOperand &argUse) { return isWritten(argUse); });
  else if (auto view = dyn_cast<ViewLikeOpInterface>(use.getOwner()))
    return llvm::any_of(view->getUses(),
                        [](OpOperand &viewUse) { return isWritten(viewUse); });
  else if (auto call = dyn_cast<func::CallOp>(use.getOwner())) {
    // Once dataflow nodes are outlined the effects live in the callee;
    // follow the operand to the matching argument.
    auto callee = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
    if (!callee || callee.isExternal())
      return true; // unknown callee: assume the worst
    return llvm::any_of(callee.getArgument(use.getOperandNumber()).getUses(),
                        [](OpOperand &argUse) { return isWritten(argUse); });
  }
  return hasEffect<MemoryEffects::Write>(use.getOwner(), use.get()) ||
         isa<StreamWriteOp>(use.getOwner());
}

//===----------------------------------------------------------------------===//
// Memory and loop analysis utils
//===----------------------------------------------------------------------===//

/// Return a pair which indicates whether the if statement is always true or
/// false, respectively. The returned result is one-hot.
std::pair<bool, bool> sar::ifAlwaysTrueOrFalse(mlir::affine::AffineIfOp ifOp) {
  auto set = ifOp.getIntegerSet();
  auto operands = SmallVector<Value, 4>(ifOp.getOperands().begin(),
                                        ifOp.getOperands().end());

  // Compose all associated AffineApplyOp into the current if operation
  // (via the map-composition API: view the set constraints as a map,
  // fully compose, and convert back).
  if (llvm::any_of(operands, [](Value v) {
        return isa_and_nonnull<AffineApplyOp>(v.getDefiningOp());
      })) {
    auto map = AffineMap::get(set.getNumDims(), set.getNumSymbols(),
                              set.getConstraints(), set.getContext());
    affine::fullyComposeAffineMapAndOperands(&map, &operands);
    set = IntegerSet::get(map.getNumDims(), map.getNumSymbols(),
                          map.getResults(), set.getEqFlags());
  }

  // Replace the original integer set and operands with the composed integer
  // set and operands.
  ifOp.setIntegerSet(set);
  ifOp->setOperands(operands);

  // Combine loop induction domains with the affine-if integer set.
  FlatAffineValueConstraints constrs;
  constrs.addAffineIfOpDomain(ifOp);
  for (auto operand : operands)
    if (isAffineForInductionVar(operand)) {
      auto iv = getForInductionVarOwner(operand);
      if (failed(constrs.addAffineForOpDomain(iv)))
        continue;
    }

  bool alwaysTrue = false;
  bool alwaysFalse = false;

  if (set.getNumInputs() == 0) {
    // If the integer set is pure constant set, determine whether the
    // condition is always true or always false.
    SmallVector<bool, 4> flagList;
    unsigned idx = 0;
    for (auto expr : set.getConstraints()) {
      bool eqFlag = set.isEq(idx++);
      auto constValue = cast<AffineConstantExpr>(expr).getValue();

      if (eqFlag)
        flagList.push_back(constValue == 0);
      else
        flagList.push_back(constValue >= 0);
    }

    // Only when all sub-conditions are met, the if statement is always true.
    // Otherwise, the statement if always false.
    if (llvm::all_of(flagList, [&](bool flag) { return flag; }))
      alwaysTrue = true;
    else
      alwaysFalse = true;

  } else if (constrs.isEmpty()) {
    // If there is no solution for the constraints, the condition will always
    // be false.
    alwaysFalse = true;
  }

  // Assert only one of the two flags are true.
  assert((!alwaysTrue || !alwaysFalse) && "unexpected if condition");
  return {alwaysTrue, alwaysFalse};
}

/// Check whether the two given if statements have the same condition.
bool sar::checkSameIfStatement(AffineIfOp lhsOp, AffineIfOp rhsOp) {
  if (lhsOp == nullptr || rhsOp == nullptr)
    return false;

  auto lhsSet = lhsOp.getIntegerSet();
  auto rhsSet = rhsOp.getIntegerSet();

  // Two ifs are "the same" only when both are statements (no results):
  // yielded values would make the merged op's results ambiguous.
  if (lhsOp.getNumResults() != 0 || rhsOp.getNumResults() != 0 ||
      lhsOp.getOperands() != rhsOp.getOperands() ||
      lhsSet.getConstraints() != rhsSet.getConstraints() ||
      lhsSet.getEqFlags() != rhsSet.getEqFlags())
    return false;
  return true;
}

/// Parse array attributes.
SmallVector<int64_t, 8> sar::getIntArrayAttrValue(Operation *op,
                                                  StringRef name) {
  SmallVector<int64_t, 8> array;
  if (auto arrayAttr = op->getAttrOfType<ArrayAttr>(name)) {
    for (auto attr : arrayAttr)
      if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        array.push_back(intAttr.getInt());
      else
        return SmallVector<int64_t, 8>();
    return array;
  } else
    return SmallVector<int64_t, 8>();
}

/// Collect all load and store operations in the block and return them in "map".
void sar::getMemAccessesMap(Block &block, MemAccessesMap &map,
                            bool includeVectorTransfer) {
  for (auto &op : block) {
    if (auto load = dyn_cast<AffineReadOpInterface>(op))
      map[load.getMemRef()].push_back(&op);

    else if (auto store = dyn_cast<AffineWriteOpInterface>(op))
      map[store.getMemRef()].push_back(&op);

    else if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
      if (includeVectorTransfer)
        map[read.getBase()].push_back(&op);

    } else if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      if (includeVectorTransfer)
        map[write.getBase()].push_back(&op);

    } else if (op.getNumRegions()) {
      // Recursively collect memory access operations in each block.
      for (auto &region : op.getRegions())
        for (auto &block : region)
          getMemAccessesMap(block, map);
    }
  }
}

bool sar::crossRegionDominates(Operation *a, Operation *b) {
  if (a == b)
    return true;
  if (b->isAncestor(a))
    return false;
  while (a->getParentOp() && !a->getParentOp()->isAncestor(b))
    a = a->getParentOp();
  assert(a->getParentOp() && "reach top-level module op");
  return DominanceInfo().dominates(a, b);
}

/// Calculate the lower and upper bound of the affine map if possible.
std::optional<std::pair<int64_t, int64_t>>
sar::getBoundOfAffineMap(AffineMap map, ValueRange operands) {
  if (map.isSingleConstant()) {
    auto constBound = map.getSingleConstantResult();
    return std::pair<int64_t, int64_t>(constBound, constBound);
  }

  // Bounds are defined only for single-result maps.
  if (map.getNumResults() != 1)
    return std::optional<std::pair<int64_t, int64_t>>();
  if (operands.size() != map.getNumInputs())
    return std::nullopt;

  bool hasNonLinearExpr = false;
  map.getResult(0).walk([&](AffineExpr expr) {
    auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!binary)
      return;
    switch (binary.getKind()) {
    case AffineExprKind::Mod:
    case AffineExprKind::FloorDiv:
    case AffineExprKind::CeilDiv:
      hasNonLinearExpr = true;
      break;
    default:
      break;
    }
  });
  if (hasNonLinearExpr)
    return std::nullopt;

  auto context = map.getContext();
  SmallVector<int64_t, 4> lbs;
  SmallVector<int64_t, 4> ubs;
  for (auto operand : operands) {
    // Only if the affine map operands are induction variable, the calculation
    // is possible.
    if (!isAffineForInductionVar(operand))
      return std::optional<std::pair<int64_t, int64_t>>();

    // Only if the owner for op of the induction variable has constant bound,
    // the calculation is possible.
    auto forOp = getForInductionVarOwner(operand);
    if (!forOp.hasConstantBounds())
      return std::optional<std::pair<int64_t, int64_t>>();

    auto lb = forOp.getConstantLowerBound();
    auto ub = forOp.getConstantUpperBound();
    auto step = forOp.getStepAsInt();

    lbs.push_back(lb);
    ubs.push_back(ub - 1 - (ub - 1 - lb) % step);
  }

  // Exhaustive corner enumeration: 2^n map evaluations for n operands.
  // Operands are the enclosing loop ivs the bound map reads, so n is the
  // nesting depth in practice and the box stays small.
  auto operandNum = operands.size();
  if (operandNum > 16)
    return std::nullopt;
  uint64_t cornerCount = uint64_t{1} << operandNum;
  SmallVector<int64_t, 16> results;
  for (uint64_t i = 0; i < cornerCount; ++i) {
    SmallVector<AffineExpr, 4> replacements;
    for (unsigned pos = 0; pos < operandNum; ++pos) {
      // Bit `pos` of `i` selects the lower or upper bound for this operand,
      // enumerating every corner of the iteration box.
      if (((i >> pos) & 1) == 0)
        replacements.push_back(getAffineConstantExpr(lbs[pos], context));
      else
        replacements.push_back(getAffineConstantExpr(ubs[pos], context));
    }
    ArrayRef<AffineExpr> allReplacements(replacements);
    auto dimReplacements = allReplacements.take_front(map.getNumDims());
    auto symbolReplacements = allReplacements.drop_front(map.getNumDims());
    auto newExpr = map.getResult(0).replaceDimsAndSymbols(dimReplacements,
                                                          symbolReplacements);

    if (auto constExpr = dyn_cast<AffineConstantExpr>(newExpr))
      results.push_back(constExpr.getValue());
    else
      return std::optional<std::pair<int64_t, int64_t>>();
  }

  auto minmax = std::minmax_element(results.begin(), results.end());
  return std::pair<int64_t, int64_t>(*minmax.first, *minmax.second);
}

bool sar::isFullyPartitioned(MemRefType memrefType) {
  if (memrefType.getRank() == 0)
    return true;

  bool fullyPartitioned = false;
  SmallVector<int64_t, 8> factors;
  getPartitionFactors(memrefType, &factors);

  auto shapes = memrefType.getShape();
  fullyPartitioned =
      factors == SmallVector<int64_t, 8>(shapes.begin(), shapes.end());

  return fullyPartitioned;
}

bool sar::isCompleteRowMajorSweep(Operation *operation, Value memref) {
  AffineMap map;
  ValueRange operands;
  if (auto load = dyn_cast<AffineLoadOp>(operation)) {
    if (load.getMemRef() != memref)
      return false;
    map = load.getAffineMap();
    operands = load.getMapOperands();
  } else if (auto store = dyn_cast<AffineStoreOp>(operation)) {
    if (store.getMemRef() != memref)
      return false;
    map = store.getAffineMap();
    operands = store.getMapOperands();
  } else {
    return false;
  }

  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || !type.hasStaticShape() || !map.isIdentity() ||
      operands.size() != (unsigned)type.getRank())
    return false;

  SmallVector<AffineForOp> loops;
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<func::FuncOp>(parent))
      break;
    auto loop = dyn_cast<AffineForOp>(parent);
    if (!loop)
      return false;
    loops.push_back(loop);
  }
  std::reverse(loops.begin(), loops.end());
  if (loops.size() != (unsigned)type.getRank())
    return false;

  for (auto [dim, loop] : llvm::enumerate(loops)) {
    if (operands[dim] != loop.getInductionVar() || !loop.hasConstantBounds() ||
        loop.getConstantLowerBound() != 0 ||
        loop.getConstantUpperBound() != type.getDimSize(dim) ||
        loop.getStep() != 1)
      return false;
  }
  return true;
}

bool sar::isNestedInLoop(Operation *operation) {
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<func::FuncOp>(parent))
      return false;
    if (isa<LoopLikeOpInterface>(parent))
      return true;
  }
  return false;
}

// Calculate partition factors through analyzing the "memrefType" and return
// them in "factors". Meanwhile, the overall partition number is calculated and
// returned as well.
int64_t sar::getPartitionFactors(MemRefType memrefType,
                                 SmallVectorImpl<int64_t> *factors) {
  int64_t accumFactor = 1;
  if (auto attr = dyn_cast<PartitionLayoutAttr>(memrefType.getLayout()))
    for (auto factor : attr.getActualFactors(memrefType.getShape())) {
      accumFactor *= factor;
      if (factors)
        factors->push_back(factor);
    }
  else if (factors)
    factors->assign(memrefType.getRank(), 1);
  return accumFactor;
}

func::FuncOp sar::getTopFunc(ModuleOp module, std::string topFuncName) {
  func::FuncOp topFunc;
  for (auto func : module.getOps<func::FuncOp>())
    if (hasTopFuncAttr(func) || func.getName() == topFuncName) {
      if (!topFunc)
        topFunc = func;
      else
        return func::FuncOp();
    }
  return topFunc;
}
