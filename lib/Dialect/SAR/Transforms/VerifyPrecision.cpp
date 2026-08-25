//===- VerifyPrecision.cpp - Check backend precision contracts ------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/STLExtras.h"

#include "sar/Dialect/SAR/IR/SAROps.h"
#include "sar/Dialect/SAR/Transforms/Passes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SARVERIFYPRECISION
#include "sar/Dialect/SAR/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;

namespace {

static bool conflictsWith(Type type, unsigned width) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth() != width;
  if (auto complexType = dyn_cast<ComplexType>(type))
    return conflictsWith(complexType.getElementType(), width);
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return conflictsWith(shapedType.getElementType(), width);
  if (auto functionType = dyn_cast<FunctionType>(type)) {
    return llvm::any_of(
               functionType.getInputs(),
               [width](Type input) { return conflictsWith(input, width); }) ||
           llvm::any_of(functionType.getResults(), [width](Type result) {
             return conflictsWith(result, width);
           });
  }
  return false;
}

/// Sampling coordinates the dialect fixes at f64, and everything computed
/// solely to produce them.
///
/// A position is an index, not a sample: it has to resolve a fractional bin
/// over the whole raster, which f32 stops doing well before the scene sizes
/// this compiler targets, so `sar.interp1d` and `sar.gather2d` require f64
/// there. Charging that against an `f32` data path would make the policy
/// unsatisfiable for every resampling kernel -- which is every SAR chain --
/// so the position operands and the arithmetic feeding only them are exempt.
/// A value used anywhere else is data and is checked as data.
static llvm::DenseSet<Value> findPositionValues(Operation *root) {
  llvm::DenseSet<Value> positions;
  SmallVector<Value> worklist;
  auto seed = [&](Value value) {
    if (positions.insert(value).second)
      worklist.push_back(value);
  };

  root->walk([&](Operation *op) {
    if (auto interp = dyn_cast<sar::Interp1DOp>(op))
      seed(interp.getPositions());
    else if (auto interp = dyn_cast<sar::Interp1DSplitOp>(op))
      seed(interp.getPositions());
    else if (auto gather = dyn_cast<sar::Gather2DOp>(op)) {
      seed(gather.getRows());
      seed(gather.getCols());
    } else if (auto gather = dyn_cast<sar::Gather2DSplitOp>(op)) {
      seed(gather.getRows());
      seed(gather.getCols());
    }
  });

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    Operation *producer = value.getDefiningOp();
    // Only a value whose every use is a position is position arithmetic;
    // one shared with the data path is data. A use counts as a position
    // when the consumer is itself position arithmetic already proven
    // exempt, or when the value lands on a resampling op's position
    // operand -- which is what seeded the walk.
    if (!producer || !llvm::all_of(value.getUsers(), [&](Operation *user) {
          if (isa<sar::Interp1DOp, sar::Interp1DSplitOp, sar::Gather2DOp,
                  sar::Gather2DSplitOp>(user))
            return true;
          return llvm::all_of(user->getResults(), [&](Value result) {
            return positions.contains(result);
          });
        }))
      continue;
    if (!llvm::all_of(producer->getResults(),
                      [&](Value result) { return positions.contains(result); }))
      continue;
    for (Value operand : producer->getOperands())
      seed(operand);
  }
  return positions;
}

struct VerifyPrecision
    : public sar::impl::SARVerifyPrecisionBase<VerifyPrecision> {
  using SARVerifyPrecisionBase::SARVerifyPrecisionBase;

  void runOnOperation() override {
    StringRef policy = precision;
    if (policy == "native")
      return;
    unsigned width = policy == "f32" ? 32 : policy == "f64" ? 64 : 0;
    if (!width) {
      getOperation().emitError()
          << "precision must be one of native, f32 or f64; got " << policy;
      return signalPassFailure();
    }

    auto positions = findPositionValues(getOperation());
    WalkResult result = getOperation().walk([&](Operation *op) {
      auto reject = [&](Type type, StringRef role) {
        if (!conflictsWith(type, width))
          return false;
        op->emitOpError() << role << " type " << type
                          << " violates precision=" << policy;
        return true;
      };

      // A function type repeats its block-argument and terminator types,
      // which are checked below with the exemption applied.
      if (isa<FunctionOpInterface>(op))
        return WalkResult::advance();
      for (Value operand : op->getOperands())
        if (!positions.contains(operand) &&
            reject(operand.getType(), "operand"))
          return WalkResult::interrupt();
      for (Value result : op->getResults())
        if (!positions.contains(result) && reject(result.getType(), "result"))
          return WalkResult::interrupt();
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            if (!positions.contains(argument) &&
                reject(argument.getType(), "block argument"))
              return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
