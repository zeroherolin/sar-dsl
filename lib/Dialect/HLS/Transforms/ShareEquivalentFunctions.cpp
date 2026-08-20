//===- ShareEquivalentFunctions.cpp - merge reusable HLS helpers ---------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SHAREEQUIVALENTFUNCTIONS
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;
using namespace mlir::sar::hls;

namespace {

static bool equivalentAttributes(func::FuncOp lhs, func::FuncOp rhs) {
  auto filtered = [](func::FuncOp func) {
    SmallVector<NamedAttribute> attributes;
    for (NamedAttribute attribute : func->getAttrs())
      if (attribute.getName().getValue() != SymbolTable::getSymbolAttrName())
        attributes.push_back(attribute);
    return attributes;
  };
  return filtered(lhs) == filtered(rhs);
}

static bool equivalentFunctions(func::FuncOp lhs, func::FuncOp rhs) {
  if (lhs.getFunctionType() != rhs.getFunctionType() ||
      !equivalentAttributes(lhs, rhs) || lhs.getBlocks().size() != 1 ||
      rhs.getBlocks().size() != 1)
    return false;

  auto &lhsBlock = lhs.front();
  auto &rhsBlock = rhs.front();
  if (lhsBlock.getNumArguments() != rhsBlock.getNumArguments() ||
      lhsBlock.getOperations().size() != rhsBlock.getOperations().size())
    return false;

  DenseMap<Value, Value> equivalent;
  for (auto [lhsArg, rhsArg] :
       llvm::zip(lhsBlock.getArguments(), rhsBlock.getArguments()))
    equivalent[lhsArg] = rhsArg;

  auto check = [&](Value lhsValue, Value rhsValue) {
    auto found = equivalent.find(lhsValue);
    return success(found != equivalent.end() && found->second == rhsValue);
  };
  auto mark = [&](Value lhsValue, Value rhsValue) {
    equivalent[lhsValue] = rhsValue;
  };

  for (auto [lhsOp, rhsOp] :
       llvm::zip(lhsBlock.getOperations(), rhsBlock.getOperations()))
    if (!OperationEquivalence::isEquivalentTo(
            &lhsOp, &rhsOp, check, mark,
            OperationEquivalence::Flags::IgnoreLocations))
      return false;
  return true;
}

static bool isCandidate(func::FuncOp func) {
  return !func.isExternal() && !hasTopFuncAttr(func) && !hasRuntimeAttr(func);
}

static bool isEngine(func::FuncOp func) {
  bool hasStorage = false;
  bool hasCall = false;
  func.walk([&](Operation *operation) {
    hasStorage |= isa<memref::AllocOp, hls::BufferOp>(operation);
    hasCall |= isa<func::CallOp>(operation);
  });
  return hasStorage && hasCall;
}

struct ShareEquivalentFunctions
    : public sar::impl::ShareEquivalentFunctionsBase<ShareEquivalentFunctions> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::StringSet<> mergedCanonical;

    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<func::FuncOp> functions;
      for (func::FuncOp func : module.getOps<func::FuncOp>())
        if (isCandidate(func))
          functions.push_back(func);

      for (unsigned i = 0; i < functions.size() && !changed; ++i)
        for (unsigned j = i + 1; j < functions.size(); ++j) {
          func::FuncOp canonical = functions[i];
          func::FuncOp duplicate = functions[j];
          if (!equivalentFunctions(canonical, duplicate))
            continue;

          module.walk([&](func::CallOp call) {
            if (call.getCallee() == duplicate.getName())
              call.setCallee(canonical.getName());
          });
          mergedCanonical.insert(canonical.getName());
          duplicate.erase();
          changed = true;
          break;
        }
    }

    Builder builder(module.getContext());
    for (auto &name : mergedCanonical) {
      auto func = module.lookupSymbol<func::FuncOp>(name.getKey());
      if (!func || !isEngine(func))
        continue;
      unsigned calls = 0;
      module.walk([&](func::CallOp call) {
        calls += call.getCallee() == func.getName();
      });
      if (calls < 2)
        continue;
      func->removeAttr("inline");
      func->setAttr("hls.shared_instance", builder.getUnitAttr());
    }
  }
};

} // namespace

std::unique_ptr<Pass> sar::createShareEquivalentFunctionsPass() {
  return std::make_unique<ShareEquivalentFunctions>();
}
