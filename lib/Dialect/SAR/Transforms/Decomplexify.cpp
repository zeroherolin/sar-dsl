//===- Decomplexify.cpp - Split complex tensors into float planes ---------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
// Rewrites every function so that no complex element types remain: complex
// tensor values become (re, im) pairs of float tensors and complex
// arithmetic expands into real arithmetic built from SAR element-wise ops.
// The transformation stays entirely at the SAR dialect level, so all
// existing lowerings apply to its output.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"

#include "sar/Dialect/SAR/IR/SARDialect.h"
#include "sar/Dialect/SAR/IR/SAROps.h"
#include "sar/Dialect/SAR/Transforms/Passes.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_SARDECOMPLEXIFY
#include "sar/Dialect/SAR/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;

namespace {

static bool isComplexTensor(Type type) {
  auto tensorTy = dyn_cast<RankedTensorType>(type);
  return tensorTy && isa<ComplexType>(tensorTy.getElementType());
}

/// tensor<...xcomplex<T>> -> tensor<...xT>
static RankedTensorType getPlaneType(Type type) {
  auto tensorTy = cast<RankedTensorType>(type);
  auto complexTy = cast<ComplexType>(tensorTy.getElementType());
  return RankedTensorType::get(tensorTy.getShape(),
                               complexTy.getElementType());
}

/// Splits a dense complex attribute into (re, im) float attributes.
static std::pair<DenseElementsAttr, DenseElementsAttr>
splitComplexAttr(DenseElementsAttr attr, RankedTensorType planeType) {
  SmallVector<APFloat> re, im;
  re.reserve(attr.isSplat() ? 1 : attr.getNumElements());
  im.reserve(re.capacity());
  if (attr.isSplat()) {
    auto value = attr.getSplatValue<std::complex<APFloat>>();
    re.push_back(value.real());
    im.push_back(value.imag());
  } else {
    for (auto value : attr.getValues<std::complex<APFloat>>()) {
      re.push_back(value.real());
      im.push_back(value.imag());
    }
  }
  return {DenseElementsAttr::get(planeType, re),
          DenseElementsAttr::get(planeType, im)};
}

/// Rewrites one function into split-complex form.
class FunctionDecomplexifier {
public:
  FunctionDecomplexifier(func::FuncOp original, OpBuilder &moduleBuilder)
      : original(original), builder(moduleBuilder) {}

  FailureOr<func::FuncOp> run();

private:
  using Split = std::pair<Value, Value>; // (re, im)

  // Value accessors for the function under construction.
  Split getSplit(Value oldValue) {
    auto it = splitValues.find(oldValue);
    assert(it != splitValues.end() && "complex value not mapped");
    return it->second;
  }
  Value getReal(Value oldValue) {
    Value mapped = realValues.lookupOrNull(oldValue);
    assert(mapped && "real value not mapped");
    return mapped;
  }

  LogicalResult rewriteOp(Operation *op);

  func::FuncOp original;
  OpBuilder &builder;
  DenseMap<Value, Split> splitValues;
  IRMapping realValues;
};

FailureOr<func::FuncOp> FunctionDecomplexifier::run() {
  // Converted signature: complex tensors expand to adjacent (re, im) pairs.
  SmallVector<Type> inputTypes, resultTypes;
  for (Type type : original.getFunctionType().getInputs()) {
    if (isComplexTensor(type)) {
      inputTypes.push_back(getPlaneType(type));
      inputTypes.push_back(getPlaneType(type));
    } else {
      inputTypes.push_back(type);
    }
  }
  for (Type type : original.getFunctionType().getResults()) {
    if (isComplexTensor(type)) {
      resultTypes.push_back(getPlaneType(type));
      resultTypes.push_back(getPlaneType(type));
    } else {
      resultTypes.push_back(type);
    }
  }

  auto replacement = builder.create<func::FuncOp>(
      original.getLoc(), original.getName(),
      builder.getFunctionType(inputTypes, resultTypes));
  replacement.setVisibility(original.getVisibility());

  Block *entry = replacement.addEntryBlock();
  unsigned newArgIndex = 0;
  for (BlockArgument arg : original.getArguments()) {
    if (isComplexTensor(arg.getType())) {
      Value re = entry->getArgument(newArgIndex++);
      Value im = entry->getArgument(newArgIndex++);
      splitValues[arg] = {re, im};
    } else {
      realValues.map(arg, entry->getArgument(newArgIndex++));
    }
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);
  for (Operation &op : original.getBody().front()) {
    if (failed(rewriteOp(&op))) {
      replacement.erase();
      return failure();
    }
  }
  return replacement;
}

LogicalResult FunctionDecomplexifier::rewriteOp(Operation *op) {
  Location loc = op->getLoc();

  bool touchesComplex =
      llvm::any_of(op->getOperandTypes(), isComplexTensor) ||
      llvm::any_of(op->getResultTypes(), isComplexTensor);

  // Ops without complex involvement are cloned with remapped operands.
  if (!touchesComplex && !isa<func::ReturnOp>(op)) {
    builder.clone(*op, realValues);
    return success();
  }

  auto planeOf = [&](Value v) { return getPlaneType(v.getType()); };

  auto mul = [&](Value a, Value b) -> Value {
    return builder.create<MulOp>(loc, a, b);
  };
  auto add = [&](Value a, Value b) -> Value {
    return builder.create<AddOp>(loc, a, b);
  };
  auto sub = [&](Value a, Value b) -> Value {
    return builder.create<SubOp>(loc, a, b);
  };

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<func::ReturnOp>([&](func::ReturnOp ret) {
        SmallVector<Value> operands;
        for (Value v : ret.getOperands()) {
          if (isComplexTensor(v.getType())) {
            auto [re, im] = getSplit(v);
            operands.push_back(re);
            operands.push_back(im);
          } else {
            operands.push_back(getReal(v));
          }
        }
        builder.create<func::ReturnOp>(loc, operands);
        return success();
      })
      .Case<ConstantOp>([&](ConstantOp cst) -> LogicalResult {
        auto dense = dyn_cast<DenseElementsAttr>(cst.getValueAttr());
        if (!dense) {
          cst.emitOpError("expected a dense complex constant");
          return failure();
        }
        auto planeType = getPlaneType(cst.getType());
        auto [reAttr, imAttr] = splitComplexAttr(dense, planeType);
        Value re = builder.create<ConstantOp>(loc, planeType, reAttr);
        Value im = builder.create<ConstantOp>(loc, planeType, imAttr);
        splitValues[cst.getResult()] = {re, im};
        return success();
      })
      .Case<AddOp, SubOp>([&](auto binOp) {
        auto [lre, lim] = getSplit(binOp.getLhs());
        auto [rre, rim] = getSplit(binOp.getRhs());
        bool isAdd = isa<AddOp>(op);
        Value re = isAdd ? add(lre, rre) : sub(lre, rre);
        Value im = isAdd ? add(lim, rim) : sub(lim, rim);
        splitValues[binOp.getResult()] = {re, im};
        return success();
      })
      .Case<MulOp>([&](MulOp mulOp) {
        auto [a, b] = getSplit(mulOp.getLhs());
        auto [c, d] = getSplit(mulOp.getRhs());
        splitValues[mulOp.getResult()] = {sub(mul(a, c), mul(b, d)),
                                          add(mul(a, d), mul(b, c))};
        return success();
      })
      .Case<DivOp>([&](DivOp divOp) {
        auto [a, b] = getSplit(divOp.getLhs());
        auto [c, d] = getSplit(divOp.getRhs());
        Value denom = add(mul(c, c), mul(d, d));
        Value re = builder.create<DivOp>(loc, add(mul(a, c), mul(b, d)),
                                         denom);
        Value im = builder.create<DivOp>(loc, sub(mul(b, c), mul(a, d)),
                                         denom);
        splitValues[divOp.getResult()] = {re, im};
        return success();
      })
      .Case<AddScalarOp>([&](AddScalarOp addScalar) {
        auto [re, im] = getSplit(addScalar.getInput());
        Value newRe = builder.create<AddScalarOp>(
            loc, re, addScalar.getScalar());
        splitValues[addScalar.getResult()] = {newRe, im};
        return success();
      })
      .Case<MulScalarOp>([&](MulScalarOp mulScalar) {
        auto [re, im] = getSplit(mulScalar.getInput());
        splitValues[mulScalar.getResult()] = {
            builder.create<MulScalarOp>(loc, re, mulScalar.getScalar()),
            builder.create<MulScalarOp>(loc, im, mulScalar.getScalar())};
        return success();
      })
      .Case<NegOp>([&](NegOp neg) {
        auto [re, im] = getSplit(neg.getInput());
        splitValues[neg.getResult()] = {builder.create<NegOp>(loc, re),
                                        builder.create<NegOp>(loc, im)};
        return success();
      })
      .Case<AbsOp>([&](AbsOp abs) {
        auto [re, im] = getSplit(abs.getInput());
        Value magnitude = builder.create<SqrtOp>(
            loc, add(mul(re, re), mul(im, im)));
        realValues.map(abs.getResult(), magnitude);
        return success();
      })
      .Case<ExpJOp>([&](ExpJOp expj) {
        Value phase = getReal(expj.getInput());
        splitValues[expj.getResult()] = {
            builder.create<CosOp>(loc, phase),
            builder.create<SinOp>(loc, phase)};
        return success();
      })
      .Case<CastOp>([&](CastOp castOp) {
        auto resultPlane = getPlaneType(castOp.getType());
        if (isComplexTensor(castOp.getInput().getType())) {
          auto [re, im] = getSplit(castOp.getInput());
          auto castPlane = [&](Value v) -> Value {
            if (v.getType() == Type(resultPlane))
              return v;
            return builder.create<CastOp>(loc, resultPlane, v);
          };
          splitValues[castOp.getResult()] = {castPlane(re), castPlane(im)};
        } else {
          Value in = getReal(castOp.getInput());
          Value re = in;
          if (in.getType() != Type(resultPlane))
            re = builder.create<CastOp>(loc, resultPlane, in);
          auto zeroAttr = DenseElementsAttr::get(
              resultPlane,
              builder.getFloatAttr(resultPlane.getElementType(), 0.0));
          Value im = builder.create<ConstantOp>(loc, resultPlane, zeroAttr);
          splitValues[castOp.getResult()] = {re, im};
        }
        return success();
      })
      .Case<TransposeOp>([&](TransposeOp transpose) {
        auto [re, im] = getSplit(transpose.getInput());
        auto plane = getPlaneType(transpose.getType());
        splitValues[transpose.getResult()] = {
            builder.create<TransposeOp>(loc, plane, re),
            builder.create<TransposeOp>(loc, plane, im)};
        return success();
      })
      .Case<FFTShiftOp>([&](FFTShiftOp shift) {
        auto [re, im] = getSplit(shift.getInput());
        auto make = [&](Value v) -> Value {
          return builder.create<FFTShiftOp>(loc, v, shift.getDim(),
                                            shift.getInverse());
        };
        splitValues[shift.getResult()] = {make(re), make(im)};
        return success();
      })
      .Case<BroadcastOp>([&](BroadcastOp bcast) {
        auto [re, im] = getSplit(bcast.getInput());
        auto plane = getPlaneType(bcast.getType());
        splitValues[bcast.getResult()] = {
            builder.create<BroadcastOp>(loc, plane, re, bcast.getDim()),
            builder.create<BroadcastOp>(loc, plane, im, bcast.getDim())};
        return success();
      })
      .Case<FFTOp, IFFTOp>([&](auto fft) {
        auto [re, im] = getSplit(fft.getInput());
        auto split = builder.create<FFTSplitOp>(
            loc, re, im, fft.getDim(), /*inverse=*/isa<IFFTOp>(op));
        splitValues[fft.getResult()] = {split.getOutRe(), split.getOutIm()};
        return success();
      })
      .Case<Interp1DOp>([&](Interp1DOp interp) {
        auto [re, im] = getSplit(interp.getData());
        auto split = builder.create<Interp1DSplitOp>(
            loc, re, im, getReal(interp.getPositions()));
        splitValues[interp.getResult()] = {split.getOutRe(),
                                           split.getOutIm()};
        return success();
      })
      .Case<StoltInterpOp>([&](StoltInterpOp stolt) {
        auto [re, im] = getSplit(stolt.getData());
        auto split = builder.create<StoltInterpSplitOp>(
            loc, re, im, getReal(stolt.getFa()), getReal(stolt.getFr()),
            stolt.getC(), stolt.getFc(), stolt.getVr(), stolt.getTShift());
        splitValues[stolt.getResult()] = {split.getOutRe(),
                                          split.getOutIm()};
        return success();
      })
      .Default([&](Operation *other) {
        return other->emitError()
               << "operation has no split-complex form; keep it on an "
                  "execution backend or decompose it before "
                  "sar-decomplexify";
      });
}

struct SARDecomplexifyPass
    : sar::impl::SARDecomplexifyBase<SARDecomplexifyPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    SmallVector<func::FuncOp> worklist;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      bool hasComplex =
          llvm::any_of(func.getFunctionType().getInputs(), isComplexTensor) ||
          llvm::any_of(func.getFunctionType().getResults(), isComplexTensor);
      if (!hasComplex) {
        // Also check for internal complex values.
        func.walk([&](Operation *op) {
          hasComplex |= llvm::any_of(op->getResultTypes(), isComplexTensor);
        });
      }
      if (hasComplex)
        worklist.push_back(func);
    }

    for (func::FuncOp func : worklist) {
      builder.setInsertionPoint(func);
      FunctionDecomplexifier rewriter(func, builder);
      std::string name = func.getName().str();
      // The replacement is created adjacent to the original; the original
      // is erased first so the symbol name stays unique.
      func.setName(name + "__complex_orig");
      FailureOr<func::FuncOp> replacement = rewriter.run();
      if (failed(replacement)) {
        signalPassFailure();
        return;
      }
      replacement->setName(name);
      func.erase();
    }
  }
};

} // namespace
