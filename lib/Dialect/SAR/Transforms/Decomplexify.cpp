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
  return RankedTensorType::get(tensorTy.getShape(), complexTy.getElementType());
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

  auto replacement =
      func::FuncOp::create(builder, original.getLoc(), original.getName(),
                           builder.getFunctionType(inputTypes, resultTypes));
  replacement.setVisibility(original.getVisibility());

  // The frontend names arguments after the kernel's Python parameters
  // (`sar.arg_names`); a split complex argument keeps the name on both of
  // its planes so the emitted design's ports still read as the user's
  // signature.
  if (auto names = original->getAttrOfType<ArrayAttr>("sar.arg_names");
      names && names.size() == original.getNumArguments()) {
    SmallVector<Attribute> planeNames;
    for (auto [name, type] :
         llvm::zip(names.getValue(), original.getArgumentTypes())) {
      auto base = cast<StringAttr>(name).getValue();
      if (isComplexTensor(type)) {
        planeNames.push_back(builder.getStringAttr(base + "_re"));
        planeNames.push_back(builder.getStringAttr(base + "_im"));
      } else {
        planeNames.push_back(name);
      }
    }
    replacement->setAttr("sar.arg_names", builder.getArrayAttr(planeNames));
  }

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

  bool touchesComplex = llvm::any_of(op->getOperandTypes(), isComplexTensor) ||
                        llvm::any_of(op->getResultTypes(), isComplexTensor);

  // Ops without complex involvement are cloned with remapped operands.
  // A loop is rebuilt either way: its body may compute complex values
  // even when every carry is real.
  if (!touchesComplex && !isa<func::ReturnOp, IterateOp>(op)) {
    builder.clone(*op, realValues);
    return success();
  }

  auto mul = [&](Value a, Value b) -> Value {
    return MulOp::create(builder, loc, a, b);
  };
  auto add = [&](Value a, Value b) -> Value {
    return AddOp::create(builder, loc, a, b);
  };
  auto sub = [&](Value a, Value b) -> Value {
    return SubOp::create(builder, loc, a, b);
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
        func::ReturnOp::create(builder, loc, operands);
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
        Value re = ConstantOp::create(builder, loc, planeType, reAttr);
        Value im = ConstantOp::create(builder, loc, planeType, imAttr);
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
        Value re =
            DivOp::create(builder, loc, add(mul(a, c), mul(b, d)), denom);
        Value im =
            DivOp::create(builder, loc, sub(mul(b, c), mul(a, d)), denom);
        splitValues[divOp.getResult()] = {re, im};
        return success();
      })
      .Case<AddScalarOp>([&](AddScalarOp addScalar) {
        auto [re, im] = getSplit(addScalar.getInput());
        Value newRe =
            AddScalarOp::create(builder, loc, re, addScalar.getScalar());
        splitValues[addScalar.getResult()] = {newRe, im};
        return success();
      })
      .Case<MulScalarOp>([&](MulScalarOp mulScalar) {
        auto [re, im] = getSplit(mulScalar.getInput());
        splitValues[mulScalar.getResult()] = {
            MulScalarOp::create(builder, loc, re, mulScalar.getScalar()),
            MulScalarOp::create(builder, loc, im, mulScalar.getScalar())};
        return success();
      })
      .Case<AbsOp>([&](AbsOp abs) {
        auto [re, im] = getSplit(abs.getInput());
        Value magnitude =
            SqrtOp::create(builder, loc, add(mul(re, re), mul(im, im)));
        realValues.map(abs.getResult(), magnitude);
        return success();
      })
      .Case<ConjOp>([&](ConjOp conj) {
        auto [re, im] = getSplit(conj.getInput());
        splitValues[conj.getResult()] = {
            re, MulScalarOp::create(builder, loc, im,
                                    builder.getF64FloatAttr(-1.0))};
        return success();
      })
      .Case<RealOp>([&](RealOp real) {
        realValues.map(real.getResult(), getSplit(real.getInput()).first);
        return success();
      })
      .Case<ImagOp>([&](ImagOp imag) {
        realValues.map(imag.getResult(), getSplit(imag.getInput()).second);
        return success();
      })
      .Case<WhereOp>([&](WhereOp where) {
        // The mask is real; complex branches select plane-wise.
        Value mask = getReal(where.getMask());
        auto [lre, lim] = getSplit(where.getLhs());
        auto [rre, rim] = getSplit(where.getRhs());
        splitValues[where.getResult()] = {
            WhereOp::create(builder, loc, lre.getType(), mask, lre, rre),
            WhereOp::create(builder, loc, lim.getType(), mask, lim, rim)};
        return success();
      })
      .Case<ComplexOp>([&](ComplexOp create) {
        splitValues[create.getResult()] = {getReal(create.getRe()),
                                           getReal(create.getIm())};
        return success();
      })
      .Case<CastOp>([&](CastOp castOp) {
        auto resultPlane = getPlaneType(castOp.getType());
        if (isComplexTensor(castOp.getInput().getType())) {
          auto [re, im] = getSplit(castOp.getInput());
          auto castPlane = [&](Value v) -> Value {
            if (v.getType() == Type(resultPlane))
              return v;
            return CastOp::create(builder, loc, resultPlane, v);
          };
          splitValues[castOp.getResult()] = {castPlane(re), castPlane(im)};
        } else {
          Value in = getReal(castOp.getInput());
          Value re = in;
          if (in.getType() != Type(resultPlane))
            re = CastOp::create(builder, loc, resultPlane, in);
          auto zeroAttr = DenseElementsAttr::get(
              resultPlane,
              builder.getFloatAttr(resultPlane.getElementType(), 0.0));
          Value im = ConstantOp::create(builder, loc, resultPlane, zeroAttr);
          splitValues[castOp.getResult()] = {re, im};
        }
        return success();
      })
      .Case<ReduceOp>([&](ReduceOp reduce) {
        // Complex reductions are sums (verified); sum splits plane-wise.
        auto [re, im] = getSplit(reduce.getInput());
        auto plane = getPlaneType(reduce.getType());
        splitValues[reduce.getResult()] = {
            ReduceOp::create(builder, loc, plane, re, reduce.getKindAttr(),
                             reduce.getDimAttr()),
            ReduceOp::create(builder, loc, plane, im, reduce.getKindAttr(),
                             reduce.getDimAttr())};
        return success();
      })
      .Case<CumsumOp>([&](CumsumOp cumsum) {
        // A prefix sum is linear, so it runs independently on each plane.
        auto [re, im] = getSplit(cumsum.getInput());
        splitValues[cumsum.getResult()] = {
            CumsumOp::create(builder, loc, re.getType(), re,
                             cumsum.getDimAttr()),
            CumsumOp::create(builder, loc, im.getType(), im,
                             cumsum.getDimAttr())};
        return success();
      })
      .Case<TransposeOp>([&](TransposeOp transpose) {
        auto [re, im] = getSplit(transpose.getInput());
        auto plane = getPlaneType(transpose.getType());
        splitValues[transpose.getResult()] = {
            TransposeOp::create(builder, loc, plane, re),
            TransposeOp::create(builder, loc, plane, im)};
        return success();
      })
      .Case<SliceOp>([&](SliceOp slice) {
        auto [re, im] = getSplit(slice.getInput());
        auto plane = getPlaneType(slice.getType());
        splitValues[slice.getResult()] = {
            SliceOp::create(builder, loc, plane, re, slice.getOffsetsAttr(),
                            slice.getSizesAttr(), slice.getStridesAttr()),
            SliceOp::create(builder, loc, plane, im, slice.getOffsetsAttr(),
                            slice.getSizesAttr(), slice.getStridesAttr())};
        return success();
      })
      .Case<ConcatOp>([&](ConcatOp concat) {
        auto [lre, lim] = getSplit(concat.getLhs());
        auto [rre, rim] = getSplit(concat.getRhs());
        auto plane = getPlaneType(concat.getType());
        splitValues[concat.getResult()] = {
            ConcatOp::create(builder, loc, plane, lre, rre,
                             concat.getDimAttr()),
            ConcatOp::create(builder, loc, plane, lim, rim,
                             concat.getDimAttr())};
        return success();
      })
      .Case<PadOp>([&](PadOp pad) {
        // The pad value lands in the real plane; the imaginary plane pads
        // with zero.
        auto [re, im] = getSplit(pad.getInput());
        auto plane = getPlaneType(pad.getType());
        splitValues[pad.getResult()] = {
            PadOp::create(builder, loc, plane, re, pad.getLowAttr(),
                          pad.getHighAttr(), pad.getValueAttr()),
            PadOp::create(builder, loc, plane, im, pad.getLowAttr(),
                          pad.getHighAttr(), builder.getF64FloatAttr(0.0))};
        return success();
      })
      .Case<ReverseOp>([&](ReverseOp reverse) {
        auto [re, im] = getSplit(reverse.getInput());
        splitValues[reverse.getResult()] = {
            ReverseOp::create(builder, loc, re.getType(), re,
                              reverse.getDimAttr()),
            ReverseOp::create(builder, loc, im.getType(), im,
                              reverse.getDimAttr())};
        return success();
      })
      .Case<FFTShiftOp>([&](FFTShiftOp shift) {
        auto [re, im] = getSplit(shift.getInput());
        auto make = [&](Value v) -> Value {
          return FFTShiftOp::create(builder, loc, v, shift.getDim(),
                                    shift.getInverse());
        };
        splitValues[shift.getResult()] = {make(re), make(im)};
        return success();
      })
      .Case<BroadcastOp>([&](BroadcastOp bcast) {
        auto [re, im] = getSplit(bcast.getInput());
        auto plane = getPlaneType(bcast.getType());
        splitValues[bcast.getResult()] = {
            BroadcastOp::create(builder, loc, plane, re, bcast.getDim()),
            BroadcastOp::create(builder, loc, plane, im, bcast.getDim())};
        return success();
      })
      .Case<FFTOp, IFFTOp>([&](auto fft) {
        auto [re, im] = getSplit(fft.getInput());
        auto split = FFTSplitOp::create(builder, loc, re, im, fft.getDim(),
                                        /*inverse=*/isa<IFFTOp>(op));
        splitValues[fft.getResult()] = {split.getOutRe(), split.getOutIm()};
        return success();
      })
      .Case<IterateOp>([&](IterateOp loop) -> LogicalResult {
        // Complex carries expand into adjacent (re, im) pairs, exactly as
        // function signatures do; the body is rewritten recursively into
        // the new loop's block.
        SmallVector<Value> inits;
        SmallVector<Type> carryTypes;
        SmallVector<Location> argLocs;
        for (Value init : loop.getInits()) {
          if (isComplexTensor(init.getType())) {
            auto [re, im] = getSplit(init);
            inits.append({re, im});
            carryTypes.append({re.getType(), im.getType()});
            argLocs.append(2, init.getLoc());
          } else {
            Value mapped = getReal(init);
            inits.push_back(mapped);
            carryTypes.push_back(mapped.getType());
            argLocs.push_back(init.getLoc());
          }
        }

        auto newLoop = IterateOp::create(builder, loc, carryTypes, inits,
                                         loop.getTripsAttr());
        {
          // createBlock moves the insertion point into the new block; the
          // guard restores it so the ops after the loop land after it.
          OpBuilder::InsertionGuard guard(builder);
          Block *block =
              builder.createBlock(&newLoop.getBody(), {}, carryTypes, argLocs);
          unsigned argIndex = 0;
          for (BlockArgument arg : loop.getBody().front().getArguments()) {
            if (isComplexTensor(arg.getType())) {
              splitValues[arg] = {block->getArgument(argIndex),
                                  block->getArgument(argIndex + 1)};
              argIndex += 2;
            } else {
              realValues.map(arg, block->getArgument(argIndex++));
            }
          }
          for (Operation &inner : loop.getBody().front())
            if (failed(rewriteOp(&inner)))
              return failure();
        }

        unsigned resultIndex = 0;
        for (Value result : loop.getResults()) {
          if (isComplexTensor(result.getType())) {
            splitValues[result] = {newLoop.getResult(resultIndex),
                                   newLoop.getResult(resultIndex + 1)};
            resultIndex += 2;
          } else {
            realValues.map(result, newLoop.getResult(resultIndex++));
          }
        }
        return success();
      })
      .Case<YieldOp>([&](YieldOp yield) {
        SmallVector<Value> operands;
        for (Value v : yield.getOperands()) {
          if (isComplexTensor(v.getType())) {
            auto [re, im] = getSplit(v);
            operands.append({re, im});
          } else {
            operands.push_back(getReal(v));
          }
        }
        YieldOp::create(builder, loc, operands);
        return success();
      })
      .Case<Gather2DOp>([&](Gather2DOp gather) {
        auto [re, im] = getSplit(gather.getData());
        auto plane = getPlaneType(gather.getType());
        auto split = Gather2DSplitOp::create(
            builder, loc, plane, plane, re, im, getReal(gather.getRows()),
            getReal(gather.getCols()), gather.getKernelAttr(),
            gather.getBoundaryAttr());
        splitValues[gather.getResult()] = {split.getOutRe(), split.getOutIm()};
        return success();
      })
      .Case<Interp1DOp>([&](Interp1DOp interp) {
        auto [re, im] = getSplit(interp.getData());
        auto split = Interp1DSplitOp::create(
            builder, loc, re, im, getReal(interp.getPositions()),
            interp.getDim(), interp.getKernel(), interp.getTaps(),
            interp.getWindow(), interp.getBeta(), interp.getBoundary());
        splitValues[interp.getResult()] = {split.getOutRe(), split.getOutIm()};
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
      // The rewrite walks one straight-line body: a declaration has no body
      // to walk, and branch successors would need their block arguments
      // remapped. Traced kernels are always single-block; anything else is
      // rejected rather than miscompiled.
      if (!func.getBody().hasOneBlock()) {
        func.emitOpError("carries complex tensors but does not have a "
                         "single-block body, which sar-decomplexify cannot "
                         "rewrite");
        signalPassFailure();
        return;
      }
      builder.setInsertionPoint(func);
      FunctionDecomplexifier rewriter(func, builder);
      std::string name = func.getName().str();
      // Rename the original out of the way so the replacement can take its
      // symbol name; the original is erased once the rewrite succeeds.
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
