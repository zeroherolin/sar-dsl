//===- SARToLinalg.cpp - Lower SAR ops to linalg --------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
// Lowers SAR structural and element-wise operations to linalg-on-tensors.
// Complex element types are preserved (they lower later through the complex
// dialect); signal-processing ops are handled by ConvertSARSignalToRuntime.
//
// `sar.cumsum` is not an element-wise sweep and does not fit linalg's
// parallel iteration model: it carries a running sum along the scanned
// axis, so it becomes an scf loop nest over tensors instead.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include "sar/Conversion/Passes.h"
#include "sar/Dialect/SAR/IR/SARDialect.h"
#include "sar/Dialect/SAR/IR/SAROps.h"

#include <tuple>
#include <utility>

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_CONVERTSARTOLINALG
#include "sar/Conversion/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;

namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Creates an element-wise linalg.generic with identity maps over `inputs`
/// producing one tensor of type `resultType`. `bodyBuilder` receives the
/// scalar block arguments corresponding to the inputs.
static Value buildElementwiseGeneric(
    OpBuilder &builder, Location loc, RankedTensorType resultType,
    ValueRange inputs,
    function_ref<Value(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  int64_t rank = resultType.getRank();
  Value init = tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                       resultType.getElementType());
  SmallVector<AffineMap> maps(inputs.size() + 1,
                              builder.getMultiDimIdentityMap(rank));
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      builder, loc, TypeRange{resultType}, inputs, ValueRange{init}, maps,
      iterators, [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        // Drop the output block argument.
        Value result =
            bodyBuilder(b, nestedLoc, args.take_front(inputs.size()));
        linalg::YieldOp::create(b, nestedLoc, result);
      });
  return generic.getResult(0);
}

/// Creates a scalar constant of the given float, integer or complex element
/// type from a double value (imaginary part zero for complex types).
static Value buildScalarConstant(OpBuilder &b, Location loc, Type elementType,
                                 double value) {
  if (auto complexTy = dyn_cast<ComplexType>(elementType)) {
    auto floatTy = cast<FloatType>(complexTy.getElementType());
    auto re = b.getFloatAttr(floatTy, value);
    auto im = b.getFloatAttr(floatTy, 0.0);
    return complex::ConstantOp::create(b, loc, complexTy,
                                       b.getArrayAttr({re, im}));
  }
  if (auto intTy = dyn_cast<IntegerType>(elementType))
    return arith::ConstantOp::create(b, loc,
                                     b.getIntegerAttr(intTy, (int64_t)value));
  auto floatTy = cast<FloatType>(elementType);
  return arith::ConstantOp::create(b, loc, b.getFloatAttr(floatTy, value));
}

enum class BinaryKind { Add, Sub, Mul, Div };

static Value buildBinaryScalarOp(OpBuilder &b, Location loc, BinaryKind kind,
                                 Value lhs, Value rhs) {
  Type type = lhs.getType();
  if (isa<FloatType>(type)) {
    switch (kind) {
    case BinaryKind::Add:
      return arith::AddFOp::create(b, loc, lhs, rhs);
    case BinaryKind::Sub:
      return arith::SubFOp::create(b, loc, lhs, rhs);
    case BinaryKind::Mul:
      return arith::MulFOp::create(b, loc, lhs, rhs);
    case BinaryKind::Div:
      return arith::DivFOp::create(b, loc, lhs, rhs);
    }
  }
  if (isa<IntegerType>(type)) {
    switch (kind) {
    case BinaryKind::Add:
      return arith::AddIOp::create(b, loc, lhs, rhs);
    case BinaryKind::Sub:
      return arith::SubIOp::create(b, loc, lhs, rhs);
    case BinaryKind::Mul:
      return arith::MulIOp::create(b, loc, lhs, rhs);
    case BinaryKind::Div:
      return arith::DivSIOp::create(b, loc, lhs, rhs);
    }
  }
  assert(isa<ComplexType>(type) && "expected float, int or complex");
  switch (kind) {
  case BinaryKind::Add:
    return complex::AddOp::create(b, loc, lhs, rhs);
  case BinaryKind::Sub:
    return complex::SubOp::create(b, loc, lhs, rhs);
  case BinaryKind::Mul:
    return complex::MulOp::create(b, loc, lhs, rhs);
  case BinaryKind::Div:
    return complex::DivOp::create(b, loc, lhs, rhs);
  }
  llvm_unreachable("unhandled binary kind");
}

/// Converts a scalar float value to another float type (ext/trunc/identity).
static Value convertFloat(OpBuilder &b, Location loc, Value value,
                          FloatType target) {
  auto source = cast<FloatType>(value.getType());
  if (source == target)
    return value;
  if (source.getWidth() < target.getWidth())
    return arith::ExtFOp::create(b, loc, target, value);
  return arith::TruncFOp::create(b, loc, target, value);
}

/// Converts a scalar between any two real element types. Integers reach the
/// dialect only as index-valued results (`sar.argmax`), so the signed forms
/// are the right ones.
static Value convertReal(OpBuilder &b, Location loc, Value value, Type target) {
  Type source = value.getType();
  if (source == target)
    return value;
  auto sourceInt = dyn_cast<IntegerType>(source);
  auto targetInt = dyn_cast<IntegerType>(target);
  if (sourceInt && targetInt)
    return sourceInt.getWidth() < targetInt.getWidth()
               ? arith::ExtSIOp::create(b, loc, target, value).getResult()
               : arith::TruncIOp::create(b, loc, target, value).getResult();
  if (sourceInt)
    return arith::SIToFPOp::create(b, loc, target, value);
  if (targetInt) {
    // Define float-to-int conversion for every IEEE input. LLVM's fptosi is
    // poison for NaN, infinities and out-of-range values; HLS C++ has the same
    // undefined edge. Feed it a proven-safe value, then select saturation
    // results (NaN -> 0) without ever executing an invalid conversion.
    auto sourceFloat = cast<FloatType>(source);
    unsigned width = targetInt.getWidth();
    double limit = std::ldexp(1.0, width - 1);
    Value lowFloat =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(sourceFloat, -limit));
    Value highFloat =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(sourceFloat, limit));
    Value zeroFloat =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(sourceFloat, 0.0));
    Value below = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                        value, lowFloat);
    Value above = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGE,
                                        value, highFloat);
    Value nan =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UNE, value, value);
    Value invalid = arith::OrIOp::create(
        b, loc, nan, arith::OrIOp::create(b, loc, below, above));
    Value safe =
        arith::SelectOp::create(b, loc, invalid, zeroFloat, value).getResult();
    Value converted = arith::FPToSIOp::create(b, loc, target, safe);
    APInt min = APInt::getSignedMinValue(width);
    APInt max = APInt::getSignedMaxValue(width);
    Value minInt =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(target, min));
    Value maxInt =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(target, max));
    Value zeroInt =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(target, 0));
    Value result =
        arith::SelectOp::create(b, loc, below, minInt, converted).getResult();
    result = arith::SelectOp::create(b, loc, above, maxInt, result);
    return arith::SelectOp::create(b, loc, nan, zeroInt, result);
  }
  return convertFloat(b, loc, value, cast<FloatType>(target));
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : OpRewritePattern<ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto typedValue = dyn_cast<TypedAttr>(op.getValueAttr());
    if (!typedValue)
      return rewriter.notifyMatchFailure(op, "constant value is not typed");
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getType(),
                                                   typedValue);
    return success();
  }
};

template <typename SarOpTy, BinaryKind kind>
struct BinaryOpLowering : OpRewritePattern<SarOpTy> {
  using OpRewritePattern<SarOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(SarOpTy op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getLhs(), op.getRhs()},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          return buildBinaryScalarOp(b, loc, kind, args[0], args[1]);
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename SarOpTy, BinaryKind kind>
struct ScalarBinaryOpLowering : OpRewritePattern<SarOpTy> {
  using OpRewritePattern<SarOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(SarOpTy op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    double scalar = op.getScalar().convertToDouble();
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getInput()},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value cst =
              buildScalarConstant(b, loc, resultType.getElementType(), scalar);
          return buildBinaryScalarOp(b, loc, kind, args[0], cst);
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename SarOpTy, typename MathOpTy>
struct FloatUnaryOpLowering : OpRewritePattern<SarOpTy> {
  using OpRewritePattern<SarOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(SarOpTy op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getInput()},
        [&](OpBuilder &b, Location loc, ValueRange args) -> Value {
          return MathOpTy::create(b, loc, args[0]);
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

using SqrtOpLowering = FloatUnaryOpLowering<SqrtOp, math::SqrtOp>;
using ExpOpLowering = FloatUnaryOpLowering<ExpOp, math::ExpOp>;
using LogOpLowering = FloatUnaryOpLowering<LogOp, math::LogOp>;
using CosOpLowering = FloatUnaryOpLowering<CosOp, math::CosOp>;
using SinOpLowering = FloatUnaryOpLowering<SinOp, math::SinOp>;

struct AbsOpLowering : OpRewritePattern<AbsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AbsOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getInput()},
        [&](OpBuilder &b, Location loc, ValueRange args) -> Value {
          if (isa<ComplexType>(args[0].getType()))
            return complex::AbsOp::create(b, loc, args[0]);
          return math::AbsFOp::create(b, loc, args[0]);
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct CmpOpLowering : OpRewritePattern<CmpOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CmpOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto predicate = llvm::StringSwitch<std::optional<arith::CmpFPredicate>>(
                         op.getPredicate())
                         .Case("eq", arith::CmpFPredicate::OEQ)
                         .Case("ne", arith::CmpFPredicate::UNE)
                         .Case("lt", arith::CmpFPredicate::OLT)
                         .Case("le", arith::CmpFPredicate::OLE)
                         .Case("gt", arith::CmpFPredicate::OGT)
                         .Case("ge", arith::CmpFPredicate::OGE)
                         .Default(std::nullopt);
    if (!predicate)
      return op.emitOpError("unknown comparison predicate");

    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getLhs(), op.getRhs()},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value held =
              arith::CmpFOp::create(b, loc, *predicate, args[0], args[1]);
          // The mask is a float so it can flow through the same ops as the
          // data it selects between.
          Value one =
              buildScalarConstant(b, loc, resultType.getElementType(), 1.0);
          Value zero =
              buildScalarConstant(b, loc, resultType.getElementType(), 0.0);
          return arith::SelectOp::create(b, loc, held, one, zero).getResult();
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct WhereOpLowering : OpRewritePattern<WhereOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(WhereOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType,
        ValueRange{op.getMask(), op.getLhs(), op.getRhs()},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value zero = buildScalarConstant(
              b, loc,
              cast<RankedTensorType>(op.getMask().getType()).getElementType(),
              0.0);
          // A mask carries 0.0 or 1.0, so the unordered form is the safe
          // reading: anything that is not exactly zero selects the left.
          Value held = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UNE,
                                             args[0], zero);
          return arith::SelectOp::create(b, loc, held, args[1], args[2])
              .getResult();
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ComplexOpLowering : OpRewritePattern<ComplexOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ComplexOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto complexTy = cast<ComplexType>(resultType.getElementType());
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getRe(), op.getIm()},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          return complex::CreateOp::create(b, loc, complexTy, args[0], args[1])
              .getResult();
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConjOpLowering : OpRewritePattern<ConjOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConjOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getInput()},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          return complex::ConjOp::create(b, loc, args[0]).getResult();
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// `sar.real` / `sar.imag`: one plane out of a complex tensor.
template <typename SarOpTy, typename ComplexOpTy>
struct ComplexPartOpLowering : OpRewritePattern<SarOpTy> {
  using OpRewritePattern<SarOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(SarOpTy op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getInput()},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          return ComplexOpTy::create(b, loc, args[0]).getResult();
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct Atan2OpLowering : OpRewritePattern<Atan2Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Atan2Op op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getY(), op.getX()},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          return math::Atan2Op::create(b, loc, args[0], args[1]).getResult();
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct CastOpLowering : OpRewritePattern<CastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    Type outElem = resultType.getElementType();
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getInput()},
        [&](OpBuilder &b, Location loc, ValueRange args) -> Value {
          Value in = args[0];
          Type inElem = in.getType();
          if (auto outComplex = dyn_cast<ComplexType>(outElem)) {
            auto outFloat = cast<FloatType>(outComplex.getElementType());
            Value re, im;
            if (auto inComplex = dyn_cast<ComplexType>(inElem)) {
              re = convertFloat(b, loc, complex::ReOp::create(b, loc, in),
                                outFloat);
              im = convertFloat(b, loc, complex::ImOp::create(b, loc, in),
                                outFloat);
            } else {
              re = convertReal(b, loc, in, outFloat);
              im = arith::ConstantOp::create(b, loc,
                                             b.getFloatAttr(outFloat, 0.0));
            }
            return complex::CreateOp::create(b, loc, outComplex, re, im);
          }
          if (auto inComplex = dyn_cast<ComplexType>(inElem))
            in = complex::ReOp::create(b, loc, in);
          return convertReal(b, loc, in, outElem);
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ReduceOpLowering : OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    Type elementType = resultType.getElementType();
    StringRef kind = op.getKind();
    if (kind != "sum" && kind != "max" && kind != "min")
      return op.emitOpError("unknown reduction kind '") << kind << "'";

    // The identity has to be the one the combinator leaves untouched, or
    // the first element folded in would be lost.
    double identity = 0.0;
    if (kind == "max")
      identity = -std::numeric_limits<double>::infinity();
    else if (kind == "min")
      identity = std::numeric_limits<double>::infinity();

    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         elementType);
    Value seed = buildScalarConstant(rewriter, loc, elementType, identity);
    Value filled = linalg::FillOp::create(rewriter, loc, ValueRange{seed},
                                          ValueRange{init})
                       .getResult(0);

    auto combine = [&](OpBuilder &b, Location nested, Value lhs,
                       Value rhs) -> Value {
      if (kind == "sum") {
        if (isa<ComplexType>(elementType))
          return complex::AddOp::create(b, nested, lhs, rhs);
        return arith::AddFOp::create(b, nested, lhs, rhs);
      }
      auto predicate =
          kind == "max" ? arith::CmpFPredicate::OGT : arith::CmpFPredicate::OLT;
      Value wins = arith::CmpFOp::create(b, nested, predicate, lhs, rhs);
      Value selected = arith::SelectOp::create(b, nested, wins, lhs, rhs);
      Value lhsNan =
          arith::CmpFOp::create(b, nested, arith::CmpFPredicate::UNE, lhs, lhs);
      Value rhsNan =
          arith::CmpFOp::create(b, nested, arith::CmpFPredicate::UNE, rhs, rhs);
      selected = arith::SelectOp::create(b, nested, rhsNan, rhs, selected);
      return arith::SelectOp::create(b, nested, lhsNan, lhs, selected);
    };

    auto reduce = linalg::ReduceOp::create(
        rewriter, loc, ValueRange{op.getInput()}, ValueRange{filled},
        ArrayRef<int64_t>{static_cast<int64_t>(op.getDim())},
        [&](OpBuilder &b, Location nested, ValueRange args) {
          linalg::YieldOp::create(b, nested,
                                  combine(b, nested, args[0], args[1]));
        });
    rewriter.replaceOp(op, reduce.getResults());
    return success();
  }
};

struct ArgMaxOpLowering : OpRewritePattern<ArgMaxOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ArgMaxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto resultType = cast<RankedTensorType>(op.getType());
    Type elementType = inputType.getElementType();
    int64_t dim = op.getDim();
    int64_t rank = inputType.getRank();

    // Carry the running best value alongside the index: a reduction that
    // yielded the index alone could not tell which candidate to keep.
    Value valueInit = tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), elementType);
    Value lowest = buildScalarConstant(
        rewriter, loc, elementType, -std::numeric_limits<double>::infinity());
    Value values = linalg::FillOp::create(rewriter, loc, ValueRange{lowest},
                                          ValueRange{valueInit})
                       .getResult(0);
    Value indexInit = tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType());
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(resultType.getElementType(), 0));
    Value indices = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                           ValueRange{indexInit})
                        .getResult(0);

    SmallVector<AffineExpr> inputExprs, resultExprs;
    for (int64_t d = 0; d < rank; ++d) {
      inputExprs.push_back(rewriter.getAffineDimExpr(d));
      if (d != dim)
        resultExprs.push_back(rewriter.getAffineDimExpr(d));
    }
    auto inputMap = AffineMap::get(rank, 0, inputExprs, op.getContext());
    auto resultMap = AffineMap::get(rank, 0, resultExprs, op.getContext());
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    iterators[dim] = utils::IteratorType::reduction;

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{values.getType(), indices.getType()},
        ValueRange{op.getInput()}, ValueRange{values, indices},
        ArrayRef<AffineMap>{inputMap, resultMap, resultMap}, iterators,
        [&](OpBuilder &b, Location nested, ValueRange args) {
          Value candidate = args[0], best = args[1], bestIndex = args[2];
          Value position = linalg::IndexOp::create(b, nested, dim);
          position = arith::IndexCastOp::create(
              b, nested, resultType.getElementType(), position);
          Value numericBetter = arith::CmpFOp::create(
              b, nested, arith::CmpFPredicate::OGT, candidate, best);
          Value candidateNan = arith::CmpFOp::create(
              b, nested, arith::CmpFPredicate::UNE, candidate, candidate);
          Value bestNan = arith::CmpFOp::create(
              b, nested, arith::CmpFPredicate::UNE, best, best);
          Value falseValue =
              arith::ConstantOp::create(b, nested, b.getBoolAttr(false));
          Value candidateFinite = arith::CmpIOp::create(
              b, nested, arith::CmpIPredicate::eq, candidateNan, falseValue);
          Value bestFinite = arith::CmpIOp::create(
              b, nested, arith::CmpIPredicate::eq, bestNan, falseValue);
          Value nanWins =
              arith::AndIOp::create(b, nested, candidateNan, bestFinite);
          Value finiteWins = arith::AndIOp::create(
              b, nested,
              arith::AndIOp::create(b, nested, candidateFinite, bestFinite),
              numericBetter);
          Value better = arith::OrIOp::create(b, nested, nanWins, finiteWins);
          Value newBest =
              arith::SelectOp::create(b, nested, better, candidate, best);
          Value newIndex =
              arith::SelectOp::create(b, nested, better, position, bestIndex);
          linalg::YieldOp::create(b, nested, ValueRange{newBest, newIndex});
        });
    rewriter.replaceOp(op, generic.getResult(1));
    return success();
  }
};

struct SliceOpLowering : OpRewritePattern<SliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();
    ArrayRef<int64_t> sliceOffsets = op.getOffsets();
    ArrayRef<int64_t> sliceStrides = op.getStrides();

    // A slice is a sweep whose source index is an affine map of the loop
    // indices (strides are >= 1 by the verifier). `tensor.extract_slice`
    // would be shorter, but it bufferizes to a view, and a view that
    // survives is a value the dataflow backend cannot place in a task --
    // and elsewhere a layout every later pass has to carry along.
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    // The source is read through `tensor.extract` rather than an indexing
    // map: a map would make linalg infer the operand's shape from the
    // iteration domain, which is the result's, not the source's.
    Location loc = op.getLoc();
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType());
    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{}, ValueRange{init},
        ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(rank)}, iterators,
        [&](OpBuilder &b, Location nested, ValueRange) {
          SmallVector<Value> indices;
          for (int64_t d = 0; d < rank; ++d) {
            Value index = linalg::IndexOp::create(b, nested, d);
            if (sliceStrides[d] != 1)
              index = arith::MulIOp::create(
                  b, nested, index,
                  arith::ConstantIndexOp::create(b, nested, sliceStrides[d]));
            if (sliceOffsets[d] != 0)
              index = arith::AddIOp::create(
                  b, nested, index,
                  arith::ConstantIndexOp::create(b, nested, sliceOffsets[d]));
            indices.push_back(index);
          }
          linalg::YieldOp::create(
              b, nested,
              tensor::ExtractOp::create(b, nested, op.getInput(), indices)
                  .getResult());
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

static Value getClampedOffset(OpBuilder &builder, Location loc,
                              Value offsetTensor, int64_t maximum) {
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value raw =
      tensor::ExtractOp::create(builder, loc, offsetTensor, ValueRange{zero});
  raw = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), raw);
  Value high = arith::ConstantIndexOp::create(builder, loc, maximum);
  return arith::MinSIOp::create(
      builder, loc, arith::MaxSIOp::create(builder, loc, raw, zero), high);
}

struct DynamicSliceOpLowering : OpRewritePattern<DynamicSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicSliceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();
    ArrayRef<int64_t> sizes = op.getSizes();
    ArrayRef<int64_t> strides = op.getStrides();
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType());
    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{}, ValueRange{init},
        ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(rank)}, iterators,
        [&](OpBuilder &builder, Location nested, ValueRange) {
          SmallVector<Value> indices;
          for (int64_t dim = 0; dim < rank; ++dim) {
            int64_t span = (sizes[dim] - 1) * strides[dim] + 1;
            Value offset =
                getClampedOffset(builder, nested, op.getOffsets()[dim],
                                 inputType.getDimSize(dim) - span);
            Value index = linalg::IndexOp::create(builder, nested, dim);
            if (strides[dim] != 1)
              index = arith::MulIOp::create(builder, nested, index,
                                            arith::ConstantIndexOp::create(
                                                builder, nested, strides[dim]));
            indices.push_back(
                arith::AddIOp::create(builder, nested, index, offset));
          }
          linalg::YieldOp::create(
              builder, nested,
              tensor::ExtractOp::create(builder, nested, op.getInput(), indices)
                  .getResult());
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct DynamicUpdateSliceOpLowering : OpRewritePattern<DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicUpdateSliceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    auto updateType = cast<RankedTensorType>(op.getUpdate().getType());
    int64_t rank = resultType.getRank();
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType());
    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{}, ValueRange{init},
        ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(rank)}, iterators,
        [&](OpBuilder &builder, Location nested, ValueRange) {
          SmallVector<Value> inputIndices;
          SmallVector<Value> updateIndices;
          Value inside;
          Value zero = arith::ConstantIndexOp::create(builder, nested, 0);
          for (int64_t dim = 0; dim < rank; ++dim) {
            Value index = linalg::IndexOp::create(builder, nested, dim);
            Value offset = getClampedOffset(
                builder, nested, op.getOffsets()[dim],
                resultType.getDimSize(dim) - updateType.getDimSize(dim));
            Value relative =
                arith::SubIOp::create(builder, nested, index, offset);
            Value extent = arith::ConstantIndexOp::create(
                builder, nested, updateType.getDimSize(dim));
            Value inDimension = arith::AndIOp::create(
                builder, nested,
                arith::CmpIOp::create(
                    builder, nested, arith::CmpIPredicate::sge, relative, zero),
                arith::CmpIOp::create(builder, nested,
                                      arith::CmpIPredicate::slt, relative,
                                      extent));
            inside = inside ? arith::AndIOp::create(builder, nested, inside,
                                                    inDimension)
                                  .getResult()
                            : inDimension;
            Value last = arith::ConstantIndexOp::create(
                builder, nested, updateType.getDimSize(dim) - 1);
            updateIndices.push_back(arith::MinSIOp::create(
                builder, nested,
                arith::MaxSIOp::create(builder, nested, relative, zero), last));
            inputIndices.push_back(index);
          }
          Value original = tensor::ExtractOp::create(
              builder, nested, op.getInput(), inputIndices);
          Value replacement = tensor::ExtractOp::create(
              builder, nested, op.getUpdate(), updateIndices);
          linalg::YieldOp::create(builder, nested,
                                  arith::SelectOp::create(builder, nested,
                                                          inside, replacement,
                                                          original)
                                      .getResult());
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct ConcatOpLowering : OpRewritePattern<ConcatOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();
    int64_t dim = op.getDim();

    // One sweep that reads whichever operand covers the position it is
    // writing. `tensor.concat`, and equally a pair of `insert_slice`s,
    // bufferize to `memref.copy` -- a call into the MLIR runtime support
    // library that neither the standalone kernel nor an HLS target links
    // against -- so the selection is done in the body instead.
    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(op.getRhs().getType());
    int64_t split = lhsType.getDimSize(dim);

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType());
    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{}, ValueRange{init},
        ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(rank)}, iterators,
        [&](OpBuilder &b, Location nested, ValueRange) {
          SmallVector<Value> lhsIndices, rhsIndices;
          Value position;
          for (int64_t d = 0; d < rank; ++d) {
            Value index = linalg::IndexOp::create(b, nested, d);
            if (d != dim) {
              lhsIndices.push_back(index);
              rhsIndices.push_back(index);
              continue;
            }
            position = index;
            Value splitCst = arith::ConstantIndexOp::create(b, nested, split);
            // Both reads are clamped into their own operand, so the one
            // that is not selected still addresses a valid element.
            Value lhsMax = arith::ConstantIndexOp::create(b, nested, split - 1);
            Value rhsMax = arith::ConstantIndexOp::create(
                b, nested, rhsType.getDimSize(dim) - 1);
            Value zero = arith::ConstantIndexOp::create(b, nested, 0);
            lhsIndices.push_back(
                arith::MinSIOp::create(b, nested, index, lhsMax));
            Value shifted = arith::SubIOp::create(b, nested, index, splitCst);
            shifted = arith::MaxSIOp::create(b, nested, shifted, zero);
            rhsIndices.push_back(
                arith::MinSIOp::create(b, nested, shifted, rhsMax));
          }
          Value splitCst = arith::ConstantIndexOp::create(b, nested, split);
          // Indices are non-negative, so the unsigned comparison is the
          // one that says what is meant.
          Value fromLhs = arith::CmpIOp::create(
              b, nested, arith::CmpIPredicate::ult, position, splitCst);
          Value left =
              tensor::ExtractOp::create(b, nested, op.getLhs(), lhsIndices);
          Value right =
              tensor::ExtractOp::create(b, nested, op.getRhs(), rhsIndices);
          linalg::YieldOp::create(
              b, nested,
              arith::SelectOp::create(b, nested, fromLhs, left, right)
                  .getResult());
        });
    Value result = generic.getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct PadOpLowering : OpRewritePattern<PadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    int64_t rank = resultType.getRank();
    ArrayRef<int64_t> low = op.getLow();

    // Written as one sweep that decides per element rather than
    // `tensor.pad`, whose bufferization emits `memref.copy` -- a call into
    // the MLIR runtime support library that neither the standalone kernel
    // nor an HLS target links against.
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType());
    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{}, ValueRange{init},
        ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(rank)}, iterators,
        [&](OpBuilder &b, Location nested, ValueRange) {
          Value inside;
          SmallVector<Value> indices;
          for (int64_t d = 0; d < rank; ++d) {
            Value index = linalg::IndexOp::create(b, nested, d);
            Value shifted = arith::SubIOp::create(
                b, nested, index,
                arith::ConstantIndexOp::create(b, nested, low[d]));
            // Clamp so the read addresses a valid element even where the
            // result takes the padding value instead.
            Value zero = arith::ConstantIndexOp::create(b, nested, 0);
            Value last = arith::ConstantIndexOp::create(
                b, nested, inputType.getDimSize(d) - 1);
            Value clamped = arith::MinSIOp::create(
                b, nested, arith::MaxSIOp::create(b, nested, shifted, zero),
                last);
            indices.push_back(clamped);

            Value inRange = arith::AndIOp::create(
                b, nested,
                arith::CmpIOp::create(b, nested, arith::CmpIPredicate::sge,
                                      shifted, zero),
                arith::CmpIOp::create(b, nested, arith::CmpIPredicate::sle,
                                      shifted, last));
            inside = inside ? arith::AndIOp::create(b, nested, inside, inRange)
                                  .getResult()
                            : inRange;
          }
          Value element =
              tensor::ExtractOp::create(b, nested, op.getInput(), indices);
          Value fill =
              buildScalarConstant(b, nested, resultType.getElementType(),
                                  op.getValue().convertToDouble());
          linalg::YieldOp::create(
              b, nested,
              arith::SelectOp::create(b, nested, inside, element, fill)
                  .getResult());
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct ReverseOpLowering : OpRewritePattern<ReverseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReverseOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();
    int64_t dim = op.getDim();
    int64_t size = resultType.getDimSize(dim);

    // out[..., i, ...] = in[..., size - 1 - i, ...]: an affine map of the
    // loop indices, so the sweep stays analyzable.
    SmallVector<AffineExpr> sourceExprs;
    for (int64_t d = 0; d < rank; ++d)
      sourceExprs.push_back(d == dim
                                ? rewriter.getAffineConstantExpr(size - 1) -
                                      rewriter.getAffineDimExpr(d)
                                : rewriter.getAffineDimExpr(d));
    auto sourceMap = AffineMap::get(rank, 0, sourceExprs, op.getContext());
    auto identity = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Location loc = op.getLoc();
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType());
    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{op.getInput()},
        ValueRange{init}, ArrayRef<AffineMap>{sourceMap, identity}, iterators,
        [&](OpBuilder &b, Location nested, ValueRange args) {
          linalg::YieldOp::create(b, nested, args.front());
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct TransposeOpLowering : OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType());
    auto transpose = linalg::TransposeOp::create(rewriter, loc, op.getInput(),
                                                 init, ArrayRef<int64_t>{1, 0});
    rewriter.replaceOp(op, transpose->getResults());
    return success();
  }
};

struct BroadcastOpLowering : OpRewritePattern<BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType());
    // The vector lies along axis `dim`; the other axis is the broadcasted
    // (added) dimension.
    int64_t addedDim = 1 - op.getDim();
    auto broadcast = linalg::BroadcastOp::create(
        rewriter, loc, op.getInput(), init, ArrayRef<int64_t>{addedDim});
    rewriter.replaceOp(op, broadcast->getResults());
    return success();
  }
};

struct FFTShiftOpLowering : OpRewritePattern<FFTShiftOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FFTShiftOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();
    int64_t dim = op.getDim();
    int64_t size = resultType.getDimSize(dim);
    // fftshift:  out[i] = in[(i + ceil(n/2)) mod n]
    // ifftshift: out[i] = in[(i + floor(n/2)) mod n]
    int64_t offset = op.getInverse() ? size / 2 : size - size / 2;

    Value input = op.getInput();
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType());

    // The shift rides on the input's indexing map rather than on a gather
    // in the body. Both express `out[i] = in[(i + offset) mod n]`, but a
    // map keeps the read an affine function of the loop indices: it lowers
    // to `affine.load`, so an HLS flow can reason about the stride, and a
    // later fusion composes the shift into the map it fuses with instead of
    // leaving a data-dependent `memref.load` a corner-turn rewrite cannot
    // see through.
    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineExpr> sourceExprs;
    for (int64_t d = 0; d < rank; ++d) {
      AffineExpr expr = getAffineDimExpr(d, ctx);
      if (d == dim)
        expr = (expr + offset) % size;
      sourceExprs.push_back(expr);
    }
    SmallVector<AffineMap> maps{AffineMap::get(rank, 0, sourceExprs, ctx),
                                rewriter.getMultiDimIdentityMap(rank)};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{input},
        ValueRange{init}, maps, iterators,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          linalg::YieldOp::create(b, nestedLoc, args[0]);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 2-D gather
//===----------------------------------------------------------------------===//

/// floor(x) as i64. fptosi truncates toward zero, so negative fractions
/// need the -1 fixup; out-of-range detection below relies on the true
/// floor (truncation would fold a -0.2 position onto sample 0).
static Value emitFloorToI64(OpBuilder &b, Location loc, Value x) {
  Value truncI = arith::FPToSIOp::create(b, loc, b.getI64Type(), x);
  Value truncF = arith::SIToFPOp::create(b, loc, b.getF64Type(), truncI);
  Value negFrac =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, x, truncF);
  Value one = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(1));
  Value fixed = arith::SubIOp::create(b, loc, truncI, one);
  return arith::SelectOp::create(b, loc, negFrac, fixed, truncI);
}

static std::pair<Value, Value> sanitizeGatherPosition(OpBuilder &b,
                                                      Location loc, Value x) {
  Value finite = math::IsFiniteOp::create(b, loc, x);
  Value lo = arith::CmpFOp::create(
      b, loc, arith::CmpFPredicate::OGT, x,
      arith::ConstantOp::create(b, loc, b.getF64FloatAttr(-0x1p62)));
  Value hi = arith::CmpFOp::create(
      b, loc, arith::CmpFPredicate::OLT, x,
      arith::ConstantOp::create(b, loc, b.getF64FloatAttr(0x1p62)));
  Value valid = arith::AndIOp::create(b, loc, finite,
                                      arith::AndIOp::create(b, loc, lo, hi));
  Value safe = arith::SelectOp::create(
      b, loc, valid, x,
      arith::ConstantOp::create(b, loc, b.getF64FloatAttr(0.0)));
  return {safe, valid};
}

/// Emits the tap arithmetic of a 2-D gather at (rowPos, colPos), f64.
/// `loadAt` supplies the sample at *clamped* index values as an (re, im)
/// pair already widened to f64; taps outside [0, rows) x [0, cols) are
/// masked to zero weight under the `zero` boundary and read the clamped
/// sample under `edge`. Returns the (re, im) accumulators in f64.
static std::pair<Value, Value> emitGather2DTaps(
    OpBuilder &b, Location loc, Value rowPos, Value colPos, int64_t rows,
    int64_t cols, StringRef kernel, StringRef boundary,
    llvm::function_ref<std::pair<Value, Value>(Value, Value)> loadAt) {
  auto i64 = [&](int64_t v) {
    return arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(v))
        .getResult();
  };
  auto f64 = [&](double v) {
    return arith::ConstantOp::create(b, loc, b.getF64FloatAttr(v)).getResult();
  };
  Value zeroI = i64(0), oneI = i64(1);
  Value rowMax = i64(rows - 1), colMax = i64(cols - 1);
  Value rowsI = i64(rows), colsI = i64(cols);
  Value zeroF = f64(0.0), oneF = f64(1.0);
  auto [safeRow, validRow] = sanitizeGatherPosition(b, loc, rowPos);
  auto [safeCol, validCol] = sanitizeGatherPosition(b, loc, colPos);
  rowPos = safeRow;
  colPos = safeCol;
  Value validPosition = arith::AndIOp::create(b, loc, validRow, validCol);
  auto finish = [&](Value re, Value im) {
    return std::pair<Value, Value>{
        arith::SelectOp::create(b, loc, validPosition, re, zeroF),
        arith::SelectOp::create(b, loc, validPosition, im, zeroF)};
  };

  auto clampToIndex = [&](Value idx, Value max) {
    Value clamped = arith::MinSIOp::create(
        b, loc, arith::MaxSIOp::create(b, loc, idx, zeroI), max);
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), clamped)
        .getResult();
  };
  auto inRange = [&](Value idx, Value extent) {
    Value lo =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, idx, zeroI);
    Value hi =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, idx, extent);
    return arith::AndIOp::create(b, loc, lo, hi).getResult();
  };

  // One tap: accumulate weight * sample, masking the weight when the tap
  // falls outside the data under the `zero` boundary.
  Value accRe = zeroF, accIm = zeroF;
  auto tap = [&](Value rowIdx, Value colIdx, Value weight) {
    if (boundary == "zero") {
      Value inb = arith::AndIOp::create(b, loc, inRange(rowIdx, rowsI),
                                        inRange(colIdx, colsI));
      weight = arith::SelectOp::create(b, loc, inb, weight, zeroF);
    }
    auto [re, im] =
        loadAt(clampToIndex(rowIdx, rowMax), clampToIndex(colIdx, colMax));
    Value wRe = arith::MulFOp::create(b, loc, re, weight);
    Value wIm = arith::MulFOp::create(b, loc, im, weight);
    accRe = arith::AddFOp::create(b, loc, accRe, wRe);
    accIm = arith::AddFOp::create(b, loc, accIm, wIm);
  };

  if (kernel == "nearest") {
    Value half = f64(0.5);
    Value rowIdx =
        emitFloorToI64(b, loc, arith::AddFOp::create(b, loc, rowPos, half));
    Value colIdx =
        emitFloorToI64(b, loc, arith::AddFOp::create(b, loc, colPos, half));
    tap(rowIdx, colIdx, oneF);
    return finish(accRe, accIm);
  }

  // Bilinear: fractional parts weight the four surrounding samples.
  Value row0 = emitFloorToI64(b, loc, rowPos);
  Value col0 = emitFloorToI64(b, loc, colPos);
  Value fr = arith::SubFOp::create(
      b, loc, rowPos, arith::SIToFPOp::create(b, loc, b.getF64Type(), row0));
  Value fc = arith::SubFOp::create(
      b, loc, colPos, arith::SIToFPOp::create(b, loc, b.getF64Type(), col0));
  Value frInv = arith::SubFOp::create(b, loc, oneF, fr);
  Value fcInv = arith::SubFOp::create(b, loc, oneF, fc);
  Value row1 = arith::AddIOp::create(b, loc, row0, oneI);
  Value col1 = arith::AddIOp::create(b, loc, col0, oneI);
  tap(row0, col0, arith::MulFOp::create(b, loc, frInv, fcInv));
  tap(row0, col1, arith::MulFOp::create(b, loc, frInv, fc));
  tap(row1, col0, arith::MulFOp::create(b, loc, fr, fcInv));
  tap(row1, col1, arith::MulFOp::create(b, loc, fr, fc));
  return finish(accRe, accIm);
}

/// `sar.gather2d` on complex tensors (the execution path, where complex
/// elements survive to linalg).
struct Gather2DOpLowering : OpRewritePattern<Gather2DOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Gather2DOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto dataType = cast<RankedTensorType>(op.getData().getType());
    auto complexTy = cast<ComplexType>(resultType.getElementType());
    auto floatTy = cast<FloatType>(complexTy.getElementType());
    int64_t rows = dataType.getDimSize(0), cols = dataType.getDimSize(1);

    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType,
        ValueRange{op.getRows(), op.getCols()},
        [&](OpBuilder &b, Location loc, ValueRange args) -> Value {
          auto loadAt = [&](Value r, Value c) -> std::pair<Value, Value> {
            Value sample = tensor::ExtractOp::create(b, loc, op.getData(),
                                                     ValueRange{r, c});
            Value re = complex::ReOp::create(b, loc, sample);
            Value im = complex::ImOp::create(b, loc, sample);
            return {convertFloat(b, loc, re, b.getF64Type()),
                    convertFloat(b, loc, im, b.getF64Type())};
          };
          auto [accRe, accIm] =
              emitGather2DTaps(b, loc, args[0], args[1], rows, cols,
                               op.getKernel(), op.getBoundary(), loadAt);
          return complex::CreateOp::create(
              b, loc, complexTy, convertFloat(b, loc, accRe, floatTy),
              convertFloat(b, loc, accIm, floatTy));
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// `sar.gather2d_split` on (re, im) planes (the affine/HLS path). One
/// generic with two results, so the taps are gathered once per element.
struct Gather2DSplitOpLowering : OpRewritePattern<Gather2DSplitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Gather2DSplitOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getOutRe().getType());
    auto planeType = cast<RankedTensorType>(op.getRe().getType());
    auto floatTy = cast<FloatType>(resultType.getElementType());
    int64_t rows = planeType.getDimSize(0), cols = planeType.getDimSize(1);
    int64_t rank = resultType.getRank();

    Value initRe =
        tensor::EmptyOp::create(rewriter, loc, resultType.getShape(), floatTy);
    Value initIm =
        tensor::EmptyOp::create(rewriter, loc, resultType.getShape(), floatTy);
    SmallVector<AffineMap> maps(4, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType, resultType},
        ValueRange{op.getRows(), op.getCols()}, ValueRange{initRe, initIm},
        maps, iterators, [&](OpBuilder &b, Location nested, ValueRange args) {
          auto loadAt = [&](Value r, Value c) -> std::pair<Value, Value> {
            Value re = tensor::ExtractOp::create(b, nested, op.getRe(),
                                                 ValueRange{r, c});
            Value im = tensor::ExtractOp::create(b, nested, op.getIm(),
                                                 ValueRange{r, c});
            return {convertFloat(b, nested, re, b.getF64Type()),
                    convertFloat(b, nested, im, b.getF64Type())};
          };
          auto [accRe, accIm] =
              emitGather2DTaps(b, nested, args[0], args[1], rows, cols,
                               op.getKernel(), op.getBoundary(), loadAt);
          linalg::YieldOp::create(
              b, nested,
              ValueRange{convertFloat(b, nested, accRe, floatTy),
                         convertFloat(b, nested, accIm, floatTy)});
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct CumsumOpLowering : OpRewritePattern<CumsumOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CumsumOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    Type elementType = resultType.getElementType();
    int64_t rank = resultType.getRank();
    int64_t dim = op.getDim();
    int64_t scanLen = resultType.getDimSize(dim);
    int64_t lineLen = rank == 1 ? 1 : resultType.getDimSize(1 - dim);

    // A scan is sequential along `dim`, so it cannot be a linalg.generic:
    // the lines are swept in parallel, each carrying a running sum.
    // out[k] = out[k - 1] + in[k] reads the running sum back out of the
    // result rather than carrying it in an iter_arg: the read-back form
    // survives bufferization unchanged, so both backends lower the same
    // loop nest.
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value lineBound = arith::ConstantIndexOp::create(rewriter, loc, lineLen);
    Value scanBound = arith::ConstantIndexOp::create(rewriter, loc, scanLen);
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         elementType);

    // Map a (line, position-along-dim) pair onto tensor indices; a rank-1
    // tensor is a single line, so only the scan position indexes it.
    auto indicesFor = [&](Value line, Value scan) {
      if (rank == 1)
        return SmallVector<Value>{scan};
      SmallVector<Value> indices(2);
      indices[dim] = scan;
      indices[1 - dim] = line;
      return indices;
    };

    auto outer = scf::ForOp::create(
        rewriter, loc, zero, lineBound, one, ValueRange{init},
        [&](OpBuilder &b, Location nested, Value line, ValueRange iter) {
          // The first element of a line is its own prefix sum.
          Value first = tensor::ExtractOp::create(b, nested, op.getInput(),
                                                  indicesFor(line, zero));
          Value seeded = tensor::InsertOp::create(b, nested, first, iter[0],
                                                  indicesFor(line, zero));
          auto inner = scf::ForOp::create(
              b, nested, one, scanBound, one, ValueRange{seeded},
              [&](OpBuilder &ib, Location iloc, Value k, ValueRange it) {
                Value back = arith::SubIOp::create(ib, iloc, k, one);
                Value previous = tensor::ExtractOp::create(
                    ib, iloc, it[0], indicesFor(line, back));
                Value element = tensor::ExtractOp::create(
                    ib, iloc, op.getInput(), indicesFor(line, k));
                Value running = buildBinaryScalarOp(ib, iloc, BinaryKind::Add,
                                                    previous, element);
                Value updated = tensor::InsertOp::create(
                    ib, iloc, running, it[0], indicesFor(line, k));
                scf::YieldOp::create(ib, iloc, updated);
              });
          scf::YieldOp::create(b, nested, inner.getResult(0));
        });
    rewriter.replaceOp(op, outer.getResult(0));
    return success();
  }
};

/// Returns `lhs` and `rhs` as an ordered (low, high) pair, with NaN ordered
/// after every number so the compare-exchange networks built on this keep a
/// consistent total order.
static std::pair<Value, Value>
orderedFloatPair(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  Value rhsIsNaN =
      arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::UNE, rhs, rhs);
  Value orderedLE =
      arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLE, lhs, rhs);
  Value lhsFirst = arith::OrIOp::create(builder, loc, rhsIsNaN, orderedLE);
  return {arith::SelectOp::create(builder, loc, lhsFirst, lhs, rhs),
          arith::SelectOp::create(builder, loc, lhsFirst, rhs, lhs)};
}

struct RankFilterOpLowering : OpRewritePattern<RankFilterOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RankFilterOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();
    int64_t dim = op.getDim();
    int64_t window = op.getWindow();
    int64_t orderRank = op.getRank();
    int64_t half = window / 2;
    int64_t extent = resultType.getDimSize(dim);

    // Every output element is independent, so the sweep stays parallel;
    // only the window sort lives in the body.
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{},
        [&](OpBuilder &b, Location loc, ValueRange) -> Value {
          Value position = linalg::IndexOp::create(b, loc, dim);
          Value line =
              rank == 1 ? Value() : linalg::IndexOp::create(b, loc, 1 - dim);
          Value low = arith::ConstantIndexOp::create(b, loc, 0);
          Value high = arith::ConstantIndexOp::create(b, loc, extent - 1);

          // Gather the window, replicating the edge sample beyond the ends.
          SmallVector<Value> taps;
          taps.reserve(window);
          for (int64_t t = 0; t < window; ++t) {
            Value offset = arith::ConstantIndexOp::create(b, loc, t - half);
            Value raw = arith::AddIOp::create(b, loc, position, offset);
            Value clamped = arith::MinSIOp::create(
                b, loc, arith::MaxSIOp::create(b, loc, raw, low), high);
            SmallVector<Value> indices(rank);
            indices[dim] = clamped;
            if (rank == 2)
              indices[1 - dim] = line;
            taps.push_back(
                tensor::ExtractOp::create(b, loc, op.getInput(), indices));
          }

          // `window` is a compile-time constant, so the selection unrolls
          // into a straight-line compare-exchange network: no control flow
          // in the body, and the requested order statistic is one of its
          // outputs.
          //
          // Only that one statistic is wanted, so the network stops as soon
          // as its slot is settled rather than sorting the whole window. A
          // pass that carries the largest tap to the top settles one slot
          // from the top, so `window - orderRank` of them settle position
          // `orderRank`; carrying the smallest down settles it in
          // `orderRank + 1`. Running whichever is shorter costs
          // `window * min(orderRank + 1, window - orderRank)` compares
          // instead of `window^2 / 2` -- for a minimum or maximum filter,
          // one pass instead of the whole sort.
          int64_t fromTop = window - orderRank;
          int64_t fromBottom = orderRank + 1;
          if (fromTop <= fromBottom) {
            for (int64_t i = 0; i < fromTop; ++i)
              for (int64_t j = 0; j + 1 < window - i; ++j)
                std::tie(taps[j], taps[j + 1]) =
                    orderedFloatPair(b, loc, taps[j], taps[j + 1]);
          } else {
            for (int64_t i = 0; i < fromBottom; ++i)
              for (int64_t j = window - 1; j > i; --j)
                std::tie(taps[j - 1], taps[j]) =
                    orderedFloatPair(b, loc, taps[j - 1], taps[j]);
          }
          return taps[orderRank];
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Batcher odd-even mergesort stages as (p, k) pairs, outermost first.
/// A line of `n` takes O(log^2 n) stages holding O(n log^2 n) comparators
/// in total, against the `n^2` of a compare-exchange bubble sort.
static SmallVector<std::pair<int64_t, int64_t>> oddEvenMergeStages(int64_t n) {
  SmallVector<std::pair<int64_t, int64_t>> stages;
  for (int64_t p = 1; p < n; p *= 2)
    for (int64_t k = p; k >= 1; k /= 2)
      stages.push_back({p, k});
  return stages;
}

struct SortOpLowering : OpRewritePattern<SortOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SortOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();
    int64_t dim = op.getDim();
    int64_t extent = resultType.getDimSize(dim);

    // Batcher's network rather than a bubble sort: the extent is a compile-
    // time constant, so the comparator schedule is too, and it costs
    // O(n log^2 n) compare-exchanges instead of O(n^2). Each stage is one
    // parallel sweep of the whole tensor -- every line advances through the
    // same stage at once, which is the line-level parallelism the sequential
    // form could not express. Only the stages are ordered; the elements
    // within one are independent.
    //
    // The network is written in its per-element form: an output element
    // either heads a comparator (taking the minimum with its partner `k`
    // above), tails one (taking the maximum with its partner `k` below), or
    // is untouched this stage. `isLow` decides which, from the element index
    // alone, so the body stays straight-line and the sweep stays parallel.
    Value sorted = op.getInput();
    for (auto [p, k] : oddEvenMergeStages(extent)) {
      Value input = sorted;
      sorted = buildElementwiseGeneric(
          rewriter, loc, resultType, ValueRange{},
          [&](OpBuilder &b, Location nested, ValueRange) -> Value {
            Value position = linalg::IndexOp::create(b, nested, dim);
            Value line = rank == 1
                             ? Value()
                             : linalg::IndexOp::create(b, nested, 1 - dim);
            auto at = [&](Value index) {
              SmallVector<Value> indices(rank);
              indices[dim] = index;
              if (rank == 2)
                indices[1 - dim] = line;
              return tensor::ExtractOp::create(b, nested, input, indices)
                  .getResult();
            };

            // Whether element `x` is the low side of a comparator in this
            // stage: it sits in the first half of its 2k block (offset by
            // `k mod p`), its partner is in range, and the pair does not
            // straddle a 2p merge boundary.
            auto isLow = [&](Value x) {
              Value kv = arith::ConstantIndexOp::create(b, nested, k);
              Value twoK = arith::ConstantIndexOp::create(b, nested, 2 * k);
              Value twoP = arith::ConstantIndexOp::create(b, nested, 2 * p);
              Value lo = arith::ConstantIndexOp::create(b, nested, k % p);
              Value hi = arith::ConstantIndexOp::create(b, nested, k % p + k);
              Value last = arith::ConstantIndexOp::create(b, nested, extent);
              Value slot = arith::RemUIOp::create(b, nested, x, twoK);
              Value inFirstHalf = arith::AndIOp::create(
                  b, nested,
                  arith::CmpIOp::create(b, nested, arith::CmpIPredicate::uge,
                                        slot, lo),
                  arith::CmpIOp::create(b, nested, arith::CmpIPredicate::ult,
                                        slot, hi));
              Value partner = arith::AddIOp::create(b, nested, x, kv);
              Value partnerInRange = arith::CmpIOp::create(
                  b, nested, arith::CmpIPredicate::ult, partner, last);
              Value sameBlock = arith::CmpIOp::create(
                  b, nested, arith::CmpIPredicate::eq,
                  arith::DivUIOp::create(b, nested, x, twoP),
                  arith::DivUIOp::create(b, nested, partner, twoP));
              return arith::AndIOp::create(
                  b, nested,
                  arith::AndIOp::create(b, nested, inFirstHalf, partnerInRange),
                  sameBlock);
            };

            Value kv = arith::ConstantIndexOp::create(b, nested, k);
            Value zero = arith::ConstantIndexOp::create(b, nested, 0);
            Value self = at(position);

            // Low side: min with the partner above.
            Value above = arith::AddIOp::create(b, nested, position, kv);
            Value aboveClamped = arith::MinUIOp::create(
                b, nested, above,
                arith::ConstantIndexOp::create(b, nested, extent - 1));
            Value lowMin =
                orderedFloatPair(b, nested, self, at(aboveClamped)).first;

            // High side: max with the partner below, when that partner is
            // itself the low side of this stage's comparator.
            Value belowRaw = arith::SubIOp::create(b, nested, position, kv);
            Value hasBelow = arith::CmpIOp::create(
                b, nested, arith::CmpIPredicate::uge, position, kv);
            Value below =
                arith::SelectOp::create(b, nested, hasBelow, belowRaw, zero);
            Value highMax = orderedFloatPair(b, nested, at(below), self).second;

            Value takesLow = isLow(position);
            Value takesHigh =
                arith::AndIOp::create(b, nested, hasBelow, isLow(below));
            Value result =
                arith::SelectOp::create(b, nested, takesHigh, highMax, self);
            return arith::SelectOp::create(b, nested, takesLow, lowMin, result);
          });
    }
    rewriter.replaceOp(op, sorted);
    return success();
  }
};

/// `sar.iterate` becomes `scf.for` over tensors: the carried values map
/// onto `iter_args`, which one-shot bufferization already understands on
/// both pipelines. The optional index argument materializes from the
/// loop counter as a one-element tensor.
struct IterateOpLowering : OpRewritePattern<IterateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IterateOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value ub = arith::ConstantIndexOp::create(
        rewriter, loc, static_cast<int64_t>(op.getTrips()));
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto forOp = scf::ForOp::create(rewriter, loc, lb, ub, step, op.getInits());

    Block *body = forOp.getBody();
    Block &src = op.getBody().front();
    auto yield = cast<sar::YieldOp>(src.getTerminator());

    SmallVector<Value> replacements;
    if (op.getIndex()) {
      // tensor.from_elements would be shorter but bufferizes to its own
      // allocation each iteration; insert-into-empty folds to a store
      // into one stack slot on both pipelines.
      rewriter.setInsertionPointToStart(body);
      Value counter = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI64Type(), forOp.getInductionVar());
      Value empty = tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{1},
                                            rewriter.getI64Type());
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      replacements.push_back(tensor::InsertOp::create(rewriter, loc, counter,
                                                      empty, ValueRange{zero}));
    }
    llvm::append_range(replacements, forOp.getRegionIterArgs());

    rewriter.inlineBlockBefore(&src, body, body->end(), replacements);
    rewriter.setInsertionPointToEnd(body);
    scf::YieldOp::create(rewriter, yield.getLoc(), yield.getResults());
    rewriter.eraseOp(yield);
    rewriter.replaceOp(op, forOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertSARToLinalgPass
    : sar::impl::ConvertSARToLinalgBase<ConvertSARToLinalgPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addIllegalDialect<SARDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    sar::populateSARToLinalgConversionPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::sar::populateSARToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ConstantOpLowering, BinaryOpLowering<AddOp, BinaryKind::Add>,
               BinaryOpLowering<SubOp, BinaryKind::Sub>,
               BinaryOpLowering<MulOp, BinaryKind::Mul>,
               BinaryOpLowering<DivOp, BinaryKind::Div>,
               ScalarBinaryOpLowering<AddScalarOp, BinaryKind::Add>,
               ScalarBinaryOpLowering<MulScalarOp, BinaryKind::Mul>,
               SqrtOpLowering, CosOpLowering, SinOpLowering, ExpOpLowering,
               LogOpLowering, Atan2OpLowering, AbsOpLowering, CastOpLowering,
               CmpOpLowering, WhereOpLowering, ComplexOpLowering,
               ConjOpLowering, ComplexPartOpLowering<RealOp, complex::ReOp>,
               ComplexPartOpLowering<ImagOp, complex::ImOp>, ReduceOpLowering,
               ArgMaxOpLowering, SliceOpLowering, DynamicSliceOpLowering,
               DynamicUpdateSliceOpLowering, ConcatOpLowering, PadOpLowering,
               ReverseOpLowering, TransposeOpLowering, BroadcastOpLowering,
               FFTShiftOpLowering, CumsumOpLowering, RankFilterOpLowering,
               SortOpLowering, Gather2DOpLowering, Gather2DSplitOpLowering,
               IterateOpLowering>(patterns.getContext());
}
