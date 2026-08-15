//===- SARToLinalg.cpp - Lower SAR ops to linalg --------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
// Lowers SAR structural and element-wise operations to linalg-on-tensors.
// Complex element types are preserved (they lower later through the complex
// dialect); signal-processing ops are handled by ConvertSARSignalToRuntime.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include "sar/Conversion/Passes.h"
#include "sar/Dialect/SAR/IR/SARDialect.h"
#include "sar/Dialect/SAR/IR/SAROps.h"

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
  Value init = builder.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                               resultType.getElementType());
  SmallVector<AffineMap> maps(inputs.size() + 1,
                              builder.getMultiDimIdentityMap(rank));
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);
  auto generic = builder.create<linalg::GenericOp>(
      loc, TypeRange{resultType}, inputs, ValueRange{init}, maps, iterators,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        // Drop the output block argument.
        Value result =
            bodyBuilder(b, nestedLoc, args.take_front(inputs.size()));
        b.create<linalg::YieldOp>(nestedLoc, result);
      });
  return generic.getResult(0);
}

/// Creates a scalar constant of the given float-or-complex element type from
/// a double value (imaginary part zero for complex types).
static Value buildScalarConstant(OpBuilder &b, Location loc, Type elementType,
                                 double value) {
  if (auto complexTy = dyn_cast<ComplexType>(elementType)) {
    auto floatTy = cast<FloatType>(complexTy.getElementType());
    auto re = b.getFloatAttr(floatTy, value);
    auto im = b.getFloatAttr(floatTy, 0.0);
    return b.create<complex::ConstantOp>(loc, complexTy,
                                         b.getArrayAttr({re, im}));
  }
  if (auto intTy = dyn_cast<IntegerType>(elementType))
    return b.create<arith::ConstantOp>(loc,
                                       b.getIntegerAttr(intTy, (int64_t)value));
  auto floatTy = cast<FloatType>(elementType);
  return b.create<arith::ConstantOp>(loc, b.getFloatAttr(floatTy, value));
}

enum class BinaryKind { Add, Sub, Mul, Div };

static Value buildBinaryScalarOp(OpBuilder &b, Location loc, BinaryKind kind,
                                 Value lhs, Value rhs) {
  Type type = lhs.getType();
  if (isa<FloatType>(type)) {
    switch (kind) {
    case BinaryKind::Add:
      return b.create<arith::AddFOp>(loc, lhs, rhs);
    case BinaryKind::Sub:
      return b.create<arith::SubFOp>(loc, lhs, rhs);
    case BinaryKind::Mul:
      return b.create<arith::MulFOp>(loc, lhs, rhs);
    case BinaryKind::Div:
      return b.create<arith::DivFOp>(loc, lhs, rhs);
    }
  }
  if (isa<IntegerType>(type)) {
    switch (kind) {
    case BinaryKind::Add:
      return b.create<arith::AddIOp>(loc, lhs, rhs);
    case BinaryKind::Sub:
      return b.create<arith::SubIOp>(loc, lhs, rhs);
    case BinaryKind::Mul:
      return b.create<arith::MulIOp>(loc, lhs, rhs);
    case BinaryKind::Div:
      return b.create<arith::DivSIOp>(loc, lhs, rhs);
    }
  }
  assert(isa<ComplexType>(type) && "expected float, int or complex");
  switch (kind) {
  case BinaryKind::Add:
    return b.create<complex::AddOp>(loc, lhs, rhs);
  case BinaryKind::Sub:
    return b.create<complex::SubOp>(loc, lhs, rhs);
  case BinaryKind::Mul:
    return b.create<complex::MulOp>(loc, lhs, rhs);
  case BinaryKind::Div:
    return b.create<complex::DivOp>(loc, lhs, rhs);
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
    return b.create<arith::ExtFOp>(loc, target, value);
  return b.create<arith::TruncFOp>(loc, target, value);
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
               ? b.create<arith::ExtSIOp>(loc, target, value).getResult()
               : b.create<arith::TruncIOp>(loc, target, value).getResult();
  if (sourceInt)
    return b.create<arith::SIToFPOp>(loc, target, value);
  if (targetInt)
    return b.create<arith::FPToSIOp>(loc, target, value);
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
          return b.create<MathOpTy>(loc, args[0]);
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
            return b.create<complex::AbsOp>(loc, args[0]);
          return b.create<math::AbsFOp>(loc, args[0]);
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
                         .Case("ne", arith::CmpFPredicate::ONE)
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
              b.create<arith::CmpFOp>(loc, *predicate, args[0], args[1]);
          // The mask is a float so it can flow through the same ops as the
          // data it selects between.
          Value one =
              buildScalarConstant(b, loc, resultType.getElementType(), 1.0);
          Value zero =
              buildScalarConstant(b, loc, resultType.getElementType(), 0.0);
          return b.create<arith::SelectOp>(loc, held, one, zero).getResult();
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
          Value held = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                               args[0], zero);
          return b.create<arith::SelectOp>(loc, held, args[1], args[2])
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
          return b.create<complex::CreateOp>(loc, complexTy, args[0], args[1])
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
          return b.create<complex::ConjOp>(loc, args[0]).getResult();
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
          return b.template create<ComplexOpTy>(loc, args[0]).getResult();
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
          return b.create<math::Atan2Op>(loc, args[0], args[1]).getResult();
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
              re = convertFloat(b, loc, b.create<complex::ReOp>(loc, in),
                                outFloat);
              im = convertFloat(b, loc, b.create<complex::ImOp>(loc, in),
                                outFloat);
            } else {
              re = convertReal(b, loc, in, outFloat);
              im = b.create<arith::ConstantOp>(loc,
                                               b.getFloatAttr(outFloat, 0.0));
            }
            return b.create<complex::CreateOp>(loc, outComplex, re, im);
          }
          if (auto inComplex = dyn_cast<ComplexType>(inElem))
            in = b.create<complex::ReOp>(loc, in);
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

    // The identity has to be the one the combinator leaves untouched, or
    // the first element folded in would be lost.
    double identity = 0.0;
    if (kind == "max")
      identity = -std::numeric_limits<double>::infinity();
    else if (kind == "min")
      identity = std::numeric_limits<double>::infinity();

    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  elementType);
    Value seed = buildScalarConstant(rewriter, loc, elementType, identity);
    Value filled =
        rewriter.create<linalg::FillOp>(loc, ValueRange{seed}, ValueRange{init})
            .getResult(0);

    auto combine = [&](OpBuilder &b, Location nested, Value lhs,
                       Value rhs) -> Value {
      if (kind == "sum") {
        if (isa<ComplexType>(elementType))
          return b.create<complex::AddOp>(nested, lhs, rhs);
        return b.create<arith::AddFOp>(nested, lhs, rhs);
      }
      auto predicate =
          kind == "max" ? arith::CmpFPredicate::OGT : arith::CmpFPredicate::OLT;
      Value wins = b.create<arith::CmpFOp>(nested, predicate, lhs, rhs);
      return b.create<arith::SelectOp>(nested, wins, lhs, rhs);
    };

    auto reduce = rewriter.create<linalg::ReduceOp>(
        loc, ValueRange{op.getInput()}, ValueRange{filled},
        ArrayRef<int64_t>{op.getDim()},
        [&](OpBuilder &b, Location nested, ValueRange args) {
          b.create<linalg::YieldOp>(nested,
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
    Value valueInit = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elementType);
    Value lowest = buildScalarConstant(
        rewriter, loc, elementType, -std::numeric_limits<double>::infinity());
    Value values = rewriter
                       .create<linalg::FillOp>(loc, ValueRange{lowest},
                                               ValueRange{valueInit})
                       .getResult(0);
    Value indexInit = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(resultType.getElementType(), 0));
    Value indices = rewriter
                        .create<linalg::FillOp>(loc, ValueRange{zero},
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

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{values.getType(), indices.getType()},
        ValueRange{op.getInput()}, ValueRange{values, indices},
        ArrayRef<AffineMap>{inputMap, resultMap, resultMap}, iterators,
        [&](OpBuilder &b, Location nested, ValueRange args) {
          Value candidate = args[0], best = args[1], bestIndex = args[2];
          Value position = b.create<linalg::IndexOp>(nested, dim);
          position = b.create<arith::IndexCastOp>(
              nested, resultType.getElementType(), position);
          Value better = b.create<arith::CmpFOp>(
              nested, arith::CmpFPredicate::OGT, candidate, best);
          Value newBest =
              b.create<arith::SelectOp>(nested, better, candidate, best);
          Value newIndex =
              b.create<arith::SelectOp>(nested, better, position, bestIndex);
          b.create<linalg::YieldOp>(nested, ValueRange{newBest, newIndex});
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

    // A forward slice becomes a sweep whose source index is an affine map
    // of the loop indices. `tensor.extract_slice` would be shorter, but it
    // bufferizes to a view, and a view that survives is a value the
    // dataflow backend cannot place in a task -- and elsewhere a layout
    // every later pass has to carry along.
    if (llvm::all_of(sliceStrides, [](int64_t s) { return s > 0; })) {
      SmallVector<utils::IteratorType> iterators(rank,
                                                 utils::IteratorType::parallel);

      // The source is read through `tensor.extract` rather than an indexing
      // map: a map would make linalg infer the operand's shape from the
      // iteration domain, which is the result's, not the source's.
      Location loc = op.getLoc();
      Value init = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), resultType.getElementType());
      auto generic = rewriter.create<linalg::GenericOp>(
          loc, TypeRange{resultType}, ValueRange{}, ValueRange{init},
          ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(rank)}, iterators,
          [&](OpBuilder &b, Location nested, ValueRange) {
            SmallVector<Value> indices;
            for (int64_t d = 0; d < rank; ++d) {
              Value index = b.create<linalg::IndexOp>(nested, d);
              if (sliceStrides[d] != 1)
                index = b.create<arith::MulIOp>(
                    nested, index,
                    b.create<arith::ConstantIndexOp>(nested, sliceStrides[d]));
              if (sliceOffsets[d] != 0)
                index = b.create<arith::AddIOp>(
                    nested, index,
                    b.create<arith::ConstantIndexOp>(nested, sliceOffsets[d]));
              indices.push_back(index);
            }
            b.create<linalg::YieldOp>(
                nested,
                b.create<tensor::ExtractOp>(nested, op.getInput(), indices)
                    .getResult());
          });
      rewriter.replaceOp(op, generic.getResults());
      return success();
    }

    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (int64_t v : op.getOffsets())
      offsets.push_back(rewriter.getIndexAttr(v));
    for (int64_t v : op.getSizes())
      sizes.push_back(rewriter.getIndexAttr(v));
    for (int64_t v : op.getStrides())
      strides.push_back(rewriter.getIndexAttr(v));
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, cast<RankedTensorType>(op.getType()), op.getInput(), offsets, sizes,
        strides);
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
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{}, ValueRange{init},
        ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(rank)}, iterators,
        [&](OpBuilder &b, Location nested, ValueRange) {
          SmallVector<Value> lhsIndices, rhsIndices;
          Value position;
          for (int64_t d = 0; d < rank; ++d) {
            Value index = b.create<linalg::IndexOp>(nested, d);
            if (d != dim) {
              lhsIndices.push_back(index);
              rhsIndices.push_back(index);
              continue;
            }
            position = index;
            Value splitCst = b.create<arith::ConstantIndexOp>(nested, split);
            // Both reads are clamped into their own operand, so the one
            // that is not selected still addresses a valid element.
            Value lhsMax = b.create<arith::ConstantIndexOp>(nested, split - 1);
            Value rhsMax = b.create<arith::ConstantIndexOp>(
                nested, rhsType.getDimSize(dim) - 1);
            Value zero = b.create<arith::ConstantIndexOp>(nested, 0);
            lhsIndices.push_back(
                b.create<arith::MinSIOp>(nested, index, lhsMax));
            Value shifted = b.create<arith::SubIOp>(nested, index, splitCst);
            shifted = b.create<arith::MaxSIOp>(nested, shifted, zero);
            rhsIndices.push_back(
                b.create<arith::MinSIOp>(nested, shifted, rhsMax));
          }
          Value splitCst = b.create<arith::ConstantIndexOp>(nested, split);
          // Indices are non-negative, so the unsigned comparison is the
          // one that says what is meant.
          Value fromLhs = b.create<arith::CmpIOp>(
              nested, arith::CmpIPredicate::ult, position, splitCst);
          Value left =
              b.create<tensor::ExtractOp>(nested, op.getLhs(), lhsIndices);
          Value right =
              b.create<tensor::ExtractOp>(nested, op.getRhs(), rhsIndices);
          b.create<linalg::YieldOp>(
              nested, b.create<arith::SelectOp>(nested, fromLhs, left, right)
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
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{}, ValueRange{init},
        ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(rank)}, iterators,
        [&](OpBuilder &b, Location nested, ValueRange) {
          Value inside;
          SmallVector<Value> indices;
          for (int64_t d = 0; d < rank; ++d) {
            Value index = b.create<linalg::IndexOp>(nested, d);
            Value shifted = b.create<arith::SubIOp>(
                nested, index,
                b.create<arith::ConstantIndexOp>(nested, low[d]));
            // Clamp so the read addresses a valid element even where the
            // result takes the padding value instead.
            Value zero = b.create<arith::ConstantIndexOp>(nested, 0);
            Value last = b.create<arith::ConstantIndexOp>(
                nested, inputType.getDimSize(d) - 1);
            Value clamped = b.create<arith::MinSIOp>(
                nested, b.create<arith::MaxSIOp>(nested, shifted, zero), last);
            indices.push_back(clamped);

            Value inRange = b.create<arith::AndIOp>(
                nested,
                b.create<arith::CmpIOp>(nested, arith::CmpIPredicate::sge,
                                        shifted, zero),
                b.create<arith::CmpIOp>(nested, arith::CmpIPredicate::sle,
                                        shifted, last));
            inside = inside ? b.create<arith::AndIOp>(nested, inside, inRange)
                                  .getResult()
                            : inRange;
          }
          Value element =
              b.create<tensor::ExtractOp>(nested, op.getInput(), indices);
          Value fill =
              buildScalarConstant(b, nested, resultType.getElementType(),
                                  op.getValue().convertToDouble());
          b.create<linalg::YieldOp>(
              nested, b.create<arith::SelectOp>(nested, inside, element, fill)
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
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{op.getInput()}, ValueRange{init},
        ArrayRef<AffineMap>{sourceMap, identity}, iterators,
        [&](OpBuilder &b, Location nested, ValueRange args) {
          b.create<linalg::YieldOp>(nested, args.front());
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
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto transpose = rewriter.create<linalg::TransposeOp>(
        loc, op.getInput(), init, ArrayRef<int64_t>{1, 0});
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
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    // The vector lies along axis `dim`; the other axis is the broadcasted
    // (added) dimension.
    int64_t addedDim = 1 - op.getDim();
    auto broadcast = rewriter.create<linalg::BroadcastOp>(
        loc, op.getInput(), init, ArrayRef<int64_t>{addedDim});
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
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    SmallVector<AffineMap> maps{rewriter.getMultiDimIdentityMap(rank)};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{}, ValueRange{init}, maps,
        iterators, [&](OpBuilder &b, Location nestedLoc, ValueRange) {
          SmallVector<Value> indices;
          for (int64_t d = 0; d < rank; ++d) {
            Value idx = b.create<linalg::IndexOp>(nestedLoc, d);
            if (d == dim) {
              Value offsetCst =
                  b.create<arith::ConstantIndexOp>(nestedLoc, offset);
              Value sizeCst = b.create<arith::ConstantIndexOp>(nestedLoc, size);
              Value sum = b.create<arith::AddIOp>(nestedLoc, idx, offsetCst);
              idx = b.create<arith::RemUIOp>(nestedLoc, sum, sizeCst);
            }
            indices.push_back(idx);
          }
          Value element =
              b.create<tensor::ExtractOp>(nestedLoc, input, indices);
          b.create<linalg::YieldOp>(nestedLoc, element);
        });
    rewriter.replaceOp(op, generic.getResults());
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
               ArgMaxOpLowering, SliceOpLowering, ConcatOpLowering,
               PadOpLowering, ReverseOpLowering, TransposeOpLowering,
               BroadcastOpLowering, FFTShiftOpLowering>(patterns.getContext());
}
