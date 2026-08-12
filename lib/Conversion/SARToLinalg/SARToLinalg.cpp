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
        rewriter, op.getLoc(), resultType,
        ValueRange{op.getLhs(), op.getRhs()},
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

struct MaxScalarOpLowering : OpRewritePattern<MaxScalarOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MaxScalarOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    double scalar = op.getScalar().convertToDouble();
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getInput()},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value cst =
              buildScalarConstant(b, loc, resultType.getElementType(), scalar);
          // cmpf+select instead of arith.maxnumf: the emitted IR must also
          // parse under the older LLVM pinned by the ScaleHLS toolchain.
          Value greater = b.create<arith::CmpFOp>(
              loc, arith::CmpFPredicate::OGT, args[0], cst);
          return b.create<arith::SelectOp>(loc, greater, args[0], cst)
              .getResult();
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NegOpLowering : OpRewritePattern<NegOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getInput()},
        [&](OpBuilder &b, Location loc, ValueRange args) -> Value {
          if (isa<ComplexType>(args[0].getType()))
            return b.create<complex::NegOp>(loc, args[0]);
          return b.create<arith::NegFOp>(loc, args[0]);
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

struct ExpJOpLowering : OpRewritePattern<ExpJOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ExpJOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto complexTy = cast<ComplexType>(resultType.getElementType());
    Value result = buildElementwiseGeneric(
        rewriter, op.getLoc(), resultType, ValueRange{op.getInput()},
        [&](OpBuilder &b, Location loc, ValueRange args) -> Value {
          Value cosVal = b.create<math::CosOp>(loc, args[0]);
          Value sinVal = b.create<math::SinOp>(loc, args[0]);
          return b.create<complex::CreateOp>(loc, complexTy, cosVal, sinVal);
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
              re = convertFloat(b, loc, in, outFloat);
              im = b.create<arith::ConstantOp>(loc,
                                               b.getFloatAttr(outFloat, 0.0));
            }
            return b.create<complex::CreateOp>(loc, outComplex, re, im);
          }
          return convertFloat(b, loc, in, cast<FloatType>(outElem));
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct TransposeOpLowering : OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
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
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
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
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
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
              Value offsetCst = b.create<arith::ConstantIndexOp>(nestedLoc,
                                                                 offset);
              Value sizeCst = b.create<arith::ConstantIndexOp>(nestedLoc,
                                                               size);
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
  patterns.add<ConstantOpLowering,
               BinaryOpLowering<AddOp, BinaryKind::Add>,
               BinaryOpLowering<SubOp, BinaryKind::Sub>,
               BinaryOpLowering<MulOp, BinaryKind::Mul>,
               BinaryOpLowering<DivOp, BinaryKind::Div>,
               ScalarBinaryOpLowering<AddScalarOp, BinaryKind::Add>,
               ScalarBinaryOpLowering<MulScalarOp, BinaryKind::Mul>,
               MaxScalarOpLowering, NegOpLowering, SqrtOpLowering,
               CosOpLowering, SinOpLowering,
               AbsOpLowering, ExpJOpLowering, CastOpLowering,
               TransposeOpLowering, BroadcastOpLowering, FFTShiftOpLowering>(
      patterns.getContext());
}
