//===- SARDialect.cpp - SAR dialect and op implementations ----------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "sar/Dialect/SAR/IR/SARDialect.h"
#include "sar/Dialect/SAR/IR/SAROps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cmath>

using namespace mlir;
using namespace mlir::sar;

#include "sar/Dialect/SAR/IR/SARDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void SARDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "sar/Dialect/SAR/IR/SAROps.cpp.inc"
      >();
}

Operation *SARDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  auto elements = dyn_cast<ElementsAttr>(value);
  if (!elements || elements.getType() != type)
    return nullptr;
  return ConstantOp::create(builder, loc, type, elements);
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static RankedTensorType getRanked(Type type) {
  return cast<RankedTensorType>(type);
}

static LogicalResult verifyPositiveShape(Operation *op, RankedTensorType type,
                                         StringRef role) {
  if (llvm::any_of(type.getShape(), [](int64_t size) { return size <= 0; }))
    return op->emitOpError() << role << " must have static positive extents";
  return success();
}

/// Returns the float precision backing an element type (f32/f64 or the
/// element of complex<f32>/complex<f64>).
static FloatType getFloatPrecision(Type elementType) {
  if (auto complexTy = dyn_cast<ComplexType>(elementType))
    return cast<FloatType>(complexTy.getElementType());
  return cast<FloatType>(elementType);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValueAttr(); }

//===----------------------------------------------------------------------===//
// Folds
//
// Only bit-exact rewrites are performed: floating-point identities that can
// change results (e.g. x + 0.0 with x == -0.0, or fft(ifft(x)) round-trips)
// are deliberately not folded.
//===----------------------------------------------------------------------===//

/// A permutation fold cap: constants at or below this many elements are
/// rearranged at compile time (window and axis tables), larger ones (whole
/// raster planes) keep the runtime permutation rather than double a huge
/// constant pool.
constexpr int64_t kMaxFoldedPermutationElements = 1 << 16;

/// Rebuilds `dense` with each element moved by `makeMapper`'s index map:
/// the shared engine of the bit-exact permutation folds (fftshift,
/// reverse), which move data without touching a bit of it.
template <typename MakeMapper>
static OpFoldResult foldPermutedConstant(DenseElementsAttr dense,
                                         MakeMapper makeMapper) {
  auto type = cast<RankedTensorType>(dense.getType());
  if (type.getNumElements() > kMaxFoldedPermutationElements)
    return {};
  if (dense.isSplat()) // any permutation of a splat is the splat
    return dense;
  auto mapper = makeMapper(type.getShape());

  SmallVector<Attribute> elements(dense.getValues<Attribute>().begin(),
                                  dense.getValues<Attribute>().end());
  SmallVector<Attribute> permuted(elements.size());
  ArrayRef<int64_t> shape = type.getShape();
  SmallVector<int64_t> index(shape.size(), 0);
  for (int64_t linear = 0, e = elements.size(); linear < e; ++linear) {
    SmallVector<int64_t> src(index);
    mapper(src);
    int64_t srcLinear = 0;
    for (auto [i, extent] : llvm::enumerate(shape))
      srcLinear = srcLinear * extent + src[i];
    permuted[linear] = elements[srcLinear];
    for (int64_t d = shape.size() - 1; d >= 0; --d) {
      if (++index[d] < shape[d])
        break;
      index[d] = 0;
    }
  }
  return DenseElementsAttr::get(type, permuted);
}

/// transpose(transpose(x)) == x (a pure permutation, bit-exact).
OpFoldResult TransposeOp::fold(FoldAdaptor) {
  if (auto inner = getInput().getDefiningOp<TransposeOp>())
    return inner.getInput();
  return {};
}

/// fftshift and ifftshift along the same axis are exact inverses for any
/// size (forward rotates by ceil(n/2), inverse by floor(n/2)). A shift of
/// a small compile-time constant folds into a rotated constant: pure data
/// movement, bit-exact.
OpFoldResult FFTShiftOp::fold(FoldAdaptor adaptor) {
  auto inner = getInput().getDefiningOp<FFTShiftOp>();
  if (inner && inner.getDim() == getDim() && inner.getInverse() != getInverse())
    return inner.getInput();
  if (auto dense = dyn_cast_or_null<DenseElementsAttr>(adaptor.getInput()))
    return foldPermutedConstant(dense, [&](ArrayRef<int64_t> shape) {
      // Same index map as the lowering: fftshift reads ceil(n/2) ahead,
      // ifftshift floor(n/2) (they differ for odd sizes).
      int64_t dim = getDim();
      int64_t n = shape[dim];
      int64_t split = getInverse() ? n / 2 : n - n / 2;
      return [dim, n, split](SmallVectorImpl<int64_t> &index) {
        index[dim] = (index[dim] + split) % n;
      };
    });
  return {};
}

/// x * 1.0 == x bit-exactly (preserves -0.0, NaN payloads and infinities).
OpFoldResult MulScalarOp::fold(FoldAdaptor) {
  if (getScalar().isExactlyValue(1.0))
    return getInput();
  return {};
}

/// conj(conj(x)) == x (bit-exact sign flip round-trip).
OpFoldResult ConjOp::fold(FoldAdaptor) {
  if (auto inner = getInput().getDefiningOp<ConjOp>())
    return inner.getInput();
  return {};
}

/// real(complex(re, im)) == re (plane extraction of a plane pair).
OpFoldResult RealOp::fold(FoldAdaptor) {
  if (auto create = getInput().getDefiningOp<ComplexOp>())
    return create.getRe();
  return {};
}

/// imag(complex(re, im)) == im.
OpFoldResult ImagOp::fold(FoldAdaptor) {
  if (auto create = getInput().getDefiningOp<ComplexOp>())
    return create.getIm();
  return {};
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult AbsOp::verify() {
  auto inputTy = getRanked(getInput().getType());
  auto resultTy = getRanked(getResult().getType());
  if (inputTy.getShape() != resultTy.getShape())
    return emitOpError("input and result shapes must match");
  if (getFloatPrecision(inputTy.getElementType()) != resultTy.getElementType())
    return emitOpError("result element type must be the float precision of "
                       "the input element type");
  return success();
}

//===----------------------------------------------------------------------===//
// CmpOp / WhereOp
//===----------------------------------------------------------------------===//

LogicalResult CmpOp::verify() {
  StringRef predicate = getPredicate();
  static const char *kinds[] = {"eq", "ne", "lt", "le", "gt", "ge"};
  if (!llvm::is_contained(kinds, predicate))
    return emitOpError("predicate must be one of eq, ne, lt, le, gt, ge");
  return success();
}

LogicalResult WhereOp::verify() {
  auto maskTy = getRanked(getMask().getType());
  auto resultTy = getRanked(getResult().getType());
  if (maskTy.getShape() != resultTy.getShape())
    return emitOpError("mask and branch shapes must match");
  if (getFloatPrecision(resultTy.getElementType()) != maskTy.getElementType())
    return emitOpError("mask precision must match the branch element "
                       "precision");
  return success();
}

//===----------------------------------------------------------------------===//
// RealOp / ImagOp / ComplexOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyComplexToFloat(Operation *op,
                                          RankedTensorType inputTy,
                                          RankedTensorType resultTy) {
  if (inputTy.getShape() != resultTy.getShape())
    return op->emitOpError("input and result shapes must match");
  if (getFloatPrecision(inputTy.getElementType()) != resultTy.getElementType())
    return op->emitOpError("result element type must be the float precision "
                           "of the input element type");
  return success();
}

LogicalResult RealOp::verify() {
  return verifyComplexToFloat(*this, getRanked(getInput().getType()),
                              getRanked(getResult().getType()));
}

LogicalResult ImagOp::verify() {
  return verifyComplexToFloat(*this, getRanked(getInput().getType()),
                              getRanked(getResult().getType()));
}

LogicalResult ComplexOp::verify() {
  auto planeTy = getRanked(getRe().getType());
  auto resultTy = getRanked(getResult().getType());
  if (planeTy.getShape() != resultTy.getShape())
    return emitOpError("plane and result shapes must match");
  auto complexTy = cast<ComplexType>(resultTy.getElementType());
  if (complexTy.getElementType() != planeTy.getElementType())
    return emitOpError("result complex precision must match the plane float "
                       "type");
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  auto inputTy = getRanked(getInput().getType());
  auto resultTy = getRanked(getResult().getType());
  if (inputTy.getShape() != resultTy.getShape())
    return emitOpError("input and result shapes must match");

  Type in = inputTy.getElementType();
  Type out = resultTy.getElementType();
  bool inComplex = isa<ComplexType>(in);
  bool outComplex = isa<ComplexType>(out);
  // complex -> float/int is not a cast (use sar.abs or sar.real).
  if (inComplex && !outComplex)
    return emitOpError("cannot cast a complex tensor to a real tensor");
  // int -> complex goes through an explicit int -> float cast first.
  if (isa<IntegerType>(in) && outComplex)
    return emitOpError("cannot cast an integer tensor directly to "
                       "complex; cast to float first");
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceOp / ArgMaxOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyReduceShapes(Operation *op, RankedTensorType inputTy,
                                        RankedTensorType resultTy,
                                        int64_t dim) {
  if (failed(verifyPositiveShape(op, inputTy, "input")) ||
      failed(verifyPositiveShape(op, resultTy, "result")))
    return failure();
  if (inputTy.getRank() != 2)
    return op->emitOpError("expects a rank-2 input");
  if (dim < 0 || dim > 1)
    return op->emitOpError("dim must be 0 or 1");
  if (resultTy.getRank() != 1 ||
      resultTy.getDimSize(0) != inputTy.getDimSize(1 - dim))
    return op->emitOpError(
        "result must be rank-1 with the size of the kept axis");
  return success();
}

LogicalResult ReduceOp::verify() {
  auto inputTy = getRanked(getInput().getType());
  auto resultTy = getRanked(getResult().getType());
  if (failed(verifyReduceShapes(*this, inputTy, resultTy, getDim())))
    return failure();
  if (inputTy.getElementType() != resultTy.getElementType())
    return emitOpError("input and result element types must match");
  StringRef kind = getKind();
  if (kind != "sum" && kind != "max" && kind != "min")
    return emitOpError("kind must be one of sum, max, min");
  if (kind != "sum" && isa<ComplexType>(inputTy.getElementType()))
    return emitOpError("max/min reductions require float elements");
  return success();
}

LogicalResult ArgMaxOp::verify() {
  return verifyReduceShapes(*this, getRanked(getInput().getType()),
                            getRanked(getResult().getType()), getDim());
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  auto inputTy = getRanked(getInput().getType());
  auto resultTy = getRanked(getResult().getType());
  if (failed(verifyPositiveShape(*this, inputTy, "input")) ||
      failed(verifyPositiveShape(*this, resultTy, "result")))
    return failure();
  if (inputTy.getRank() != 2)
    return emitOpError("expects a rank-2 input");
  if (inputTy.getElementType() != resultTy.getElementType())
    return emitOpError("input and result element types must match");
  if (resultTy.getRank() != 2 ||
      resultTy.getDimSize(0) != inputTy.getDimSize(1) ||
      resultTy.getDimSize(1) != inputTy.getDimSize(0))
    return emitOpError("result shape must be the transpose of the input "
                       "shape");
  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::verify() {
  auto inputTy = getRanked(getInput().getType());
  auto resultTy = getRanked(getResult().getType());
  if (failed(verifyPositiveShape(*this, inputTy, "input")) ||
      failed(verifyPositiveShape(*this, resultTy, "result")))
    return failure();
  if (inputTy.getRank() != 1)
    return emitOpError("expects a rank-1 input");
  if (resultTy.getRank() != 2)
    return emitOpError("expects a rank-2 result");
  if (inputTy.getElementType() != resultTy.getElementType())
    return emitOpError("input and result element types must match");
  int64_t dim = getDim();
  if (dim < 0 || dim > 1)
    return emitOpError("dim must be 0 or 1 for a rank-2 result");
  if (resultTy.getDimSize(dim) != inputTy.getDimSize(0))
    return emitOpError("result size along dim must equal the input length");
  return success();
}

//===----------------------------------------------------------------------===//
// SliceOp / ConcatOp / PadOp
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::verify() {
  auto inputTy = getRanked(getInput().getType());
  auto resultTy = getRanked(getResult().getType());
  if (failed(verifyPositiveShape(*this, inputTy, "input")) ||
      failed(verifyPositiveShape(*this, resultTy, "result")))
    return failure();
  int64_t rank = inputTy.getRank();
  if (inputTy.getElementType() != resultTy.getElementType())
    return emitOpError("input and result element types must match");
  if (static_cast<int64_t>(getOffsets().size()) != rank ||
      static_cast<int64_t>(getSizes().size()) != rank ||
      static_cast<int64_t>(getStrides().size()) != rank ||
      resultTy.getRank() != rank)
    return emitOpError("offsets, sizes, strides and result rank must "
                       "match the input rank");
  for (int64_t d = 0; d < rank; ++d) {
    int64_t offset = getOffsets()[d], size = getSizes()[d];
    int64_t stride = getStrides()[d];
    if (stride < 1)
      return emitOpError("strides must be positive");
    if (offset < 0 || size < 1 ||
        offset + (size - 1) * stride >= inputTy.getDimSize(d))
      return emitOpError("slice bounds exceed the input along dim ") << d;
    if (resultTy.getDimSize(d) != size)
      return emitOpError("result shape must equal sizes");
  }
  return success();
}

static LogicalResult verifyDynamicOffsets(Operation *op, ValueRange offsets,
                                          int64_t rank) {
  if (static_cast<int64_t>(offsets.size()) != rank)
    return op->emitOpError("requires one scalar offset tensor per input axis");
  for (auto [dim, offset] : llvm::enumerate(offsets)) {
    auto type = getRanked(offset.getType());
    if (!type || type.getRank() != 1 || type.getDimSize(0) != 1 ||
        !type.getElementType().isInteger(64))
      return op->emitOpError("offset #")
             << dim << " must have type tensor<1xi64>";
  }
  return success();
}

LogicalResult DynamicSliceOp::verify() {
  auto inputTy = getRanked(getInput().getType());
  auto resultTy = getRanked(getResult().getType());
  if (failed(verifyPositiveShape(*this, inputTy, "input")) ||
      failed(verifyPositiveShape(*this, resultTy, "result")))
    return failure();
  int64_t rank = inputTy.getRank();
  if (inputTy.getElementType() != resultTy.getElementType())
    return emitOpError("input and result element types must match");
  if (resultTy.getRank() != rank ||
      static_cast<int64_t>(getSizes().size()) != rank ||
      static_cast<int64_t>(getStrides().size()) != rank)
    return emitOpError("sizes, strides and result rank must match the input "
                       "rank");
  if (failed(verifyDynamicOffsets(*this, getOffsets(), rank)))
    return failure();
  for (int64_t dim = 0; dim < rank; ++dim) {
    int64_t size = getSizes()[dim];
    int64_t stride = getStrides()[dim];
    if (size < 1 || stride < 1)
      return emitOpError("sizes and strides must be positive");
    if (size > 1 + (inputTy.getDimSize(dim) - 1) / stride)
      return emitOpError("slice span exceeds the input along dim ") << dim;
    if (resultTy.getDimSize(dim) != size)
      return emitOpError("result shape must equal sizes");
  }
  return success();
}

LogicalResult DynamicUpdateSliceOp::verify() {
  auto inputTy = getRanked(getInput().getType());
  auto updateTy = getRanked(getUpdate().getType());
  auto resultTy = getRanked(getResult().getType());
  if (failed(verifyPositiveShape(*this, inputTy, "input")) ||
      failed(verifyPositiveShape(*this, updateTy, "update")) ||
      failed(verifyPositiveShape(*this, resultTy, "result")))
    return failure();
  int64_t rank = inputTy.getRank();
  if (updateTy.getRank() != rank || resultTy != inputTy)
    return emitOpError("update rank must match the input and result type must "
                       "equal the input type");
  if (updateTy.getElementType() != inputTy.getElementType())
    return emitOpError("input and update element types must match");
  if (failed(verifyDynamicOffsets(*this, getOffsets(), rank)))
    return failure();
  for (int64_t dim = 0; dim < rank; ++dim)
    if (updateTy.getDimSize(dim) > inputTy.getDimSize(dim))
      return emitOpError("update exceeds the input along dim ") << dim;
  return success();
}

LogicalResult ConcatOp::verify() {
  auto lhsTy = getRanked(getLhs().getType());
  auto rhsTy = getRanked(getRhs().getType());
  auto resultTy = getRanked(getResult().getType());
  if (failed(verifyPositiveShape(*this, lhsTy, "left operand")) ||
      failed(verifyPositiveShape(*this, rhsTy, "right operand")) ||
      failed(verifyPositiveShape(*this, resultTy, "result")))
    return failure();
  int64_t rank = lhsTy.getRank();
  int64_t dim = getDim();
  if (lhsTy.getElementType() != rhsTy.getElementType() ||
      lhsTy.getElementType() != resultTy.getElementType())
    return emitOpError("element types must match");
  if (rhsTy.getRank() != rank || resultTy.getRank() != rank)
    return emitOpError("operand and result ranks must match");
  if (dim < 0 || dim >= rank)
    return emitOpError("dim is out of range for the operand rank");
  for (int64_t d = 0; d < rank; ++d) {
    int64_t expected = d == dim ? lhsTy.getDimSize(d) + rhsTy.getDimSize(d)
                                : lhsTy.getDimSize(d);
    if (d != dim && rhsTy.getDimSize(d) != lhsTy.getDimSize(d))
      return emitOpError("operand shapes must match outside dim");
    if (resultTy.getDimSize(d) != expected)
      return emitOpError("result shape must be the concatenation of the "
                         "operand shapes");
  }
  return success();
}

LogicalResult PadOp::verify() {
  auto inputTy = getRanked(getInput().getType());
  auto resultTy = getRanked(getResult().getType());
  if (failed(verifyPositiveShape(*this, inputTy, "input")) ||
      failed(verifyPositiveShape(*this, resultTy, "result")))
    return failure();
  int64_t rank = inputTy.getRank();
  if (inputTy.getElementType() != resultTy.getElementType())
    return emitOpError("input and result element types must match");
  if (static_cast<int64_t>(getLow().size()) != rank ||
      static_cast<int64_t>(getHigh().size()) != rank ||
      resultTy.getRank() != rank)
    return emitOpError("low, high and result rank must match the input "
                       "rank");
  for (int64_t d = 0; d < rank; ++d) {
    if (getLow()[d] < 0 || getHigh()[d] < 0)
      return emitOpError("padding amounts must be non-negative");
    if (resultTy.getDimSize(d) !=
        inputTy.getDimSize(d) + getLow()[d] + getHigh()[d])
      return emitOpError("result shape must be the padded input shape");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

LogicalResult ReverseOp::verify() {
  auto type = getRanked(getInput().getType());
  if (failed(verifyPositiveShape(*this, type, "input")))
    return failure();
  // The I64Attr getter is unsigned, so a negative dim reads as a huge
  // value and the single comparison rejects both directions.
  if (getDim() >= static_cast<uint64_t>(type.getRank()))
    return emitOpError("dim is out of range for the input rank");
  return success();
}

/// reverse(reverse(x)) == x along the same axis (a pure permutation);
/// a reversed small constant folds to a reversed constant.
OpFoldResult ReverseOp::fold(FoldAdaptor adaptor) {
  if (auto inner = getInput().getDefiningOp<ReverseOp>())
    if (inner.getDim() == getDim())
      return inner.getInput();
  if (auto dense = dyn_cast_or_null<DenseElementsAttr>(adaptor.getInput()))
    return foldPermutedConstant(dense, [&](ArrayRef<int64_t> shape) {
      int64_t dim = getDim();
      int64_t n = shape[dim];
      return [dim, n](SmallVectorImpl<int64_t> &index) {
        index[dim] = n - 1 - index[dim];
      };
    });
  return {};
}

//===----------------------------------------------------------------------===//
// FFTShiftOp
//===----------------------------------------------------------------------===//

LogicalResult FFTShiftOp::verify() {
  auto type = getRanked(getInput().getType());
  if (failed(verifyPositiveShape(*this, type, "input")))
    return failure();
  int64_t dim = getDim();
  if (dim < 0 || dim >= type.getRank())
    return emitOpError("dim is out of range for the input rank");
  return success();
}

namespace {
/// The shift calculus: element-wise operations commute with the pure
/// permutation `fftshift` is, bit-exactly -- rotating then operating
/// touches the same values as operating then rotating. Hoisting the
/// shift over element-wise consumers moves shifts toward each other,
/// where the shift/unshift fold cancels them; a spectrum multiplied in
/// its shifted form then costs no rotation at all.

/// binop(fftshift(a, d), fftshift(b, d)) -> fftshift(binop(a, b), d).
template <typename BinOpTy>
struct HoistShiftOverBinary : public OpRewritePattern<BinOpTy> {
  using OpRewritePattern<BinOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinOpTy op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs().template getDefiningOp<FFTShiftOp>();
    auto rhs = op.getRhs().template getDefiningOp<FFTShiftOp>();
    if (!lhs || !rhs || lhs.getDim() != rhs.getDim() ||
        lhs.getInverse() != rhs.getInverse())
      return failure();
    Value inner =
        BinOpTy::create(rewriter, op.getLoc(), lhs.getInput(), rhs.getInput());
    rewriter.replaceOpWithNewOp<FFTShiftOp>(op, inner, lhs.getDimAttr(),
                                            lhs.getInverseAttr());
    return success();
  }
};

/// unary(fftshift(x, d)) -> fftshift(unary(x), d) for single-operand
/// element-wise ops whose result shape matches the input.
template <typename UnaryOpTy>
struct HoistShiftOverUnary : public OpRewritePattern<UnaryOpTy> {
  using OpRewritePattern<UnaryOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnaryOpTy op,
                                PatternRewriter &rewriter) const override {
    auto shift = op.getInput().template getDefiningOp<FFTShiftOp>();
    if (!shift)
      return failure();
    auto inner =
        UnaryOpTy::create(rewriter, op.getLoc(), op.getResult().getType(),
                          shift.getInput(), op->getAttrs());
    rewriter.replaceOpWithNewOp<FFTShiftOp>(
        op, inner.getResult(), shift.getDimAttr(), shift.getInverseAttr());
    return success();
  }
};
} // namespace

void FFTShiftOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<HoistShiftOverBinary<AddOp>, HoistShiftOverBinary<SubOp>,
              HoistShiftOverBinary<MulOp>, HoistShiftOverBinary<DivOp>>(
      context);
  results
      .add<HoistShiftOverUnary<ConjOp>, HoistShiftOverUnary<RealOp>,
           HoistShiftOverUnary<ImagOp>, HoistShiftOverUnary<AbsOp>,
           HoistShiftOverUnary<MulScalarOp>, HoistShiftOverUnary<AddScalarOp>>(
          context);
}

//===----------------------------------------------------------------------===//
// FFTOp / IFFTOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyFFTLike(Operation *op, RankedTensorType type,
                                   int64_t dim) {
  if (failed(verifyPositiveShape(op, type, "input")))
    return failure();
  if (type.getRank() < 1 || type.getRank() > 2)
    return op->emitOpError("expects a rank-1 or rank-2 input");
  if (dim < 0 || dim >= type.getRank())
    return op->emitOpError("dim is out of range for the input rank");
  if (type.getDimSize(dim) < 2)
    return op->emitOpError("size along dim must be at least 2");
  return success();
}

LogicalResult FFTOp::verify() {
  return verifyFFTLike(*this, getRanked(getInput().getType()), getDim());
}

LogicalResult IFFTOp::verify() {
  return verifyFFTLike(*this, getRanked(getInput().getType()), getDim());
}

LogicalResult FFTSplitOp::verify() {
  return verifyFFTLike(*this, getRanked(getRe().getType()), getDim());
}

//===----------------------------------------------------------------------===//
// Interp1DOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyInterpLike(Operation *op, RankedTensorType dataTy,
                                      RankedTensorType posTy) {
  if (failed(verifyPositiveShape(op, dataTy, "data")) ||
      failed(verifyPositiveShape(op, posTy, "positions")))
    return failure();
  if (dataTy.getRank() != 2)
    return op->emitOpError("expects rank-2 data");
  if (posTy.getShape() != dataTy.getShape())
    return op->emitOpError("positions shape must match the data shape");
  if (!posTy.getElementType().isF64())
    return op->emitOpError("positions must have f64 elements");
  return success();
}

/// Validates the interpolation-kernel attributes shared by
/// sar.interp1d and sar.interp1d_split.
static LogicalResult verifyInterpKernel(Operation *op, StringRef kernel,
                                        int64_t taps, StringRef window,
                                        double beta, StringRef boundary) {
  if (kernel != "nearest" && kernel != "linear" && kernel != "cubic" &&
      kernel != "sinc")
    return op->emitOpError("kernel must be one of: nearest, linear, "
                           "cubic, sinc");
  if (window != "rect" && window != "hann" && window != "hamming" &&
      window != "kaiser")
    return op->emitOpError("window must be one of: rect, hann, "
                           "hamming, kaiser");
  // The upper bound keeps the unrolled I0 series accurate in HLS lowering.
  if (window == "kaiser" &&
      (!std::isfinite(beta) || beta <= 0.0 || beta > 12.0))
    return op->emitOpError("beta must be finite and in (0, 12]");
  if (kernel == "sinc") {
    if (taps < 4 || taps > 32 || taps % 2 != 0)
      return op->emitOpError("taps must be even and in [4, 32]");
  }
  if (boundary != "zero" && boundary != "edge" && boundary != "reflect")
    return op->emitOpError("boundary must be one of: zero, edge, reflect");
  return success();
}

LogicalResult Interp1DOp::verify() {
  if (getDim() > 1)
    return emitOpError("dim must be 0 or 1");
  if (failed(verifyInterpKernel(*this, getKernel(), getTaps(), getWindow(),
                                getBeta().convertToDouble(), getBoundary())))
    return failure();
  return verifyInterpLike(*this, getRanked(getData().getType()),
                          getRanked(getPositions().getType()));
}

LogicalResult Interp1DSplitOp::verify() {
  if (getDim() > 1)
    return emitOpError("dim must be 0 or 1");
  if (failed(verifyInterpKernel(*this, getKernel(), getTaps(), getWindow(),
                                getBeta().convertToDouble(), getBoundary())))
    return failure();
  return verifyInterpLike(*this, getRanked(getRe().getType()),
                          getRanked(getPositions().getType()));
}

//===----------------------------------------------------------------------===//
// IterateOp
//===----------------------------------------------------------------------===//

LogicalResult IterateOp::verify() {
  if (static_cast<int64_t>(getTrips()) < 1)
    return emitOpError("trips must be at least 1");
  Block &block = getBody().front();
  ValueRange inits = getInits();
  unsigned indexArgs = getIndex() ? 1 : 0;
  if (block.getNumArguments() != inits.size() + indexArgs ||
      getNumResults() != inits.size())
    return emitOpError("carries one block argument and one result per init")
           << (getIndex() ? " (plus the leading index argument)" : "");
  if (getIndex()) {
    auto indexType =
        RankedTensorType::get({1}, IntegerType::get(getContext(), 64));
    if (block.getArgument(0).getType() != indexType)
      return emitOpError("the index block argument must be tensor<1xi64>");
  }
  auto yield = cast<YieldOp>(block.getTerminator());
  if (yield.getNumOperands() != inits.size())
    return emitOpError("yields one value per init");
  for (auto [index, init] : llvm::enumerate(inits)) {
    Type type = init.getType();
    if (block.getArgument(index + indexArgs).getType() != type ||
        getResult(index).getType() != type ||
        yield.getOperand(index).getType() != type)
      return emitOpError("carry #")
             << index << " must keep one type through the init, the block "
             << "argument, the yield and the result";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Gather2DOp
//===----------------------------------------------------------------------===//

/// Shared checks of sar.gather2d and sar.gather2d_split: `dataTy` is the
/// (plane) type gathered from, `resultTy` the corresponding output.
static LogicalResult verifyGatherLike(Operation *op, RankedTensorType dataTy,
                                      RankedTensorType posTy,
                                      RankedTensorType resultTy,
                                      StringRef kernel, StringRef boundary) {
  if (failed(verifyPositiveShape(op, dataTy, "data")) ||
      failed(verifyPositiveShape(op, posTy, "positions")) ||
      failed(verifyPositiveShape(op, resultTy, "result")))
    return failure();
  if (dataTy.getRank() != 2)
    return op->emitOpError("expects rank-2 data");
  if (posTy.getRank() != 2 || !posTy.getElementType().isF64())
    return op->emitOpError("positions must be rank-2 f64 tensors");
  if (resultTy.getShape() != posTy.getShape())
    return op->emitOpError("result shape must match the position shape");
  if (kernel != "nearest" && kernel != "linear")
    return op->emitOpError("kernel must be one of: nearest, linear");
  if (boundary != "zero" && boundary != "edge")
    return op->emitOpError("boundary must be one of: zero, edge");
  return success();
}

LogicalResult Gather2DOp::verify() {
  auto dataTy = getRanked(getData().getType());
  auto resultTy = getRanked(getResult().getType());
  if (dataTy.getElementType() != resultTy.getElementType())
    return emitOpError("data and result element types must match");
  return verifyGatherLike(*this, dataTy, getRanked(getRows().getType()),
                          resultTy, getKernel(), getBoundary());
}

LogicalResult Gather2DSplitOp::verify() {
  auto planeTy = getRanked(getRe().getType());
  auto resultTy = getRanked(getOutRe().getType());
  if (planeTy.getElementType() != resultTy.getElementType())
    return emitOpError("plane and result element types must match");
  return verifyGatherLike(*this, planeTy, getRanked(getRows().getType()),
                          resultTy, getKernel(), getBoundary());
}

namespace {
/// Normalizes `sar.interp1d {dim = 0}` into transposes around the
/// canonical `dim = 1` form, the only one the lowerings implement.
struct NormalizeInterp1DDim : OpRewritePattern<Interp1DOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Interp1DOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getDim() != 0)
      return failure();
    Location loc = op.getLoc();
    auto transposedOf = [](Value v) {
      auto ty = cast<RankedTensorType>(v.getType());
      return RankedTensorType::get({ty.getDimSize(1), ty.getDimSize(0)},
                                   ty.getElementType());
    };
    Value data = TransposeOp::create(rewriter, loc, transposedOf(op.getData()),
                                     op.getData());
    Value positions = TransposeOp::create(
        rewriter, loc, transposedOf(op.getPositions()), op.getPositions());
    Value interp =
        Interp1DOp::create(rewriter, loc, data.getType(), data, positions,
                           /*dim=*/1, op.getKernel(), op.getTaps(),
                           op.getWindow(), op.getBeta(), op.getBoundary());
    rewriter.replaceOpWithNewOp<TransposeOp>(op, op.getType(), interp);
    return success();
  }
};
} // namespace

void Interp1DOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<NormalizeInterp1DDim>(context);
}

//===----------------------------------------------------------------------===//
// CumsumOp / RankFilterOp
//===----------------------------------------------------------------------===//

LogicalResult CumsumOp::verify() {
  auto type = getRanked(getInput().getType());
  if (failed(verifyPositiveShape(*this, type, "input")))
    return failure();
  if (type.getRank() < 1 || type.getRank() > 2)
    return emitOpError("expects a rank-1 or rank-2 input");
  Type element = type.getElementType();
  if (!isa<FloatType>(element) && !isa<ComplexType>(element))
    return emitOpError("expects float or complex elements");
  int64_t dim = getDim();
  if (dim < 0 || dim >= type.getRank())
    return emitOpError("dim is out of range for the input rank");
  return success();
}

LogicalResult RankFilterOp::verify() {
  auto type = getRanked(getInput().getType());
  if (failed(verifyPositiveShape(*this, type, "input")))
    return failure();
  if (type.getRank() < 1 || type.getRank() > 2)
    return emitOpError("expects a rank-1 or rank-2 input");
  if (!isa<FloatType>(type.getElementType()))
    return emitOpError("expects float elements");
  int64_t window = getWindow();
  if (window < 1 || window % 2 == 0)
    return emitOpError("window must be a positive odd integer");
  int64_t rank = getRank();
  if (rank < 0 || rank >= window)
    return emitOpError("rank must be in [0, window)");
  int64_t dim = getDim();
  if (dim < 0 || dim >= type.getRank())
    return emitOpError("dim is out of range for the input rank");
  return success();
}

LogicalResult SortOp::verify() {
  auto type = getRanked(getInput().getType());
  if (failed(verifyPositiveShape(*this, type, "input")))
    return failure();
  if (type.getRank() < 1 || type.getRank() > 2)
    return emitOpError("expects a rank-1 or rank-2 input");
  if (!isa<FloatType>(type.getElementType()))
    return emitOpError("expects float elements");
  int64_t dim = getDim();
  if (dim < 0 || dim >= type.getRank())
    return emitOpError("dim is out of range for the input rank");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "sar/Dialect/SAR/IR/SAROps.cpp.inc"
