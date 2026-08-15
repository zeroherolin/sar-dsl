//===- SARDialect.cpp - SAR dialect and op implementations ---------------===//
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

/// transpose(transpose(x)) == x (a pure permutation, bit-exact).
OpFoldResult TransposeOp::fold(FoldAdaptor) {
  if (auto inner = getInput().getDefiningOp<TransposeOp>())
    return inner.getInput();
  return {};
}

/// fftshift and ifftshift along the same axis are exact inverses for any
/// size (forward rotates by ceil(n/2), inverse by floor(n/2)).
OpFoldResult FFTShiftOp::fold(FoldAdaptor) {
  auto inner = getInput().getDefiningOp<FFTShiftOp>();
  if (inner && inner.getDim() == getDim() && inner.getInverse() != getInverse())
    return inner.getInput();
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

LogicalResult ConcatOp::verify() {
  auto lhsTy = getRanked(getLhs().getType());
  auto rhsTy = getRanked(getRhs().getType());
  auto resultTy = getRanked(getResult().getType());
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
  if (getDim() >= static_cast<uint64_t>(type.getRank()))
    return emitOpError("dim is out of range for the input rank");
  return success();
}

/// reverse(reverse(x)) == x along the same axis (a pure permutation).
OpFoldResult ReverseOp::fold(FoldAdaptor) {
  if (auto inner = getInput().getDefiningOp<ReverseOp>())
    if (inner.getDim() == getDim())
      return inner.getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// FFTShiftOp
//===----------------------------------------------------------------------===//

LogicalResult FFTShiftOp::verify() {
  auto type = getRanked(getInput().getType());
  int64_t dim = getDim();
  if (dim < 0 || dim >= type.getRank())
    return emitOpError("dim is out of range for the input rank");
  return success();
}

//===----------------------------------------------------------------------===//
// FFTOp / IFFTOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyFFTLike(Operation *op, RankedTensorType type,
                                   int64_t dim) {
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
                                        double beta) {
  if (kernel != "nearest" && kernel != "linear" && kernel != "cubic" &&
      kernel != "sinc")
    return op->emitOpError("kernel must be one of: nearest, linear, "
                           "cubic, sinc");
  if (kernel == "sinc") {
    if (taps < 4 || taps > 32 || taps % 2 != 0)
      return op->emitOpError("taps must be even and in [4, 32]");
    if (window != "rect" && window != "hann" && window != "hamming" &&
        window != "kaiser")
      return op->emitOpError("window must be one of: rect, hann, "
                             "hamming, kaiser");
    // The upper bound keeps the unrolled I0 series in the affine (HLS)
    // lowering accurate.
    if (window == "kaiser" && (beta <= 0.0 || beta > 12.0))
      return op->emitOpError("beta must be in (0, 12]");
  }
  return success();
}

LogicalResult Interp1DOp::verify() {
  if (getDim() > 1)
    return emitOpError("dim must be 0 or 1");
  if (failed(verifyInterpKernel(*this, getKernel(), getTaps(), getWindow(),
                                getBeta().convertToDouble())))
    return failure();
  return verifyInterpLike(*this, getRanked(getData().getType()),
                          getRanked(getPositions().getType()));
}

LogicalResult Interp1DSplitOp::verify() {
  if (getDim() > 1)
    return emitOpError("dim must be 0 or 1");
  if (failed(verifyInterpKernel(*this, getKernel(), getTaps(), getWindow(),
                                getBeta().convertToDouble())))
    return failure();
  return verifyInterpLike(*this, getRanked(getRe().getType()),
                          getRanked(getPositions().getType()));
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
    Value interp = Interp1DOp::create(
        rewriter, loc, data.getType(), data, positions, /*dim=*/1,
        op.getKernel(), op.getTaps(), op.getWindow(), op.getBeta());
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
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "sar/Dialect/SAR/IR/SAROps.cpp.inc"
