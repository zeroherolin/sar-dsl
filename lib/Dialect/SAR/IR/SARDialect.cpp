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
  return builder.create<ConstantOp>(loc, type, elements);
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
  if (inner && inner.getDim() == getDim() &&
      inner.getInverse() != getInverse())
    return inner.getInput();
  return {};
}

/// x * 1.0 == x bit-exactly (preserves -0.0, NaN payloads and infinities).
OpFoldResult MulScalarOp::fold(FoldAdaptor) {
  if (getScalar().isExactlyValue(1.0))
    return getInput();
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
// ExpJOp
//===----------------------------------------------------------------------===//

LogicalResult ExpJOp::verify() {
  auto inputTy = getRanked(getInput().getType());
  auto resultTy = getRanked(getResult().getType());
  if (inputTy.getShape() != resultTy.getShape())
    return emitOpError("input and result shapes must match");
  auto complexTy = cast<ComplexType>(resultTy.getElementType());
  if (complexTy.getElementType() != inputTy.getElementType())
    return emitOpError("result complex precision must match the input float "
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
  // complex -> float is not a cast (use sar.abs or a dedicated op).
  if (inComplex && !outComplex)
    return emitOpError("cannot cast a complex tensor to a float tensor");
  return success();
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
  int64_t size = type.getDimSize(dim);
  if (size < 2 || !llvm::isPowerOf2_64(size))
    return op->emitOpError("size along dim must be a power of two (>= 2)");
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

LogicalResult Interp1DOp::verify() {
  return verifyInterpLike(*this, getRanked(getData().getType()),
                          getRanked(getPositions().getType()));
}

LogicalResult Interp1DSplitOp::verify() {
  return verifyInterpLike(*this, getRanked(getRe().getType()),
                          getRanked(getPositions().getType()));
}

//===----------------------------------------------------------------------===//
// StoltInterpOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyStoltLike(Operation *op, RankedTensorType dataTy,
                                     RankedTensorType faTy,
                                     RankedTensorType frTy) {
  if (dataTy.getRank() != 2)
    return op->emitOpError("expects rank-2 data");
  if (faTy.getRank() != 1 || frTy.getRank() != 1)
    return op->emitOpError("fa and fr must be rank-1 tensors");
  if (!faTy.getElementType().isF64() || !frTy.getElementType().isF64())
    return op->emitOpError("fa and fr must have f64 elements");
  if (faTy.getDimSize(0) != dataTy.getDimSize(0))
    return op->emitOpError("fa length must equal the azimuth (first) "
                           "dimension of data");
  if (frTy.getDimSize(0) != dataTy.getDimSize(1))
    return op->emitOpError("fr length must equal the range (second) "
                           "dimension of data");
  if (frTy.getDimSize(0) < 2)
    return op->emitOpError("fr must contain at least two samples");
  return success();
}

LogicalResult StoltInterpOp::verify() {
  if (getData().getType() != getResult().getType())
    return emitOpError("data and result types must match");
  return verifyStoltLike(*this, getRanked(getData().getType()),
                         getRanked(getFa().getType()),
                         getRanked(getFr().getType()));
}

LogicalResult StoltInterpSplitOp::verify() {
  return verifyStoltLike(*this, getRanked(getRe().getType()),
                         getRanked(getFa().getType()),
                         getRanked(getFr().getType()));
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "sar/Dialect/SAR/IR/SAROps.cpp.inc"
