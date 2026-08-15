//===- SARInterpToAffine.cpp - Interpolation gathers as affine loops ------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
// Lowers the split-complex interpolation op (sar.interp1d_split) into
// affine loop nests for HLS flows. The loop structure is affine; the
// gather itself is a data-dependent memref.load
// with a clamped index, and out-of-range taps are masked with selects so
// the loop body stays straight-line (pipelining-friendly, no control flow).
//
// All position/weight arithmetic is performed in f64, mirroring the CPU
// runtime bit-for-bit for f64 data; f32 planes are widened for the
// accumulation and truncated on store.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "sar/Conversion/Passes.h"
#include "sar/Dialect/SAR/IR/SARDialect.h"
#include "sar/Dialect/SAR/IR/SAROps.h"

#include <cmath>

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_CONVERTSARINTERPTOAFFINE
#include "sar/Conversion/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;

namespace {

/// Compile-time interpolation kernel configuration (attribute values of
/// the op being lowered).
struct InterpSpec {
  StringRef kernel;
  int64_t taps;
  StringRef window;
  double beta;
};

/// Modified Bessel I0 for the host-side normalization constant.
static double besselI0(double x) {
  double sum = 1.0, term = 1.0, halfX = x / 2.0;
  for (int m = 1; m <= 40; ++m) {
    term *= (halfX / m) * (halfX / m);
    sum += term;
    if (term < 1e-16 * sum)
      break;
  }
  return sum;
}

/// Scalar helpers around an OpBuilder at a fixed location.
struct ScalarBuilder {
  OpBuilder &b;
  Location loc;
  FloatType f64;

  Value cst(double v) {
    return arith::ConstantOp::create(b, loc, b.getFloatAttr(f64, v));
  }
  Value add(Value x, Value y) { return arith::AddFOp::create(b, loc, x, y); }
  Value sub(Value x, Value y) { return arith::SubFOp::create(b, loc, x, y); }
  Value mul(Value x, Value y) { return arith::MulFOp::create(b, loc, x, y); }
  Value div(Value x, Value y) { return arith::DivFOp::create(b, loc, x, y); }
  Value cos(Value x) { return math::CosOp::create(b, loc, x); }
  Value sin(Value x) { return math::SinOp::create(b, loc, x); }

  /// Widens a loaded element to f64 if needed.
  Value toF64(Value v) {
    if (v.getType() == f64)
      return v;
    return arith::ExtFOp::create(b, loc, f64, v);
  }
  /// Narrows an f64 value to the storage element type if needed.
  Value fromF64(Value v, Type elemType) {
    if (elemType == f64)
      return v;
    return arith::TruncFOp::create(b, loc, elemType, v);
  }
};

/// Emits floor(posF64) as i64 via truncation plus a negative-fraction
/// fixup: fptosi rounds toward zero, and the HLS emitter has no
/// math.floor.
static Value emitFloorI64(ScalarBuilder &s, Value posF64) {
  OpBuilder &b = s.b;
  Location loc = s.loc;
  Value truncI = arith::FPToSIOp::create(b, loc, b.getI64Type(), posF64);
  Value truncF = arith::SIToFPOp::create(b, loc, s.f64, truncI);
  Value hasNegFrac =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, posF64, truncF);
  Value oneI = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(1));
  Value truncMinusOne = arith::SubIOp::create(b, loc, truncI, oneI);
  return arith::SelectOp::create(b, loc, hasNegFrac, truncMinusOne, truncI);
}

/// Emits the window taper at t = d / (taps/2) for the sinc kernel.
/// Kaiser uses an unrolled power series for I0 (straight-line, no loops):
/// with u = (beta^2 / 4) (1 - t^2), I0 = sum_m u^m / (m!)^2.
static Value emitSincWindow(ScalarBuilder &s, const InterpSpec &spec, Value t) {
  OpBuilder &b = s.b;
  Location loc = s.loc;
  if (spec.window == "rect")
    return s.cst(1.0);
  if (spec.window == "hann" || spec.window == "hamming") {
    double c0 = spec.window == "hann" ? 0.5 : 0.54;
    double c1 = spec.window == "hann" ? 0.5 : 0.46;
    return s.add(s.cst(c0), s.mul(s.cst(c1), s.cos(s.mul(t, s.cst(M_PI)))));
  }
  // kaiser
  Value tt = s.mul(t, t);
  Value arg = s.sub(s.cst(1.0), tt);
  Value zero = s.cst(0.0);
  Value neg =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, arg, zero);
  arg = arith::SelectOp::create(b, loc, neg, zero, arg);
  Value u = s.mul(s.cst(spec.beta * spec.beta / 4.0), arg);
  Value sum = s.cst(1.0), term = s.cst(1.0);
  for (int m = 1; m <= 32; ++m) {
    term = s.mul(term, s.mul(u, s.cst(1.0 / (double(m) * double(m)))));
    sum = s.add(sum, term);
  }
  return s.mul(sum, s.cst(1.0 / besselI0(spec.beta)));
}

/// Emits the kernel-based gather from one row of (re, im) buffers at
/// fractional position `posF64`. Returns the (re, im) accumulators in f64.
/// Taps are statically unrolled; out-of-range taps read a clamped index and
/// are masked to zero weight, keeping the body free of control flow.
static std::pair<Value, Value>
emitInterpGather(ScalarBuilder &s, const InterpSpec &spec, Value reBuf,
                 Value imBuf, Value row, Value posF64, int64_t cols) {
  OpBuilder &b = s.b;
  Location loc = s.loc;

  // Tap range relative to floor(position), and the rounding base:
  // nearest uses floor(position + 0.5) with a single tap.
  int64_t tapLo, tapHi;
  Value base = posF64;
  if (spec.kernel == "nearest") {
    tapLo = tapHi = 0;
    base = s.add(posF64, s.cst(0.5));
  } else if (spec.kernel == "linear") {
    tapLo = 0;
    tapHi = 1;
  } else if (spec.kernel == "cubic") {
    tapLo = -1;
    tapHi = 2;
  } else {
    tapLo = 1 - spec.taps / 2;
    tapHi = spec.taps / 2;
  }
  Value idx0 = emitFloorI64(s, base);
  Value idx0F = arith::SIToFPOp::create(b, loc, s.f64, idx0);
  Value frac = s.sub(base, idx0F);

  Value zeroI = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(0));
  Value maxI = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(cols - 1));
  Value colsI = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(cols));

  Value one = s.cst(1.0);
  Value zero = s.cst(0.0);

  // Keys cubic convolution weights (a = -0.5) as polynomials of frac;
  // straight-line, no branches.
  SmallVector<Value, 4> cubicW;
  if (spec.kernel == "cubic") {
    Value f1 = frac, f2 = s.mul(frac, frac), f3 = s.mul(f2, frac);
    auto poly = [&](double a3, double a2, double a1, double a0) {
      Value v = s.mul(s.cst(a3), f3);
      v = s.add(v, s.mul(s.cst(a2), f2));
      v = s.add(v, s.mul(s.cst(a1), f1));
      return s.add(v, s.cst(a0));
    };
    cubicW = {poly(-0.5, 1.0, -0.5, 0.0), poly(1.5, -2.5, 0.0, 1.0),
              poly(-1.5, 2.0, 0.5, 0.0), poly(0.5, -0.5, 0.0, 0.0)};
  }

  Value accRe = zero, accIm = zero;
  for (int64_t k = tapLo; k <= tapHi; ++k) {
    Value kI = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(k));
    Value idxK = arith::AddIOp::create(b, loc, idx0, kI);

    Value inLo =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, idxK, zeroI);
    Value inHi =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, idxK, colsI);
    Value inBounds = arith::AndIOp::create(b, loc, inLo, inHi);

    Value clamped = arith::MinSIOp::create(
        b, loc, arith::MaxSIOp::create(b, loc, idxK, zeroI), maxI);
    Value col = arith::IndexCastOp::create(b, loc, b.getIndexType(), clamped);

    Value vRe =
        s.toF64(memref::LoadOp::create(b, loc, reBuf, ValueRange{row, col}));
    Value vIm =
        s.toF64(memref::LoadOp::create(b, loc, imBuf, ValueRange{row, col}));

    Value weight;
    if (spec.kernel == "nearest") {
      weight = one;
    } else if (spec.kernel == "linear") {
      weight = k == 0 ? s.sub(one, frac) : frac;
    } else if (spec.kernel == "cubic") {
      weight = cubicW[k + 1];
    } else {
      // w(d) = sinc(d) * window(d / (taps/2)), numpy sinc convention.
      Value dist = s.sub(posF64, arith::SIToFPOp::create(b, loc, s.f64, idxK));
      Value pd = s.mul(dist, s.cst(M_PI));
      Value sincRaw = s.div(s.sin(pd), pd);
      Value small = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                          math::AbsFOp::create(b, loc, dist),
                                          s.cst(1e-12));
      Value sinc = arith::SelectOp::create(b, loc, small, one, sincRaw);
      Value t = s.mul(dist, s.cst(1.0 / double(spec.taps / 2)));
      weight = s.mul(sinc, emitSincWindow(s, spec, t));
    }
    weight = arith::SelectOp::create(b, loc, inBounds, weight, zero);

    accRe = s.add(accRe, s.mul(vRe, weight));
    accIm = s.add(accIm, s.mul(vIm, weight));
  }
  return {accRe, accIm};
}

/// Materializes a tensor as a buffer (identity layout).
static Value toBuffer(PatternRewriter &rewriter, Location loc, Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto bufferType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  return bufferization::ToBufferOp::create(rewriter, loc, bufferType, tensor);
}

static Value toResultTensor(PatternRewriter &rewriter, Location loc,
                            RankedTensorType type, Value alloc) {
  return bufferization::ToTensorOp::create(
      rewriter, loc, type, alloc, /*restrict=*/true, /*writable=*/true);
}

struct Interp1DSplitLowering : OpRewritePattern<Interp1DSplitOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Interp1DSplitOp op,
                                PatternRewriter &rewriter) const override {
    // dim = 0 is normalized into transposes by canonicalization, which
    // runs before sar-decomplexify in the affine pipeline.
    if (op.getDim() != 1)
      return op.emitOpError(
          "interpolation with dim = 0 must be canonicalized to dim = 1 with "
          "transposes; run --canonicalize before this pass");
    Location loc = op.getLoc();
    auto tensorType = cast<RankedTensorType>(op.getRe().getType());
    Type elemType = tensorType.getElementType();
    int64_t rows = tensorType.getDimSize(0);
    int64_t cols = tensorType.getDimSize(1);
    auto bufferType = MemRefType::get(tensorType.getShape(), elemType);

    Value reBuf = toBuffer(rewriter, loc, op.getRe());
    Value imBuf = toBuffer(rewriter, loc, op.getIm());
    Value posBuf = toBuffer(rewriter, loc, op.getPositions());
    Value outRe = memref::AllocOp::create(rewriter, loc, bufferType);
    Value outIm = memref::AllocOp::create(rewriter, loc, bufferType);

    auto rowLoop = affine::AffineForOp::create(rewriter, loc, 0, rows);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      auto colLoop = affine::AffineForOp::create(rewriter, loc, 0, cols);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      Value i = rowLoop.getInductionVar();
      Value j = colLoop.getInductionVar();

      ScalarBuilder s{rewriter, loc, rewriter.getF64Type()};
      Value pos =
          affine::AffineLoadOp::create(rewriter, loc, posBuf, ValueRange{i, j});
      InterpSpec spec{op.getKernel(), static_cast<int64_t>(op.getTaps()),
                      op.getWindow(), op.getBeta().convertToDouble()};
      auto [accRe, accIm] =
          emitInterpGather(s, spec, reBuf, imBuf, i, pos, cols);
      affine::AffineStoreOp::create(rewriter, loc, s.fromF64(accRe, elemType),
                                    outRe, ValueRange{i, j});
      affine::AffineStoreOp::create(rewriter, loc, s.fromF64(accIm, elemType),
                                    outIm, ValueRange{i, j});
    }

    rewriter.replaceOp(op, {toResultTensor(rewriter, loc, tensorType, outRe),
                            toResultTensor(rewriter, loc, tensorType, outIm)});
    return success();
  }
};

struct ConvertSARInterpToAffinePass
    : sar::impl::ConvertSARInterpToAffineBase<ConvertSARInterpToAffinePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addIllegalOp<Interp1DSplitOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    patterns.add<Interp1DSplitLowering>(context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
