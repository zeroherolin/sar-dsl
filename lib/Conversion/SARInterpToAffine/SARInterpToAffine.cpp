//===- SARInterpToAffine.cpp - Windowed-sinc gathers as affine loops ------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
// Lowers the split-complex interpolation ops (sar.interp1d_split,
// sar.stolt_interp_split) into affine loop nests for HLS flows. The loop
// structure is affine; the gather itself is a data-dependent memref.load
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

constexpr int64_t kTapLo = -3;
constexpr int64_t kTapHi = 4;

/// Scalar helpers around an OpBuilder at a fixed location.
struct ScalarBuilder {
  OpBuilder &b;
  Location loc;
  FloatType f64;

  Value cst(double v) {
    return b.create<arith::ConstantOp>(loc, b.getFloatAttr(f64, v));
  }
  Value add(Value x, Value y) { return b.create<arith::AddFOp>(loc, x, y); }
  Value sub(Value x, Value y) { return b.create<arith::SubFOp>(loc, x, y); }
  Value mul(Value x, Value y) { return b.create<arith::MulFOp>(loc, x, y); }
  Value div(Value x, Value y) { return b.create<arith::DivFOp>(loc, x, y); }
  Value cos(Value x) { return b.create<math::CosOp>(loc, x); }
  Value sin(Value x) { return b.create<math::SinOp>(loc, x); }

  /// Widens a loaded element to f64 if needed.
  Value toF64(Value v) {
    if (v.getType() == f64)
      return v;
    return b.create<arith::ExtFOp>(loc, f64, v);
  }
  /// Narrows an f64 value to the storage element type if needed.
  Value fromF64(Value v, Type elemType) {
    if (elemType == f64)
      return v;
    return b.create<arith::TruncFOp>(loc, elemType, v);
  }
};

/// Emits the 8-tap windowed-sinc gather from one row of (re, im) buffers at
/// fractional position `posF64`. Returns the (re, im) accumulators in f64.
/// Taps are statically unrolled; out-of-range taps read a clamped index and
/// are masked to zero weight, keeping the body free of control flow.
static std::pair<Value, Value>
emitWindowedSincGather(ScalarBuilder &s, Value reBuf, Value imBuf, Value row,
                       Value posF64, int64_t cols) {
  OpBuilder &b = s.b;
  Location loc = s.loc;

  // floor() via truncation plus a negative-fraction fixup: fptosi rounds
  // toward zero, and the ScaleHLS HLS emitter has no math.floor.
  Value truncI = b.create<arith::FPToSIOp>(loc, b.getI64Type(), posF64);
  Value truncF = b.create<arith::SIToFPOp>(loc, s.f64, truncI);
  Value hasNegFrac = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT,
                                             posF64, truncF);
  Value oneI = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(1));
  Value truncMinusOne = b.create<arith::SubIOp>(loc, truncI, oneI);
  Value idx0 = b.create<arith::SelectOp>(loc, hasNegFrac, truncMinusOne,
                                         truncI);

  Value zeroI = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(0));
  Value maxI = b.create<arith::ConstantOp>(loc,
                                           b.getI64IntegerAttr(cols - 1));
  Value colsI = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(cols));

  Value pi = s.cst(M_PI);
  Value piQuarter = s.cst(M_PI / 4.0);
  Value half = s.cst(0.5);
  Value one = s.cst(1.0);
  Value zero = s.cst(0.0);
  Value eps = s.cst(1e-12);

  Value accRe = zero, accIm = zero;
  for (int64_t k = kTapLo; k <= kTapHi; ++k) {
    Value kI = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(k));
    Value idxK = b.create<arith::AddIOp>(loc, idx0, kI);

    Value inLo = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                         idxK, zeroI);
    Value inHi = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                         idxK, colsI);
    Value inBounds = b.create<arith::AndIOp>(loc, inLo, inHi);

    Value clamped = b.create<arith::MinSIOp>(
        loc, b.create<arith::MaxSIOp>(loc, idxK, zeroI), maxI);
    Value col = b.create<arith::IndexCastOp>(loc, b.getIndexType(), clamped);

    Value vRe = s.toF64(
        b.create<memref::LoadOp>(loc, reBuf, ValueRange{row, col}));
    Value vIm = s.toF64(
        b.create<memref::LoadOp>(loc, imBuf, ValueRange{row, col}));

    // w(d) = sinc(d) * (0.5 + 0.5 cos(pi d / 4)), numpy sinc convention.
    Value dist = s.sub(posF64, b.create<arith::SIToFPOp>(loc, s.f64, idxK));
    Value pd = s.mul(dist, pi);
    Value sincRaw = s.div(s.sin(pd), pd);
    Value small = b.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT,
        b.create<math::AbsFOp>(loc, dist), eps);
    Value sinc = b.create<arith::SelectOp>(loc, small, one, sincRaw);
    Value window = s.add(half, s.mul(half, s.cos(s.mul(dist, piQuarter))));
    Value weight = s.mul(sinc, window);
    weight = b.create<arith::SelectOp>(loc, inBounds, weight, zero);

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
  return rewriter.create<bufferization::ToBufferOp>(loc, bufferType, tensor);
}

static Value toResultTensor(PatternRewriter &rewriter, Location loc,
                            RankedTensorType type, Value alloc) {
  return rewriter.create<bufferization::ToTensorOp>(
      loc, type, alloc, /*restrict=*/true, /*writable=*/true);
}

struct Interp1DSplitLowering : OpRewritePattern<Interp1DSplitOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Interp1DSplitOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto tensorType = cast<RankedTensorType>(op.getRe().getType());
    Type elemType = tensorType.getElementType();
    int64_t rows = tensorType.getDimSize(0);
    int64_t cols = tensorType.getDimSize(1);
    auto bufferType = MemRefType::get(tensorType.getShape(), elemType);

    Value reBuf = toBuffer(rewriter, loc, op.getRe());
    Value imBuf = toBuffer(rewriter, loc, op.getIm());
    Value posBuf = toBuffer(rewriter, loc, op.getPositions());
    Value outRe = rewriter.create<memref::AllocOp>(loc, bufferType);
    Value outIm = rewriter.create<memref::AllocOp>(loc, bufferType);

    auto rowLoop = rewriter.create<affine::AffineForOp>(loc, 0, rows);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      auto colLoop = rewriter.create<affine::AffineForOp>(loc, 0, cols);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      Value i = rowLoop.getInductionVar();
      Value j = colLoop.getInductionVar();

      ScalarBuilder s{rewriter, loc, rewriter.getF64Type()};
      Value pos = rewriter.create<affine::AffineLoadOp>(loc, posBuf,
                                                        ValueRange{i, j});
      auto [accRe, accIm] =
          emitWindowedSincGather(s, reBuf, imBuf, i, pos, cols);
      rewriter.create<affine::AffineStoreOp>(loc, s.fromF64(accRe, elemType),
                                             outRe, ValueRange{i, j});
      rewriter.create<affine::AffineStoreOp>(loc, s.fromF64(accIm, elemType),
                                             outIm, ValueRange{i, j});
    }

    rewriter.replaceOp(op, {toResultTensor(rewriter, loc, tensorType, outRe),
                            toResultTensor(rewriter, loc, tensorType, outIm)});
    return success();
  }
};

struct StoltInterpSplitLowering : OpRewritePattern<StoltInterpSplitOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(StoltInterpSplitOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto tensorType = cast<RankedTensorType>(op.getRe().getType());
    Type elemType = tensorType.getElementType();
    int64_t rows = tensorType.getDimSize(0);
    int64_t cols = tensorType.getDimSize(1);
    auto bufferType = MemRefType::get(tensorType.getShape(), elemType);
    auto f64 = rewriter.getF64Type();
    auto scratchType = MemRefType::get(tensorType.getShape(), f64);

    Value reBuf = toBuffer(rewriter, loc, op.getRe());
    Value imBuf = toBuffer(rewriter, loc, op.getIm());
    Value faBuf = toBuffer(rewriter, loc, op.getFa());
    Value frBuf = toBuffer(rewriter, loc, op.getFr());
    Value smoothRe = rewriter.create<memref::AllocOp>(loc, scratchType);
    Value smoothIm = rewriter.create<memref::AllocOp>(loc, scratchType);
    Value outRe = rewriter.create<memref::AllocOp>(loc, bufferType);
    Value outIm = rewriter.create<memref::AllocOp>(loc, bufferType);

    ScalarBuilder s{rewriter, loc, f64};
    Value cVal = s.cst(op.getC().convertToDouble());
    Value fcVal = s.cst(op.getFc().convertToDouble());
    Value twoVr = s.cst(2.0 * op.getVr().convertToDouble());
    double tShift = op.getTShift().convertToDouble();
    Value twoPiT = s.cst(2.0 * M_PI * tShift);
    Value negTwoPiT = s.cst(-2.0 * M_PI * tShift);
    Value epsTerm = s.cst(1e-10);

    Value idx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value idx1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value fStart = rewriter.create<memref::LoadOp>(loc, frBuf,
                                                   ValueRange{idx0});
    Value df = s.sub(
        rewriter.create<memref::LoadOp>(loc, frBuf, ValueRange{idx1}),
        fStart);

    // Stage 1: smoothing ramp, smooth = data * exp(+2 pi j fr[j] t_shift).
    {
      auto rowLoop = rewriter.create<affine::AffineForOp>(loc, 0, rows);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      auto colLoop = rewriter.create<affine::AffineForOp>(loc, 0, cols);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      Value i = rowLoop.getInductionVar();
      Value j = colLoop.getInductionVar();

      Value vRe = s.toF64(rewriter.create<affine::AffineLoadOp>(
          loc, reBuf, ValueRange{i, j}));
      Value vIm = s.toF64(rewriter.create<affine::AffineLoadOp>(
          loc, imBuf, ValueRange{i, j}));
      Value frj = rewriter.create<affine::AffineLoadOp>(loc, frBuf,
                                                        ValueRange{j});
      Value phase = s.mul(twoPiT, frj);
      Value cs = s.cos(phase), sn = s.sin(phase);
      rewriter.create<affine::AffineStoreOp>(
          loc, s.sub(s.mul(vRe, cs), s.mul(vIm, sn)), smoothRe,
          ValueRange{i, j});
      rewriter.create<affine::AffineStoreOp>(
          loc, s.add(s.mul(vRe, sn), s.mul(vIm, cs)), smoothIm,
          ValueRange{i, j});
    }

    // Stage 2: Stolt frequency mapping, gather and de-smoothing ramp.
    {
      auto rowLoop = rewriter.create<affine::AffineForOp>(loc, 0, rows);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      Value i = rowLoop.getInductionVar();

      Value fai = rewriter.create<affine::AffineLoadOp>(loc, faBuf,
                                                        ValueRange{i});
      Value faTerm = s.div(s.mul(cVal, fai), twoVr);
      Value faTerm2 = s.mul(faTerm, faTerm);

      auto colLoop = rewriter.create<affine::AffineForOp>(loc, 0, cols);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      Value j = colLoop.getInductionVar();

      Value frj = rewriter.create<affine::AffineLoadOp>(loc, frBuf,
                                                        ValueRange{j});
      Value shifted = s.add(frj, fcVal);
      Value term = s.add(s.mul(shifted, shifted), faTerm2);
      // cmpf+select instead of arith.maxnumf: keeps the IR parseable by
      // the older LLVM pinned by the ScaleHLS toolchain.
      Value greater = rewriter.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::OGT, term, epsTerm);
      term = rewriter.create<arith::SelectOp>(loc, greater, term, epsTerm);
      Value frq = s.sub(rewriter.create<math::SqrtOp>(loc, term), fcVal);
      Value pos = s.div(s.sub(frq, fStart), df);

      auto [accRe, accIm] =
          emitWindowedSincGather(s, smoothRe, smoothIm, i, pos, cols);

      Value phase = s.mul(negTwoPiT, frq);
      Value cs = s.cos(phase), sn = s.sin(phase);
      Value resRe = s.sub(s.mul(accRe, cs), s.mul(accIm, sn));
      Value resIm = s.add(s.mul(accRe, sn), s.mul(accIm, cs));
      rewriter.create<affine::AffineStoreOp>(loc, s.fromF64(resRe, elemType),
                                             outRe, ValueRange{i, j});
      rewriter.create<affine::AffineStoreOp>(loc, s.fromF64(resIm, elemType),
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
    target.addIllegalOp<Interp1DSplitOp, StoltInterpSplitOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    patterns.add<Interp1DSplitLowering, StoltInterpSplitLowering>(context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
