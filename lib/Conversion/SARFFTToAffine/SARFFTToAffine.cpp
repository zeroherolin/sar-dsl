//===- SARFFTToAffine.cpp - Stockham FFT as affine loops ------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
// Lowers sar.fft_split (the split-complex FFT produced by sar-decomplexify)
// into affine loop nests implementing the radix-2 Stockham autosort FFT.
// Stockham is chosen over Cooley-Tukey deliberately: it needs no
// bit-reversal permutation, so every memory access is an affine function of
// the loop indices -- exactly what HLS flows (ScaleHLS-HIDA) require for
// loop analysis, pipelining and array partitioning.
//
// Per stage t (statically unrolled; n_cur = L >> t, m = n_cur / 2,
// s = 1 << t, DIF formulation):
//
//   for p in [0, m), q in [0, s):
//     a = X[q + s p];  b = X[q + s (p + m)]
//     Y[q + 2 s p]     = a + b
//     Y[q + s (2p + 1)] = (a - b) * w,   w = exp(-+ 2 pi i p / n_cur)
//
// Twiddle factors are precomputed into constant memref globals; X/Y are
// ping-pong scratch buffers alternated between the unrolled stages.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "sar/Conversion/Passes.h"
#include "sar/Dialect/SAR/IR/SARDialect.h"
#include "sar/Dialect/SAR/IR/SAROps.h"

#include <cmath>

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_CONVERTSARFFTTOAFFINE
#include "sar/Conversion/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;

namespace {

/// Ensures a private constant memref global holding `values` exists and
/// returns its name.
static std::string ensureTwiddleGlobal(PatternRewriter &rewriter,
                                       ModuleOp module, StringRef prefix,
                                       int64_t length, FloatType elemType,
                                       ArrayRef<double> values) {
  std::string name =
      (prefix + "_" + Twine(length) + "_" +
       (elemType.isF32() ? "f32" : "f64")).str();
  if (module.lookupSymbol(name))
    return name;

  auto memrefType = MemRefType::get({length}, elemType);
  auto tensorType = RankedTensorType::get({length}, elemType);
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (double v : values)
    attrs.push_back(rewriter.getFloatAttr(elemType, v));
  auto initial = DenseElementsAttr::get(tensorType, attrs);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<memref::GlobalOp>(
      module.getLoc(), name, rewriter.getStringAttr("private"), memrefType,
      initial, /*constant=*/true, /*alignment=*/nullptr);
  return name;
}

struct FFTSplitToAffinePattern : OpRewritePattern<FFTSplitOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FFTSplitOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto tensorType = cast<RankedTensorType>(op.getRe().getType());
    auto elemType = cast<FloatType>(tensorType.getElementType());
    MLIRContext *ctx = rewriter.getContext();

    int64_t rank = tensorType.getRank();
    int64_t dim = op.getDim();
    int64_t length = tensorType.getDimSize(dim);
    int64_t lines = rank == 2 ? tensorType.getDimSize(1 - dim) : 1;
    bool inverse = op.getInverse();
    int stages = llvm::Log2_64(length);

    // Flattened per-stage twiddle tables: stage t occupies
    // [L - (L >> t), L - (L >> t) + m) with m = L >> (t + 1) entries of
    // cos/sin(2 pi p / n_cur).
    SmallVector<double> cosTable, sinTable;
    cosTable.reserve(length - 1);
    sinTable.reserve(length - 1);
    for (int t = 0; t < stages; ++t) {
      int64_t nCur = length >> t;
      for (int64_t p = 0; p < nCur / 2; ++p) {
        double angle = 2.0 * M_PI * static_cast<double>(p) /
                       static_cast<double>(nCur);
        cosTable.push_back(std::cos(angle));
        sinTable.push_back(std::sin(angle));
      }
    }
    std::string cosName = ensureTwiddleGlobal(
        rewriter, module, "__sar_fft_twiddle_cos", length - 1, elemType,
        cosTable);
    std::string sinName = ensureTwiddleGlobal(
        rewriter, module, "__sar_fft_twiddle_sin", length - 1, elemType,
        sinTable);

    auto bufferType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    auto twiddleType = MemRefType::get({length - 1}, elemType);

    Value inRe = rewriter.create<bufferization::ToBufferOp>(loc, bufferType,
                                                            op.getRe());
    Value inIm = rewriter.create<bufferization::ToBufferOp>(loc, bufferType,
                                                            op.getIm());
    Value cosBuf =
        rewriter.create<memref::GetGlobalOp>(loc, twiddleType, cosName);
    Value sinBuf =
        rewriter.create<memref::GetGlobalOp>(loc, twiddleType, sinName);

    // Ping-pong scratch buffers; the pair holding the final stage result is
    // exposed through to_tensor.
    Value bufARe = rewriter.create<memref::AllocOp>(loc, bufferType);
    Value bufAIm = rewriter.create<memref::AllocOp>(loc, bufferType);
    Value bufBRe = rewriter.create<memref::AllocOp>(loc, bufferType);
    Value bufBIm = rewriter.create<memref::AllocOp>(loc, bufferType);

    // Identity access map over (line, elem) in storage order.
    auto lineDim = getAffineDimExpr(0, ctx);
    auto pDim = getAffineDimExpr(1, ctx);
    auto qDim = getAffineDimExpr(2, ctx);

    // Builds the storage map for an element index expression. Map operands
    // are always (line, p, q).
    auto storageMap = [&](AffineExpr elem) {
      if (rank == 1)
        return AffineMap::get(3, 0, {elem}, ctx);
      if (dim == 1)
        return AffineMap::get(3, 0, {lineDim, elem}, ctx);
      return AffineMap::get(3, 0, {elem, lineDim}, ctx);
    };

    // ---- initial copy: input -> A ------------------------------------ //
    {
      auto lineLoop = rewriter.create<affine::AffineForOp>(loc, 0, lines);
      OpBuilder::InsertionGuard g1(rewriter);
      rewriter.setInsertionPointToStart(lineLoop.getBody());
      auto elemLoop = rewriter.create<affine::AffineForOp>(loc, 0, length);
      rewriter.setInsertionPointToStart(elemLoop.getBody());
      // Reuse the (line, p, q) operand convention with q == 0.
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      SmallVector<Value> operands{lineLoop.getInductionVar(),
                                  elemLoop.getInductionVar(), zero};
      AffineMap map = storageMap(pDim);
      for (auto [src, dst] :
           {std::pair(inRe, bufARe), std::pair(inIm, bufAIm)}) {
        Value v = rewriter.create<affine::AffineLoadOp>(loc, src, map,
                                                        operands);
        rewriter.create<affine::AffineStoreOp>(loc, v, dst, map, operands);
      }
    }

    // ---- statically unrolled Stockham stages -------------------------- //
    Value curRe = bufARe, curIm = bufAIm;
    Value nxtRe = bufBRe, nxtIm = bufBIm;
    for (int t = 0; t < stages; ++t) {
      int64_t nCur = length >> t;
      int64_t m = nCur / 2;
      int64_t s = int64_t{1} << t;
      int64_t twiddleOffset = length - (length >> t);

      auto lineLoop = rewriter.create<affine::AffineForOp>(loc, 0, lines);
      OpBuilder::InsertionGuard g1(rewriter);
      rewriter.setInsertionPointToStart(lineLoop.getBody());
      auto pLoop = rewriter.create<affine::AffineForOp>(loc, 0, m);
      rewriter.setInsertionPointToStart(pLoop.getBody());
      auto qLoop = rewriter.create<affine::AffineForOp>(loc, 0, s);
      rewriter.setInsertionPointToStart(qLoop.getBody());

      SmallVector<Value> operands{lineLoop.getInductionVar(),
                                  pLoop.getInductionVar(),
                                  qLoop.getInductionVar()};

      AffineMap mapA = storageMap(qDim + pDim * s);
      AffineMap mapB = storageMap(qDim + (pDim + m) * s);
      AffineMap mapOutEven = storageMap(qDim + pDim * 2 * s);
      AffineMap mapOutOdd = storageMap(qDim + (pDim * 2 + 1) * s);
      AffineMap twiddleMap = AffineMap::get(3, 0, {pDim + twiddleOffset},
                                            ctx);

      auto load = [&](Value buf, AffineMap map) -> Value {
        return rewriter.create<affine::AffineLoadOp>(loc, buf, map,
                                                     operands);
      };
      auto store = [&](Value v, Value buf, AffineMap map) {
        rewriter.create<affine::AffineStoreOp>(loc, v, buf, map, operands);
      };

      Value aRe = load(curRe, mapA), aIm = load(curIm, mapA);
      Value bRe = load(curRe, mapB), bIm = load(curIm, mapB);
      Value c = load(cosBuf, twiddleMap), sn = load(sinBuf, twiddleMap);

      store(rewriter.create<arith::AddFOp>(loc, aRe, bRe), nxtRe,
            mapOutEven);
      store(rewriter.create<arith::AddFOp>(loc, aIm, bIm), nxtIm,
            mapOutEven);

      Value dRe = rewriter.create<arith::SubFOp>(loc, aRe, bRe);
      Value dIm = rewriter.create<arith::SubFOp>(loc, aIm, bIm);
      // forward: w = c - i s -> (dRe c + dIm s) + i (dIm c - dRe s)
      // inverse: w = c + i s -> (dRe c - dIm s) + i (dIm c + dRe s)
      Value reC = rewriter.create<arith::MulFOp>(loc, dRe, c);
      Value reS = rewriter.create<arith::MulFOp>(loc, dIm, sn);
      Value imC = rewriter.create<arith::MulFOp>(loc, dIm, c);
      Value imS = rewriter.create<arith::MulFOp>(loc, dRe, sn);
      Value outRe, outIm;
      if (!inverse) {
        outRe = rewriter.create<arith::AddFOp>(loc, reC, reS);
        outIm = rewriter.create<arith::SubFOp>(loc, imC, imS);
      } else {
        outRe = rewriter.create<arith::SubFOp>(loc, reC, reS);
        outIm = rewriter.create<arith::AddFOp>(loc, imC, imS);
      }
      store(outRe, nxtRe, mapOutOdd);
      store(outIm, nxtIm, mapOutOdd);

      std::swap(curRe, nxtRe);
      std::swap(curIm, nxtIm);
    }

    // ---- final copy (with 1/L scale for the inverse transform) -------- //
    Value outBufRe = rewriter.create<memref::AllocOp>(loc, bufferType);
    Value outBufIm = rewriter.create<memref::AllocOp>(loc, bufferType);
    {
      auto lineLoop = rewriter.create<affine::AffineForOp>(loc, 0, lines);
      OpBuilder::InsertionGuard g1(rewriter);
      rewriter.setInsertionPointToStart(lineLoop.getBody());
      auto elemLoop = rewriter.create<affine::AffineForOp>(loc, 0, length);
      rewriter.setInsertionPointToStart(elemLoop.getBody());
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      SmallVector<Value> operands{lineLoop.getInductionVar(),
                                  elemLoop.getInductionVar(), zero};
      AffineMap map = storageMap(pDim);
      Value scale;
      if (inverse)
        scale = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(elemType,
                                       1.0 / static_cast<double>(length)));
      for (auto [src, dst] :
           {std::pair(curRe, outBufRe), std::pair(curIm, outBufIm)}) {
        Value v =
            rewriter.create<affine::AffineLoadOp>(loc, src, map, operands);
        if (inverse)
          v = rewriter.create<arith::MulFOp>(loc, v, scale);
        rewriter.create<affine::AffineStoreOp>(loc, v, dst, map, operands);
      }
    }

    // Scratch buffers are not deallocated here: the CPU validation path
    // runs the ownership-based deallocation pipeline (which rejects
    // pre-existing deallocs), and HLS flows map allocations to on-chip
    // arrays with no free operation.
    Value resRe = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, outBufRe, /*restrict=*/true, /*writable=*/true);
    Value resIm = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, outBufIm, /*restrict=*/true, /*writable=*/true);
    rewriter.replaceOp(op, {resRe, resIm});
    return success();
  }
};

struct ConvertSARFFTToAffinePass
    : sar::impl::ConvertSARFFTToAffineBase<ConvertSARFFTToAffinePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addIllegalOp<FFTSplitOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    patterns.add<FFTSplitToAffinePattern>(context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
