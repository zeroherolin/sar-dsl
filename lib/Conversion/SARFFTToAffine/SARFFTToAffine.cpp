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
// Twiddle factors are precomputed into constant memref globals; X and Y are
// one scratch line per stage, which keeps the unrolled stages a chain rather
// than a cycle -- see emitStockham.
//
// Sizes that are not powers of two go through Bluestein's chirp-z
// reduction, which expresses the DFT as a convolution that a padded
// power-of-two Stockham transform can carry:
//
//   X[k] = w[k] * sum_j (x[j] w[j]) b[k - j],
//   w[j] = exp(-i pi j^2 / n),  b[d] = exp(+i pi d^2 / n)
//
// evaluated as a circular convolution of length M = 2^ceil(log2(2n-1)).
// Because n is a compile-time constant, the chirp and the spectrum
// B = FFT_M(b) are both folded on the host and emitted as constant globals,
// so the device-side work is two power-of-two Stockham transforms plus
// three element-wise passes -- all affine, all HLS-friendly.
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
#include <complex>
#include <vector>

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_CONVERTSARFFTTOAFFINE
#include "sar/Conversion/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;

namespace {

/// Maps an element index expression onto the storage indices of a buffer.
/// Map operands are always (line, p, q).
using StorageMapFn = llvm::function_ref<AffineMap(AffineExpr)>;

/// Ensures a private constant memref global named `name` exists and returns
/// its name.
static std::string ensureConstantGlobal(PatternRewriter &rewriter,
                                        ModuleOp module, StringRef name,
                                        FloatType elemType,
                                        ArrayRef<double> values) {
  if (module.lookupSymbol(name))
    return name.str();

  int64_t length = static_cast<int64_t>(values.size());
  auto memrefType = MemRefType::get({length}, elemType);
  auto tensorType = RankedTensorType::get({length}, elemType);
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (double v : values)
    attrs.push_back(rewriter.getFloatAttr(elemType, v));
  auto initial = DenseElementsAttr::get(tensorType, attrs);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  memref::GlobalOp::create(rewriter, module.getLoc(), name,
                           rewriter.getStringAttr("private"), memrefType,
                           initial, /*constant=*/true, /*alignment=*/nullptr);
  return name.str();
}

/// The suffix distinguishing globals of different element precision.
static StringRef typeSuffix(FloatType elemType) {
  return elemType.isF32() ? "f32" : "f64";
}

/// Creates (or reuses) the flattened per-stage Stockham twiddle tables for a
/// transform of size `length`, returning the loaded cos/sin buffers.
static std::pair<Value, Value>
materializeTwiddles(PatternRewriter &rewriter, Location loc, ModuleOp module,
                    int64_t length, FloatType elemType) {
  // Stage t occupies [L - (L >> t), ...) with (L >> (t+1)) entries of
  // cos/sin(2 pi p / n_cur).
  SmallVector<double> cosTable, sinTable;
  cosTable.reserve(length - 1);
  sinTable.reserve(length - 1);
  for (int t = 0, stages = llvm::Log2_64(length); t < stages; ++t) {
    int64_t nCur = length >> t;
    for (int64_t p = 0; p < nCur / 2; ++p) {
      double angle =
          2.0 * M_PI * static_cast<double>(p) / static_cast<double>(nCur);
      cosTable.push_back(std::cos(angle));
      sinTable.push_back(std::sin(angle));
    }
  }
  std::string suffix = (Twine(length) + "_" + typeSuffix(elemType)).str();
  std::string cosName = ensureConstantGlobal(
      rewriter, module, ("__sar_fft_twiddle_cos_" + suffix), elemType,
      cosTable);
  std::string sinName = ensureConstantGlobal(
      rewriter, module, ("__sar_fft_twiddle_sin_" + suffix), elemType,
      sinTable);
  auto twiddleType = MemRefType::get({length - 1}, elemType);
  return {memref::GetGlobalOp::create(rewriter, loc, twiddleType, cosName),
          memref::GetGlobalOp::create(rewriter, loc, twiddleType, sinName)};
}

/// Emits the statically unrolled Stockham stages for a transform of size
/// `length`, for the line selected by `lineIV`.
///
/// The caller owns the line loop and passes one scratch line per intermediate
/// stage, so the working set is O(length * log2(length)) rather than
/// O(lines * length): a 16384-point transform needs a few megabytes of
/// scratch whatever the scene height, which keeps it on chip.
///
/// Each stage reads the previous stage's buffer and writes its own. Reusing
/// two buffers in ping-pong would be smaller, but it makes the stages a cycle
/// in the dataflow graph -- the backend then cannot order them and falls back
/// to running the whole transform sequentially. Distinct buffers keep the
/// stages a chain, which is what lets a dataflow backend overlap one line's
/// late stages with the next line's early ones.
///
/// Stage 0 reads `srcRe`/`srcIm` and the last stage writes `dstRe`/`dstIm`
/// (scaled by `scale` when set), so the transform costs exactly its
/// log2(length) butterfly passes -- no copy in, no copy out.
static void emitStockham(PatternRewriter &rewriter, Location loc,
                         MLIRContext *ctx, int64_t length, Value lineIV,
                         Value srcRe, Value srcIm,
                         ArrayRef<std::pair<Value, Value>> scratch, Value dstRe,
                         Value dstIm, Value cosBuf, Value sinBuf, bool inverse,
                         Value scale, StorageMapFn srcDstMap) {
  auto pDim = getAffineDimExpr(1, ctx);
  auto qDim = getAffineDimExpr(2, ctx);
  int stages = llvm::Log2_64(length);
  assert(scratch.size() >= static_cast<size_t>(stages - 1) &&
         "one scratch line per intermediate stage");

  // Scratch is a single line, so it is indexed by the element expression
  // alone; the source and destination keep the caller's 2-D layout.
  auto scratchMap = [&](AffineExpr elem) {
    return AffineMap::get(3, 0, {elem}, ctx);
  };

  Value curRe = srcRe, curIm = srcIm;
  bool curIsScratch = false;

  for (int t = 0; t < stages; ++t) {
    Value nxtRe, nxtIm;
    bool nxtIsScratch = t != stages - 1;
    if (nxtIsScratch) {
      nxtRe = scratch[t].first;
      nxtIm = scratch[t].second;
    } else {
      nxtRe = dstRe;
      nxtIm = dstIm;
    }

    int64_t nCur = length >> t;
    int64_t m = nCur / 2;
    int64_t s = int64_t{1} << t;
    int64_t twiddleOffset = length - (length >> t);

    OpBuilder::InsertionGuard g1(rewriter);
    auto pLoop = affine::AffineForOp::create(rewriter, loc, 0, m);
    rewriter.setInsertionPointToStart(pLoop.getBody());
    auto qLoop = affine::AffineForOp::create(rewriter, loc, 0, s);
    rewriter.setInsertionPointToStart(qLoop.getBody());

    SmallVector<Value> operands{lineIV, pLoop.getInductionVar(),
                                qLoop.getInductionVar()};

    auto srcMap = curIsScratch ? StorageMapFn(scratchMap) : srcDstMap;
    auto dstMap = nxtIsScratch ? StorageMapFn(scratchMap) : srcDstMap;
    AffineMap mapA = srcMap(qDim + pDim * s);
    AffineMap mapB = srcMap(qDim + (pDim + m) * s);
    AffineMap mapOutEven = dstMap(qDim + pDim * 2 * s);
    AffineMap mapOutOdd = dstMap(qDim + (pDim * 2 + 1) * s);
    AffineMap twiddleMap = AffineMap::get(3, 0, {pDim + twiddleOffset}, ctx);

    auto load = [&](Value buf, AffineMap map) -> Value {
      return affine::AffineLoadOp::create(rewriter, loc, buf, map, operands);
    };
    // The 1/L of an inverse transform rides along on the final stage's
    // stores, so no extra pass over the data is needed.
    bool scaleHere = scale && t == stages - 1;
    auto store = [&](Value v, Value buf, AffineMap map) {
      if (scaleHere)
        v = arith::MulFOp::create(rewriter, loc, v, scale);
      affine::AffineStoreOp::create(rewriter, loc, v, buf, map, operands);
    };

    Value aRe = load(curRe, mapA), aIm = load(curIm, mapA);
    Value bRe = load(curRe, mapB), bIm = load(curIm, mapB);
    Value c = load(cosBuf, twiddleMap), sn = load(sinBuf, twiddleMap);

    store(arith::AddFOp::create(rewriter, loc, aRe, bRe), nxtRe, mapOutEven);
    store(arith::AddFOp::create(rewriter, loc, aIm, bIm), nxtIm, mapOutEven);

    Value dRe = arith::SubFOp::create(rewriter, loc, aRe, bRe);
    Value dIm = arith::SubFOp::create(rewriter, loc, aIm, bIm);
    // forward: w = c - i s -> (dRe c + dIm s) + i (dIm c - dRe s)
    // inverse: w = c + i s -> (dRe c - dIm s) + i (dIm c + dRe s)
    Value reC = arith::MulFOp::create(rewriter, loc, dRe, c);
    Value reS = arith::MulFOp::create(rewriter, loc, dIm, sn);
    Value imC = arith::MulFOp::create(rewriter, loc, dIm, c);
    Value imS = arith::MulFOp::create(rewriter, loc, dRe, sn);
    Value outRe, outIm;
    if (!inverse) {
      outRe = arith::AddFOp::create(rewriter, loc, reC, reS);
      outIm = arith::SubFOp::create(rewriter, loc, imC, imS);
    } else {
      outRe = arith::SubFOp::create(rewriter, loc, reC, reS);
      outIm = arith::AddFOp::create(rewriter, loc, imC, imS);
    }
    store(outRe, nxtRe, mapOutOdd);
    store(outIm, nxtIm, mapOutOdd);

    curRe = nxtRe;
    curIm = nxtIm;
    curIsScratch = nxtIsScratch;
  }
}

/// In-place radix-2 forward DFT used to fold the Bluestein kernel spectrum
/// at compile time (host side, double precision).
static void hostFFT(std::vector<std::complex<double>> &data) {
  size_t n = data.size();
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(data[i], data[j]);
  }
  for (size_t len = 2; len <= n; len <<= 1) {
    double angle = -2.0 * M_PI / static_cast<double>(len);
    std::complex<double> step(std::cos(angle), std::sin(angle));
    for (size_t i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (size_t k = 0; k < len / 2; ++k) {
        std::complex<double> u = data[i + k];
        std::complex<double> v = data[i + k + len / 2] * w;
        data[i + k] = u + v;
        data[i + k + len / 2] = u - v;
        w *= step;
      }
    }
  }
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

    auto lineDim = getAffineDimExpr(0, ctx);

    // Storage map for an element index expression; operands are
    // (line, p, q). The same map serves the padded Bluestein buffers: it
    // encodes the layout, not the extents.
    auto storageMap = [&](AffineExpr elem) {
      if (rank == 1)
        return AffineMap::get(3, 0, {elem}, ctx);
      if (dim == 1)
        return AffineMap::get(3, 0, {lineDim, elem}, ctx);
      return AffineMap::get(3, 0, {elem, lineDim}, ctx);
    };

    auto bufferType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());

    Value inRe = bufferization::ToBufferOp::create(rewriter, loc, bufferType,
                                                   op.getRe());
    Value inIm = bufferization::ToBufferOp::create(rewriter, loc, bufferType,
                                                   op.getIm());

    Value outBufRe = memref::AllocOp::create(rewriter, loc, bufferType);
    Value outBufIm = memref::AllocOp::create(rewriter, loc, bufferType);

    if (llvm::isPowerOf2_64(length))
      emitPowerOfTwo(rewriter, loc, ctx, module, length, lines, elemType,
                     bufferType, inRe, inIm, outBufRe, outBufIm, inverse,
                     storageMap);
    else
      emitBluestein(rewriter, loc, ctx, module, length, dim, lines, elemType,
                    tensorType, inRe, inIm, outBufRe, outBufIm, inverse,
                    storageMap);

    // Scratch buffers are not deallocated here: the CPU validation path
    // runs the ownership-based deallocation pipeline (which rejects
    // pre-existing deallocs), and HLS flows map allocations to on-chip
    // arrays with no free operation.
    Value resRe =
        bufferization::ToTensorOp::create(rewriter, loc, tensorType, outBufRe,
                                          /*restrict=*/true, /*writable=*/true);
    Value resIm =
        bufferization::ToTensorOp::create(rewriter, loc, tensorType, outBufIm,
                                          /*restrict=*/true, /*writable=*/true);
    rewriter.replaceOp(op, {resRe, resIm});
    return success();
  }

private:
  /// Direct Stockham transform.
  ///
  /// The stages read the input in place of a copy-in pass and write the
  /// result in place of a copy-out pass, so the whole transform is exactly
  /// its log2(L) butterfly passes.
  static void emitPowerOfTwo(PatternRewriter &rewriter, Location loc,
                             MLIRContext *ctx, ModuleOp module, int64_t length,
                             int64_t lines, FloatType elemType,
                             MemRefType bufferType, Value inRe, Value inIm,
                             Value outBufRe, Value outBufIm, bool inverse,
                             StorageMapFn storageMap) {
    auto [cosBuf, sinBuf] =
        materializeTwiddles(rewriter, loc, module, length, elemType);

    Value scale;
    if (inverse)
      scale = arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getFloatAttr(elemType, 1.0 / static_cast<double>(length)));

    // Stage 0 reads the input and the last stage writes the result, so the
    // stages in between each need one scratch line. That is all the working
    // set the transform has, whatever the scene height.
    int stages = llvm::Log2_64(length);
    auto lineType = MemRefType::get({length}, elemType);
    SmallVector<std::pair<Value, Value>> scratch;
    for (int t = 0; t + 1 < stages; ++t)
      scratch.emplace_back(
          memref::AllocOp::create(rewriter, loc, lineType).getResult(),
          memref::AllocOp::create(rewriter, loc, lineType).getResult());

    // Lines are independent, so the line loop wraps the whole transform and
    // is what a dataflow backend pipelines over.
    auto lineLoop = affine::AffineForOp::create(rewriter, loc, 0, lines);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(lineLoop.getBody());
    emitStockham(rewriter, loc, ctx, length, lineLoop.getInductionVar(), inRe,
                 inIm, scratch, outBufRe, outBufIm, cosBuf, sinBuf, inverse,
                 scale, storageMap);
  }

  /// Bluestein's chirp-z reduction for non-power-of-two sizes.
  ///
  /// The inverse transform reuses the forward path through the conjugate
  /// identity ifft(x) = conj(fft(conj(x))) / n, so only one chirp table and
  /// one kernel spectrum are needed per size.
  static void emitBluestein(PatternRewriter &rewriter, Location loc,
                            MLIRContext *ctx, ModuleOp module, int64_t length,
                            int64_t dim, int64_t lines, FloatType elemType,
                            RankedTensorType tensorType, Value inRe, Value inIm,
                            Value outBufRe, Value outBufIm, bool inverse,
                            StorageMapFn storageMap) {
    auto pDim = getAffineDimExpr(1, ctx);

    // Bluestein convolves two length-n sequences, so the circular
    // convolution needs at least 2n-1 points to avoid wrap-around.
    int64_t padded = llvm::NextPowerOf2(2 * length - 1);

    // ---- host-folded constants ---------------------------------------- //
    // chirp w[j] = exp(-i pi j^2 / n). exp has period 2 pi i, so j^2 can be
    // reduced modulo 2n first; folding j before squaring keeps the product
    // from overflowing for large transforms.
    SmallVector<double> chirpCos, chirpSin;
    chirpCos.reserve(length);
    chirpSin.reserve(length);
    for (int64_t j = 0; j < length; ++j) {
      int64_t sq = ((j % (2 * length)) * j) % (2 * length);
      double angle =
          -M_PI * static_cast<double>(sq) / static_cast<double>(length);
      chirpCos.push_back(std::cos(angle));
      chirpSin.push_back(std::sin(angle));
    }

    // b[d] = exp(+i pi d^2 / n) laid out circularly over the padded length,
    // then transformed once on the host: B = FFT_M(b).
    std::vector<std::complex<double>> kernel(padded, {0.0, 0.0});
    for (int64_t j = 0; j < length; ++j) {
      std::complex<double> v(chirpCos[j], -chirpSin[j]); // conj(w[j])
      kernel[j] = v;
      if (j > 0)
        kernel[padded - j] = v;
    }
    hostFFT(kernel);
    SmallVector<double> kernelRe, kernelIm;
    kernelRe.reserve(padded);
    kernelIm.reserve(padded);
    for (const auto &v : kernel) {
      kernelRe.push_back(v.real());
      kernelIm.push_back(v.imag());
    }

    std::string suffix = (Twine(length) + "_" + typeSuffix(elemType)).str();
    auto lineType = MemRefType::get({length}, elemType);
    auto kernelType = MemRefType::get({padded}, elemType);
    Value chirpCosBuf = memref::GetGlobalOp::create(
        rewriter, loc, lineType,
        ensureConstantGlobal(rewriter, module,
                             ("__sar_bluestein_chirp_cos_" + suffix), elemType,
                             chirpCos));
    Value chirpSinBuf = memref::GetGlobalOp::create(
        rewriter, loc, lineType,
        ensureConstantGlobal(rewriter, module,
                             ("__sar_bluestein_chirp_sin_" + suffix), elemType,
                             chirpSin));
    Value kernelReBuf = memref::GetGlobalOp::create(
        rewriter, loc, kernelType,
        ensureConstantGlobal(rewriter, module,
                             ("__sar_bluestein_kernel_re_" + suffix), elemType,
                             kernelRe));
    Value kernelImBuf = memref::GetGlobalOp::create(
        rewriter, loc, kernelType,
        ensureConstantGlobal(rewriter, module,
                             ("__sar_bluestein_kernel_im_" + suffix), elemType,
                             kernelIm));

    auto [cosBuf, sinBuf] =
        materializeTwiddles(rewriter, loc, module, padded, elemType);

    // Every buffer is one padded line wide: the working set is O(M), not
    // O(lines * M), so it stays on chip whatever the scene height.
    auto scratchType = MemRefType::get({padded}, elemType);
    auto alloc = [&] {
      return memref::AllocOp::create(rewriter, loc, scratchType).getResult();
    };
    Value stageRe = alloc(), stageIm = alloc();
    Value specRe = alloc(), specIm = alloc();
    Value prodRe = alloc(), prodIm = alloc();
    Value convRe = alloc(), convIm = alloc();

    // The two transforms get their own stage buffers. Sharing them would put
    // both on the same scratch and turn the chain into a cycle, which is the
    // one shape a dataflow backend cannot schedule.
    int paddedStages = llvm::Log2_64(padded);
    SmallVector<std::pair<Value, Value>> fwdScratch, invScratch;
    for (int t = 0; t + 1 < paddedStages; ++t) {
      fwdScratch.emplace_back(alloc(), alloc());
      invScratch.emplace_back(alloc(), alloc());
    }

    AffineMap ioMap = storageMap(pDim);
    AffineMap lineMap = AffineMap::get(3, 0, {pDim}, ctx);
    auto scratchMap = [&](AffineExpr elem) {
      return AffineMap::get(3, 0, {elem}, ctx);
    };

    // Lines are independent; one loop carries the whole chirp-z reduction.
    auto lineLoop = affine::AffineForOp::create(rewriter, loc, 0, lines);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(lineLoop.getBody());
    Value lineIV = lineLoop.getInductionVar();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

    auto elementLoop = [&](int64_t lo, int64_t hi, auto body) {
      OpBuilder::InsertionGuard g(rewriter);
      auto loop = affine::AffineForOp::create(rewriter, loc, lo, hi);
      rewriter.setInsertionPointToStart(loop.getBody());
      SmallVector<Value> operands{lineIV, loop.getInductionVar(), zero};
      body(operands);
    };
    auto load = [&](Value buf, AffineMap map, ValueRange ops) -> Value {
      return affine::AffineLoadOp::create(rewriter, loc, buf, map, ops);
    };
    auto store = [&](Value v, Value buf, AffineMap map, ValueRange ops) {
      affine::AffineStoreOp::create(rewriter, loc, v, buf, map, ops);
    };
    auto mul = [&](Value a, Value b) {
      return arith::MulFOp::create(rewriter, loc, a, b).getResult();
    };

    // ---- pre-multiply by the chirp, zero-padded to M ------------------ //
    elementLoop(0, length, [&](ValueRange ops) {
      Value xr = load(inRe, ioMap, ops);
      Value xi = load(inIm, ioMap, ops);
      if (inverse) // conj(x)
        xi = arith::NegFOp::create(rewriter, loc, xi);
      Value wr = load(chirpCosBuf, lineMap, ops);
      Value wi = load(chirpSinBuf, lineMap, ops);
      store(arith::SubFOp::create(rewriter, loc, mul(xr, wr), mul(xi, wi)),
            stageRe, lineMap, ops);
      store(arith::AddFOp::create(rewriter, loc, mul(xr, wi), mul(xi, wr)),
            stageIm, lineMap, ops);
    });
    elementLoop(length, padded, [&](ValueRange ops) {
      Value fzero = arith::ConstantOp::create(
          rewriter, loc, rewriter.getFloatAttr(elemType, 0.0));
      store(fzero, stageRe, lineMap, ops);
      store(fzero, stageIm, lineMap, ops);
    });

    // ---- forward transform, pointwise product, inverse transform ------ //
    emitStockham(rewriter, loc, ctx, padded, lineIV, stageRe, stageIm,
                 fwdScratch, specRe, specIm, cosBuf, sinBuf,
                 /*inverse=*/false, /*scale=*/nullptr, scratchMap);

    elementLoop(0, padded, [&](ValueRange ops) {
      Value ar = load(specRe, lineMap, ops), ai = load(specIm, lineMap, ops);
      Value br = load(kernelReBuf, lineMap, ops);
      Value bi = load(kernelImBuf, lineMap, ops);
      store(arith::SubFOp::create(rewriter, loc, mul(ar, br), mul(ai, bi)),
            prodRe, lineMap, ops);
      store(arith::AddFOp::create(rewriter, loc, mul(ar, bi), mul(ai, br)),
            prodIm, lineMap, ops);
    });

    emitStockham(rewriter, loc, ctx, padded, lineIV, prodRe, prodIm, invScratch,
                 convRe, convIm, cosBuf, sinBuf,
                 /*inverse=*/true, /*scale=*/nullptr, scratchMap);

    // ---- post-multiply by the chirp, cropping back to n --------------- //
    // `emitStockham` emits butterflies only, so the 1/M the convolution
    // theorem needs is applied here, together with the 1/n of an inverse
    // transform.
    double post = 1.0 / static_cast<double>(padded);
    if (inverse)
      post /= static_cast<double>(length);
    Value postScale = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elemType, post));
    elementLoop(0, length, [&](ValueRange ops) {
      Value cr = load(convRe, lineMap, ops), ci = load(convIm, lineMap, ops);
      Value wr = load(chirpCosBuf, lineMap, ops);
      Value wi = load(chirpSinBuf, lineMap, ops);
      Value re = arith::SubFOp::create(rewriter, loc, mul(cr, wr), mul(ci, wi));
      Value im = arith::AddFOp::create(rewriter, loc, mul(cr, wi), mul(ci, wr));
      if (inverse) // conj of the forward result
        im = arith::NegFOp::create(rewriter, loc, im);
      store(mul(re, postScale), outBufRe, ioMap, ops);
      store(mul(im, postScale), outBufIm, ioMap, ops);
    });
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
