//===- SARFFTToAffine.cpp - Stockham FFT as affine loops ------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
// Lowers sar.fft_split (the split-complex FFT produced by sar-decomplexify)
// into affine loop nests implementing a mixed radix-4/2 Stockham autosort FFT.
// Stockham is chosen over Cooley-Tukey deliberately: it needs no
// bit-reversal permutation, so every memory access is an affine function of
// the loop indices -- exactly what an HLS flow requires for
// loop analysis, pipelining and array partitioning.
//
// Radix-4 stages consume exponent pairs; an odd exponent starts with one
// radix-2 stage. Both use the same DIF Stockham layout:
//
//   for p in [0, n_cur/radix), q in [0, s):
//     DFT_radix(X[q + s*(p + j*m)]) -> Y[q + s*(radix*p + k)]
//     Y[k>0] *= exp(-+ 2 pi i p*k / n_cur)
//
// The generated structure is shaped for an HLS backend end to end. Lines are
// processed in blocks of `fft-parallel-rows` lanes: a prefetch sweep copies
// the block from the source planes into on-chip line buffers with unit-stride
// external accesses (`fft-io-unroll` elements per beat, so the bus can widen
// and burst), the butterfly stages run entirely on the line buffers with the
// lane loop innermost (twiddle factors are fetched once per butterfly and
// shared across lanes), and a mirrored write-back sweep stores the block.
// Buffers carry banking hints (`hls.partition_*`) matched to the butterfly
// and transfer strides; the CPU validation pipeline ignores them.
//
// Twiddle factors are precomputed into constant memref globals; the stages
// between prefetch and write-back draw scratch lines from a pool, which keeps
// the unrolled stages a chain rather than a cycle -- see emitStockham and
// scratchSlots.
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
// three element-wise passes -- all affine, all HLS-friendly. The chirp-z
// path stays line-serial: its working set is already on chip and the
// non-power-of-two sizes it serves are not the production rasters.
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
#include "sar/Support/HLSHints.h"

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

/// A pair of float-plane buffers holding the real and imaginary halves.
struct PlanePair {
  Value re, im;
};

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

/// Mixed-radix schedule for a power-of-two transform. One radix-2 stage is
/// used only when the exponent is odd; all remaining pairs are radix-4.
static SmallVector<int> radixSchedule(int64_t length) {
  int exponent = llvm::Log2_64(length);
  SmallVector<int> radices;
  if (exponent & 1)
    radices.push_back(2);
  radices.append(exponent / 2, 4);
  return radices;
}

/// Attaches the banking hint the HLS partition pass applies verbatim.
static void setPartitionHint(OpBuilder &builder, Operation *op,
                             ArrayRef<StringRef> kinds,
                             ArrayRef<int64_t> factors) {
  op->setAttr(kPartitionKindsAttr, builder.getStrArrayAttr(kinds));
  op->setAttr(kPartitionFactorsAttr, builder.getI64ArrayAttr(factors));
}

/// Creates (or reuses) the flattened per-stage Stockham twiddle tables for a
/// transform of size `length`, returning the loaded cos/sin buffers.
///
/// A radix-4 butterfly reads the three adjacent entries p*3 .. p*3+2 in one
/// iteration, so the tables are padded to a multiple of three and hinted
/// cyclic-3: each read lands in its own bank. The padding is never read.
static std::pair<Value, Value>
materializeTwiddles(PatternRewriter &rewriter, Location loc, ModuleOp module,
                    int64_t length, FloatType elemType) {
  // Stage t occupies [L - (L >> t), ...) with (L >> (t+1)) entries of
  // cos/sin(2 pi p / n_cur).
  SmallVector<double> cosTable, sinTable;
  cosTable.reserve(length + 2);
  sinTable.reserve(length + 2);
  int64_t span = 1;
  for (int radix : radixSchedule(length)) {
    int64_t nCur = length / span;
    int64_t butterflies = nCur / radix;
    for (int64_t p = 0; p < butterflies; ++p)
      for (int output = 1; output < radix; ++output) {
        double angle = 2.0 * M_PI * static_cast<double>(p * output) /
                       static_cast<double>(nCur);
        cosTable.push_back(std::cos(angle));
        sinTable.push_back(std::sin(angle));
      }
    span *= radix;
  }
  assert(cosTable.size() == static_cast<size_t>(length - 1));
  while (cosTable.size() % 3 != 0) {
    cosTable.push_back(0.0);
    sinTable.push_back(0.0);
  }
  std::string suffix = (Twine(length) + "_" + typeSuffix(elemType)).str();
  std::string cosName = ensureConstantGlobal(
      rewriter, module, ("__sar_fft_twiddle_cos_" + suffix), elemType,
      cosTable);
  std::string sinName = ensureConstantGlobal(
      rewriter, module, ("__sar_fft_twiddle_sin_" + suffix), elemType,
      sinTable);
  auto twiddleType =
      MemRefType::get({static_cast<int64_t>(cosTable.size())}, elemType);
  auto cosBuf =
      memref::GetGlobalOp::create(rewriter, loc, twiddleType, cosName);
  auto sinBuf =
      memref::GetGlobalOp::create(rewriter, loc, twiddleType, sinName);
  // Banking pays a whole memory primitive per bank, so only tables large
  // enough to span several primitives are worth splitting.
  if (cosTable.size() % 3 == 0 && length >= 256) {
    setPartitionHint(rewriter, cosBuf, {"cyclic"}, {3});
    setPartitionHint(rewriter, sinBuf, {"cyclic"}, {3});
  }
  return {cosBuf, sinBuf};
}

/// Returns the number of distinct scratch-buffer slots required for a
/// `stages`-stage transform when stages are grouped into groups of
/// `stageGroup` (0 = full unroll = one slot per intermediate stage).
///
/// With full unroll (stageGroup == 0), every intermediate stage writes its own
/// buffer, giving stages-1 slots in total.
///
/// With stageGroup == k, stages are packed into ceil(stages/k) groups; the
/// first stage reads the prefetched block and the last writes the write-back
/// block, so the groups in between need ceil(stages/k) - 1 slots. Two is the
/// floor once more than one stage writes scratch: a Stockham butterfly reads
/// X[q + sp], X[q + s(p + m)] and writes Y[q + 2sp], Y[q + s(2p + 1)], so a
/// stage whose source and destination were the same line would overwrite
/// values a later iteration still has to read.
static int scratchSlots(int stages, unsigned stageGroup) {
  int intermediates = stages - 1;
  if (intermediates <= 0)
    return 0;
  if (stageGroup == 0)
    return intermediates; // full unroll
  int groups = (stages + static_cast<int>(stageGroup) - 1) /
               static_cast<int>(stageGroup);
  int slots = groups - 1;
  if (slots < 2)
    slots = intermediates < 2 ? intermediates : 2;
  return slots > intermediates ? intermediates : slots;
}

/// Allocates one line-block buffer: `lanes` lines of `length` elements
/// (rank-1 when `lanes` is one). A parallel block is hinted complete over
/// the lane dimension -- each unrolled lane owns its banks, which no local
/// access analysis can recover from the compact lane loop. `elemFactor`
/// banks the element dimension cyclically for the io-unrolled transfer
/// sweeps; only the prefetch and write-back blocks need it, so the
/// butterfly scratch passes zero and keeps one primitive per lane.
static Value allocLineBuffer(PatternRewriter &rewriter, Location loc,
                             int64_t lanes, int64_t length, FloatType elemType,
                             int64_t elemFactor = 0) {
  auto type = lanes > 1 ? MemRefType::get({lanes, length}, elemType)
                        : MemRefType::get({length}, elemType);
  auto alloc = memref::AllocOp::create(rewriter, loc, type);
  if (lanes > 1 && elemFactor > 1)
    setPartitionHint(rewriter, alloc, {"complete", "cyclic"},
                     {lanes, elemFactor});
  else if (lanes > 1)
    setPartitionHint(rewriter, alloc, {"complete", "none"}, {lanes, 1});
  return alloc;
}

/// Emits the statically unrolled Stockham stages for one block of `lanes`
/// lines, from the prefetched `src` block through the scratch pool into the
/// `dst` block. All buffers are lane-major line blocks; nothing here touches
/// the transform's external planes.
///
/// Each stage is a (p, q) butterfly nest with the lane loop innermost. The
/// lane loop carries `hls.unroll_factor` instead of being cloned: the
/// emitter prints an unroll directive, so the generated source stays one
/// engine description however many lanes run. Twiddle factors are loaded
/// once per butterfly, above the lane loop, and shared by every lane.
///
/// Each group reads the previous group's output and writes its own. Reusing
/// two buffers in ping-pong across all stages would be smaller but turns the
/// stage chain into a cycle in the dataflow graph -- the backend cannot order
/// a cycle and falls back to sequential execution. Distinct group buffers keep
/// the transform a chain.
///
/// The 1/L of an inverse transform rides along on the final stage's stores
/// (`scale`), so no extra pass over the data is needed.
static void emitStockham(PatternRewriter &rewriter, Location loc,
                         MLIRContext *ctx, int64_t length, int64_t lanes,
                         PlanePair src, ArrayRef<PlanePair> scratch,
                         PlanePair dst, Value cosBuf, Value sinBuf,
                         bool inverse, Value scale, unsigned stageGroup) {
  auto pDim = getAffineDimExpr(0, ctx);
  auto qDim = getAffineDimExpr(1, ctx);
  auto radices = radixSchedule(length);
  int stages = static_cast<int>(radices.size());
  int slots = scratchSlots(stages, stageGroup);
  assert(scratch.size() >= static_cast<size_t>(slots) &&
         "not enough scratch lines for the chosen stage grouping");

  // Element maps onto the lane-major line blocks. Butterfly operands are
  // (p, q) plus, when lanes run in parallel, the innermost lane.
  unsigned numDims = lanes > 1 ? 3 : 2;
  auto elementMap = [&](AffineExpr elem) {
    if (lanes > 1)
      return AffineMap::get(numDims, 0, {getAffineDimExpr(2, ctx), elem}, ctx);
    return AffineMap::get(numDims, 0, {elem}, ctx);
  };
  // Twiddle loads sit above the lane loop and depend on p alone.
  auto twiddleMap = [&](AffineExpr index) {
    return AffineMap::get(1, 0, {index}, ctx);
  };

  PlanePair cur = src;
  int64_t span = 1;
  int64_t twiddleOffset = 0;
  for (int t = 0; t < stages; ++t) {
    int radix = radices[t];
    // Round-robin over the scratch pool: full unroll makes `t % slots`
    // the identity (one line per stage); a grouped pool sends later
    // stages back onto earlier lines. The pool never shrinks below two
    // (see scratchSlots), so no butterfly is emitted in place.
    PlanePair nxt = t != stages - 1 ? scratch[t % slots] : dst;

    int64_t nCur = length / span;
    int64_t m = nCur / radix;

    OpBuilder::InsertionGuard g1(rewriter);
    auto pLoop = affine::AffineForOp::create(rewriter, loc, 0, m);
    rewriter.setInsertionPointToStart(pLoop.getBody());
    auto qLoop = affine::AffineForOp::create(rewriter, loc, 0, span);
    rewriter.setInsertionPointToStart(qLoop.getBody());

    // One twiddle fetch per butterfly, shared across all lanes.
    SmallVector<Value, 3> twCos, twSin;
    for (int output = 1; output < radix; ++output) {
      AffineExpr twiddleIndex =
          pDim * (radix - 1) + (output - 1) + twiddleOffset;
      twCos.push_back(affine::AffineLoadOp::create(
          rewriter, loc, cosBuf, twiddleMap(twiddleIndex),
          ValueRange{pLoop.getInductionVar()}));
      twSin.push_back(affine::AffineLoadOp::create(
          rewriter, loc, sinBuf, twiddleMap(twiddleIndex),
          ValueRange{pLoop.getInductionVar()}));
    }

    SmallVector<Value> operands{pLoop.getInductionVar(),
                                qLoop.getInductionVar()};
    if (lanes > 1) {
      auto laneLoop = affine::AffineForOp::create(rewriter, loc, 0, lanes);
      laneLoop->setAttr(kUnrollFactorAttr,
                        rewriter.getI64IntegerAttr(lanes));
      rewriter.setInsertionPointToStart(laneLoop.getBody());
      operands.push_back(laneLoop.getInductionVar());
    }

    auto load = [&](Value buf, AffineMap map) -> Value {
      return affine::AffineLoadOp::create(rewriter, loc, buf, map, operands);
    };
    bool scaleHere = scale && t == stages - 1;
    auto store = [&](Value v, Value buf, AffineMap map) {
      if (scaleHere)
        v = arith::MulFOp::create(rewriter, loc, v, scale);
      affine::AffineStoreOp::create(rewriter, loc, v, buf, map, operands);
    };

    auto add = [&](Value lhs, Value rhs) -> Value {
      return arith::AddFOp::create(rewriter, loc, lhs, rhs);
    };
    auto sub = [&](Value lhs, Value rhs) -> Value {
      return arith::SubFOp::create(rewriter, loc, lhs, rhs);
    };
    auto mul = [&](Value lhs, Value rhs) -> Value {
      return arith::MulFOp::create(rewriter, loc, lhs, rhs);
    };

    SmallVector<Value, 4> inputRe, inputIm;
    for (int input = 0; input < radix; ++input) {
      AffineExpr index = qDim + (pDim + input * m) * span;
      inputRe.push_back(load(cur.re, elementMap(index)));
      inputIm.push_back(load(cur.im, elementMap(index)));
    }

    SmallVector<Value, 4> dftRe(radix), dftIm(radix);
    if (radix == 2) {
      dftRe[0] = add(inputRe[0], inputRe[1]);
      dftIm[0] = add(inputIm[0], inputIm[1]);
      dftRe[1] = sub(inputRe[0], inputRe[1]);
      dftIm[1] = sub(inputIm[0], inputIm[1]);
    } else {
      auto sum4 = [&](Value a, Value b, Value c, Value d) {
        return add(add(a, b), add(c, d));
      };
      dftRe[0] = sum4(inputRe[0], inputRe[1], inputRe[2], inputRe[3]);
      dftIm[0] = sum4(inputIm[0], inputIm[1], inputIm[2], inputIm[3]);
      dftRe[2] = add(sub(inputRe[0], inputRe[1]), sub(inputRe[2], inputRe[3]));
      dftIm[2] = add(sub(inputIm[0], inputIm[1]), sub(inputIm[2], inputIm[3]));

      Value forward1Re =
          sub(add(inputRe[0], inputIm[1]), add(inputRe[2], inputIm[3]));
      Value forward1Im =
          add(sub(inputIm[0], inputRe[1]), sub(inputRe[3], inputIm[2]));
      Value forward3Re =
          add(sub(inputRe[0], inputIm[1]), sub(inputIm[3], inputRe[2]));
      Value forward3Im =
          sub(add(inputIm[0], inputRe[1]), add(inputIm[2], inputRe[3]));
      dftRe[1] = inverse ? forward3Re : forward1Re;
      dftIm[1] = inverse ? forward3Im : forward1Im;
      dftRe[3] = inverse ? forward1Re : forward3Re;
      dftIm[3] = inverse ? forward1Im : forward3Im;
    }

    for (int output = 0; output < radix; ++output) {
      Value outRe = dftRe[output], outIm = dftIm[output];
      if (output != 0) {
        Value cosine = twCos[output - 1];
        Value sine = twSin[output - 1];
        Value reC = mul(outRe, cosine), reS = mul(outIm, sine);
        Value imC = mul(outIm, cosine), imS = mul(outRe, sine);
        if (!inverse) {
          outRe = add(reC, reS);
          outIm = sub(imC, imS);
        } else {
          outRe = sub(reC, reS);
          outIm = add(imC, imS);
        }
      }
      AffineExpr outputIndex = qDim + (pDim * radix + output) * span;
      store(outRe, nxt.re, elementMap(outputIndex));
      store(outIm, nxt.im, elementMap(outputIndex));
    }

    cur = nxt;
    twiddleOffset += m * (radix - 1);
    span *= radix;
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
  FFTSplitToAffinePattern(MLIRContext *context, unsigned stageGroup,
                          unsigned parallelRows, unsigned ioUnroll)
      : OpRewritePattern<FFTSplitOp>(context), stageGroup(stageGroup),
        parallelRows(parallelRows), ioUnroll(ioUnroll) {}

  /// How many consecutive Stockham stages share one scratch slot; 0 keeps
  /// the full unroll (one slot per intermediate stage).
  unsigned stageGroup;
  /// Lines transformed in parallel per block (the lane count).
  unsigned parallelRows;
  /// Elements moved per external access in the prefetch/write-back sweeps.
  unsigned ioUnroll;

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
      emitPowerOfTwo(rewriter, loc, ctx, module, length, lines, rank, dim,
                     elemType, {inRe, inIm}, {outBufRe, outBufIm}, inverse,
                     stageGroup, parallelRows, ioUnroll);
    else
      emitBluestein(rewriter, loc, ctx, module, length, dim, lines, elemType,
                    tensorType, inRe, inIm, outBufRe, outBufIm, inverse,
                    storageMap, stageGroup);

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
  /// Emits one transfer sweep between the external planes and a local line
  /// block: the prefetch (`toLocal`) or its write-back mirror.
  ///
  /// The loop order puts whichever index is contiguous in the external
  /// planes innermost, statically unrolled by `io` (`hls.unroll_factor`),
  /// so the bus sees unit-stride runs it can widen and burst:
  ///
  ///   dim == 1 (or rank 1): for lane { for j step io { for w unroll } }
  ///   dim == 0:             for j { for lane unroll }
  ///
  /// where the external element index is contiguous in (j, w) for dim 1 and
  /// in lane for dim 0 (adjacent lines are adjacent columns).
  static void emitTransfer(PatternRewriter &rewriter, Location loc,
                           MLIRContext *ctx, bool toLocal, int64_t rank,
                           int64_t dim, int64_t length, int64_t lanes,
                           int64_t io, Value blockIV, PlanePair ext,
                           PlanePair local) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Dim positions in the composed maps; operands are collected in loop
    // nesting order.
    SmallVector<Value> operands{blockIV};
    AffineExpr blockExpr = getAffineDimExpr(0, ctx);
    AffineExpr laneExpr, elemExpr;
    unsigned numDims = 1;

    auto transferBody = [&]() {
      AffineExpr lineExpr = lanes > 1 ? blockExpr + laneExpr : blockExpr;
      SmallVector<AffineExpr, 2> extExprs;
      if (rank == 1)
        extExprs = {elemExpr};
      else if (dim == 1)
        extExprs = {lineExpr, elemExpr};
      else
        extExprs = {elemExpr, lineExpr};
      auto extMap = AffineMap::get(numDims, 0, extExprs, ctx);
      SmallVector<AffineExpr, 2> localExprs;
      if (lanes > 1)
        localExprs = {laneExpr, elemExpr};
      else
        localExprs = {elemExpr};
      auto localMap = AffineMap::get(numDims, 0, localExprs, ctx);

      for (auto [extBuf, localBuf] :
           {std::pair{ext.re, local.re}, std::pair{ext.im, local.im}}) {
        Value src = toLocal ? extBuf : localBuf;
        auto srcMap = toLocal ? extMap : localMap;
        Value dst = toLocal ? localBuf : extBuf;
        auto dstMap = toLocal ? localMap : extMap;
        Value v =
            affine::AffineLoadOp::create(rewriter, loc, src, srcMap, operands);
        affine::AffineStoreOp::create(rewriter, loc, v, dst, dstMap, operands);
      }
    };

    if (rank == 2 && dim == 0) {
      // Columns: adjacent lanes are contiguous in external storage.
      auto elemLoop = affine::AffineForOp::create(rewriter, loc, 0, length);
      rewriter.setInsertionPointToStart(elemLoop.getBody());
      operands.push_back(elemLoop.getInductionVar());
      elemExpr = getAffineDimExpr(numDims++, ctx);
      if (lanes > 1) {
        auto laneLoop = affine::AffineForOp::create(rewriter, loc, 0, lanes);
        laneLoop->setAttr(kUnrollFactorAttr,
                          rewriter.getI64IntegerAttr(lanes));
        rewriter.setInsertionPointToStart(laneLoop.getBody());
        operands.push_back(laneLoop.getInductionVar());
        laneExpr = getAffineDimExpr(numDims++, ctx);
      }
      transferBody();
      return;
    }

    // Rows (or a rank-1 vector): the element index is contiguous.
    if (lanes > 1) {
      auto laneLoop = affine::AffineForOp::create(rewriter, loc, 0, lanes);
      rewriter.setInsertionPointToStart(laneLoop.getBody());
      operands.push_back(laneLoop.getInductionVar());
      laneExpr = getAffineDimExpr(numDims++, ctx);
    }
    auto elemLoop =
        affine::AffineForOp::create(rewriter, loc, 0, length, io);
    rewriter.setInsertionPointToStart(elemLoop.getBody());
    operands.push_back(elemLoop.getInductionVar());
    elemExpr = getAffineDimExpr(numDims++, ctx);
    if (io > 1) {
      auto beatLoop = affine::AffineForOp::create(rewriter, loc, 0, io);
      beatLoop->setAttr(kUnrollFactorAttr, rewriter.getI64IntegerAttr(io));
      rewriter.setInsertionPointToStart(beatLoop.getBody());
      operands.push_back(beatLoop.getInductionVar());
      elemExpr = elemExpr + getAffineDimExpr(numDims++, ctx);
    }
    transferBody();
  }

  /// Direct Stockham transform over blocks of `parallelRows` lines.
  ///
  /// Every block iteration prefetches its lines into an on-chip block,
  /// runs the butterfly stages on chip with the lane loop innermost, and
  /// writes the block back -- so the external planes only ever see the
  /// unit-stride transfer sweeps.
  static void emitPowerOfTwo(PatternRewriter &rewriter, Location loc,
                             MLIRContext *ctx, ModuleOp module, int64_t length,
                             int64_t lines, int64_t rank, int64_t dim,
                             FloatType elemType, PlanePair in, PlanePair out,
                             bool inverse, unsigned stageGroup,
                             unsigned requestedParallelRows,
                             unsigned requestedIoUnroll) {
    auto [cosBuf, sinBuf] =
        materializeTwiddles(rewriter, loc, module, length, elemType);

    Value scale;
    if (inverse)
      scale = arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getFloatAttr(elemType, 1.0 / static_cast<double>(length)));

    int64_t lanes =
        std::min<int64_t>(std::max(1u, requestedParallelRows), lines);
    while (lanes > 1 && lines % lanes != 0)
      lanes >>= 1;
    int64_t io = std::min<int64_t>(std::max(1u, requestedIoUnroll), length);
    while (io > 1 && length % io != 0)
      io >>= 1;

    int stages = static_cast<int>(radixSchedule(length).size());
    int slots = scratchSlots(stages, stageGroup);
    auto allocBlock = [&](int64_t elemFactor) -> PlanePair {
      return {allocLineBuffer(rewriter, loc, lanes, length, elemType,
                              elemFactor),
              allocLineBuffer(rewriter, loc, lanes, length, elemType,
                              elemFactor)};
    };
    // Only the transfer blocks bank their element dimension: the io-wide
    // sweeps store io consecutive elements per cycle. The butterfly
    // scratch keeps one bank per lane -- radix taps conflict under any
    // static banking, and the lanes absorb the II they cost.
    PlanePair srcBlock = allocBlock(io);
    PlanePair dstBlock = allocBlock(io);
    SmallVector<PlanePair> scratch;
    for (int i = 0; i < slots; ++i)
      scratch.push_back(allocBlock(0));

    auto blockLoop =
        affine::AffineForOp::create(rewriter, loc, 0, lines, lanes);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(blockLoop.getBody());
    Value blockIV = blockLoop.getInductionVar();

    emitTransfer(rewriter, loc, ctx, /*toLocal=*/true, rank, dim, length,
                 lanes, io, blockIV, in, srcBlock);
    emitStockham(rewriter, loc, ctx, length, lanes, srcBlock, scratch,
                 dstBlock, cosBuf, sinBuf, inverse, scale, stageGroup);
    emitTransfer(rewriter, loc, ctx, /*toLocal=*/false, rank, dim, length,
                 lanes, io, blockIV, out, dstBlock);
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
                            llvm::function_ref<AffineMap(AffineExpr)>
                                storageMap,
                            unsigned stageGroup) {
    auto pDim = getAffineDimExpr(1, ctx);

    // Bluestein convolves two length-n sequences, so the circular
    // convolution needs at least 2n-1 points to avoid wrap-around.
    int64_t padded = llvm::NextPowerOf2(2 * length - 1);

    // ---- host-folded constants ---------------------------------------- //
    // chirp w[j] = exp(-i pi j^2 / n). exp has period 2 pi i, so j^2 is
    // reduced modulo 2n before entering the angle; the int64 product j*j
    // cannot overflow for any realistic transform (j < n << 2^31).
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
    PlanePair stage = {alloc(), alloc()};
    PlanePair spec = {alloc(), alloc()};
    PlanePair prod = {alloc(), alloc()};
    PlanePair conv = {alloc(), alloc()};

    // The two transforms get their own stage buffers. Sharing them would put
    // both on the same scratch and turn the chain into a cycle, which is the
    // one shape a dataflow backend cannot schedule. Within each transform the
    // grouping option decides how many slots the stages draw from.
    int paddedStages = static_cast<int>(radixSchedule(padded).size());
    int paddedSlots = scratchSlots(paddedStages, stageGroup);
    SmallVector<PlanePair> fwdScratch, invScratch;
    for (int i = 0; i < paddedSlots; ++i) {
      fwdScratch.push_back({alloc(), alloc()});
      invScratch.push_back({alloc(), alloc()});
    }

    AffineMap ioMap = storageMap(pDim);
    AffineMap lineMap = AffineMap::get(3, 0, {pDim}, ctx);

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
            stage.re, lineMap, ops);
      store(arith::AddFOp::create(rewriter, loc, mul(xr, wi), mul(xi, wr)),
            stage.im, lineMap, ops);
    });
    elementLoop(length, padded, [&](ValueRange ops) {
      Value fzero = arith::ConstantOp::create(
          rewriter, loc, rewriter.getFloatAttr(elemType, 0.0));
      store(fzero, stage.re, lineMap, ops);
      store(fzero, stage.im, lineMap, ops);
    });

    // ---- forward transform, pointwise product, inverse transform ------ //
    emitStockham(rewriter, loc, ctx, padded, /*lanes=*/1, stage, fwdScratch,
                 spec, cosBuf, sinBuf,
                 /*inverse=*/false, /*scale=*/nullptr, stageGroup);

    elementLoop(0, padded, [&](ValueRange ops) {
      Value ar = load(spec.re, lineMap, ops), ai = load(spec.im, lineMap, ops);
      Value br = load(kernelReBuf, lineMap, ops);
      Value bi = load(kernelImBuf, lineMap, ops);
      store(arith::SubFOp::create(rewriter, loc, mul(ar, br), mul(ai, bi)),
            prod.re, lineMap, ops);
      store(arith::AddFOp::create(rewriter, loc, mul(ar, bi), mul(ai, br)),
            prod.im, lineMap, ops);
    });

    emitStockham(rewriter, loc, ctx, padded, /*lanes=*/1, prod, invScratch,
                 conv, cosBuf, sinBuf,
                 /*inverse=*/true, /*scale=*/nullptr, stageGroup);

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
      Value cr = load(conv.re, lineMap, ops), ci = load(conv.im, lineMap, ops);
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
  using sar::impl::ConvertSARFFTToAffineBase<
      ConvertSARFFTToAffinePass>::ConvertSARFFTToAffineBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addIllegalOp<FFTSplitOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    patterns.add<FFTSplitToAffinePattern>(context, fftStageGroup,
                                          fftParallelRows, fftIoUnroll);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
