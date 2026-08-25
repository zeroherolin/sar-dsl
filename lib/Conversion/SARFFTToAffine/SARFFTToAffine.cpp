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
// processed in blocks of lines: a prefetch sweep copies the block from the
// source planes into on-chip line buffers with unit-stride external accesses
// (`fft-io-unroll` elements per beat, so the bus can widen and burst), the
// butterfly stages run entirely on the line buffers with
// `fft-parallel-rows` compute lanes, and a mirrored write-back sweep stores
// the block. For a slow-axis transform, the transfer block may be wider than
// the compute engine: sub-blocks reuse the same butterfly lanes while the
// external sweep still fills a complete packed word.
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
#include <numeric>
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

/// Creates (or reuses) one Stockham twiddle table per stage, returning the
/// loaded cos/sin buffers. Stages execute concurrently in a dataflow design;
/// giving each one only the entries it reads prevents Vitis from replicating
/// the complete transform table for every reader.
///
/// A radix-4 butterfly reads the three adjacent entries p*3 .. p*3+2 in one
/// iteration, so that stage is hinted cyclic-3: each read lands in its own
/// bank.
static SmallVector<PlanePair> materializeTwiddles(PatternRewriter &rewriter,
                                                  Location loc, ModuleOp module,
                                                  int64_t length,
                                                  FloatType elemType) {
  SmallVector<PlanePair> tables;
  auto radices = radixSchedule(length);
  tables.reserve(radices.size());
  int64_t span = 1;
  for (auto [stage, radix] : llvm::enumerate(radices)) {
    int64_t nCur = length / span;
    int64_t butterflies = nCur / radix;
    SmallVector<double> cosTable, sinTable;
    cosTable.reserve(butterflies * (radix - 1));
    sinTable.reserve(butterflies * (radix - 1));
    for (int64_t p = 0; p < butterflies; ++p)
      for (int output = 1; output < radix; ++output) {
        double angle = 2.0 * M_PI * static_cast<double>(p * output) /
                       static_cast<double>(nCur);
        cosTable.push_back(std::cos(angle));
        sinTable.push_back(std::sin(angle));
      }
    std::string suffix =
        (Twine(length) + "_s" + Twine(stage) + "_" + typeSuffix(elemType))
            .str();
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
    // Banking pays a whole memory primitive per bank, so only production
    // radix-4 tables large enough to amortize three banks are split.
    if (radix == 4 && length >= 256) {
      setPartitionHint(rewriter, cosBuf, {"cyclic"}, {3});
      setPartitionHint(rewriter, sinBuf, {"cyclic"}, {3});
    }
    tables.push_back({cosBuf, sinBuf});
    span *= radix;
  }
  return tables;
}

/// Scratch slots between grouped Stockham stages. Full unroll uses one slot
/// per intermediate stage; grouped execution uses one per internal group,
/// with a two-slot floor to keep source and destination distinct.
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

/// Banks a transfer block needs to sustain an `io`-wide sweep.
///
/// The sweep stores `io` consecutive elements per cycle, and a two-port
/// block RAM serves two of them per bank, so `ceil(io / 2)` banks already
/// meet the demand. Matching the bank count to `io` instead -- the obvious
/// choice -- buys no extra throughput and splits the line into more, and
/// therefore smaller, physical blocks: every bank occupies a whole memory
/// primitive whatever it holds. Measured on the omega-K engine at 1024,
/// dropping the count this way holds the transfer loops at II 1 and the
/// design's latency within 0.24% while returning 28% of its block RAM and
/// 37% of its UltraRAM, which the lane budget can then spend on
/// parallelism that does convert into latency.
static int64_t transferBanks(int64_t io) { return io > 1 ? (io + 1) / 2 : 0; }

/// Allocates a line block, completely partitioned by lane. `elemFactor`
/// optionally banks transfer buffers along the element dimension.
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

/// Emits grouped Stockham stages over lane-major line blocks. The compact lane
/// loop carries an unroll hint, twiddles are shared across lanes, and each
/// group writes a distinct scratch slot. Inverse scaling is folded into the
/// final stores.
static void emitStockham(PatternRewriter &rewriter, Location loc,
                         MLIRContext *ctx, int64_t length, int64_t lanes,
                         PlanePair src, ArrayRef<PlanePair> scratch,
                         PlanePair dst, ArrayRef<PlanePair> twiddles,
                         bool inverse, Value scale, unsigned stageGroup,
                         Value transferLaneBase = {}) {
  auto pDim = getAffineDimExpr(0, ctx);
  auto qDim = getAffineDimExpr(1, ctx);
  auto radices = radixSchedule(length);
  int stages = static_cast<int>(radices.size());
  assert(twiddles.size() == radices.size() &&
         "one twiddle table is required per Stockham stage");
  int slots = scratchSlots(stages, stageGroup);
  assert(scratch.size() >= static_cast<size_t>(slots) &&
         "not enough scratch lines for the chosen stage grouping");

  // Element maps onto the lane-major line blocks. Butterfly operands are
  // (p, q) plus, when lanes run in parallel, the innermost lane.
  bool hasLaneLoop = lanes > 1;
  unsigned numDims = 2 + hasLaneLoop + static_cast<bool>(transferLaneBase);
  AffineExpr laneDim = hasLaneLoop ? getAffineDimExpr(2, ctx) : AffineExpr();
  AffineExpr transferLaneDim =
      transferLaneBase
          ? getAffineDimExpr(2 + static_cast<unsigned>(hasLaneLoop), ctx)
          : AffineExpr();
  auto elementMap = [&](AffineExpr elem, bool transferBlock) {
    if (transferBlock) {
      assert(transferLaneBase &&
             "a transfer sub-block needs its lane base operand");
      AffineExpr lane = transferLaneDim;
      if (hasLaneLoop)
        lane = lane + laneDim;
      return AffineMap::get(numDims, 0, {lane, elem}, ctx);
    }
    if (hasLaneLoop)
      return AffineMap::get(numDims, 0, {laneDim, elem}, ctx);
    return AffineMap::get(numDims, 0, {elem}, ctx);
  };
  // Twiddle loads sit above the lane loop and depend on p alone.
  auto twiddleMap = [&](AffineExpr index) {
    return AffineMap::get(1, 0, {index}, ctx);
  };

  PlanePair cur = src;
  int64_t span = 1;
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
    qLoop->setAttr(kMinIIAttr, rewriter.getI64IntegerAttr(2));
    rewriter.setInsertionPointToStart(qLoop.getBody());

    // One twiddle fetch per butterfly, shared across all lanes.
    Value cosBuf = twiddles[t].re;
    Value sinBuf = twiddles[t].im;
    SmallVector<Value, 3> twCos, twSin;
    for (int output = 1; output < radix; ++output) {
      AffineExpr twiddleIndex = pDim * (radix - 1) + (output - 1);
      twCos.push_back(affine::AffineLoadOp::create(
          rewriter, loc, cosBuf, twiddleMap(twiddleIndex),
          ValueRange{pLoop.getInductionVar()}));
      twSin.push_back(affine::AffineLoadOp::create(
          rewriter, loc, sinBuf, twiddleMap(twiddleIndex),
          ValueRange{pLoop.getInductionVar()}));
    }

    SmallVector<Value> operands{pLoop.getInductionVar(),
                                qLoop.getInductionVar()};
    if (hasLaneLoop) {
      auto laneLoop = affine::AffineForOp::create(rewriter, loc, 0, lanes);
      laneLoop->setAttr(kUnrollFactorAttr, rewriter.getI64IntegerAttr(lanes));
      rewriter.setInsertionPointToStart(laneLoop.getBody());
      operands.push_back(laneLoop.getInductionVar());
    }
    if (transferLaneBase)
      operands.push_back(transferLaneBase);

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
      bool fromTransferBlock = transferLaneBase && t == 0;
      inputRe.push_back(load(cur.re, elementMap(index, fromTransferBlock)));
      inputIm.push_back(load(cur.im, elementMap(index, fromTransferBlock)));
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
      bool toTransferBlock = transferLaneBase && t == stages - 1;
      store(outRe, nxt.re, elementMap(outputIndex, toTransferBlock));
      store(outIm, nxt.im, elementMap(outputIndex, toTransferBlock));
    }

    cur = nxt;
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
  /// Transfers between external planes and a line block. Loop order and the
  /// `io` unroll hint keep the external index contiguous for either FFT axis.
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
      // Columns. What lies contiguously in external storage here is the
      // *line* index, not the element index: consecutive lines of one block
      // are consecutive addresses within a row. So the lane loop is the one
      // that has to sweep whole bus words, and it carries the `io` unroll
      // that the row case gives to the element loop.
      //
      // Splitting it into an outer step and an inner run of `io` is what
      // lets the packing pass see the run: a single unrolled lane loop
      // leaves `io` separate scalar accesses to one word, and each becomes
      // its own read-modify-write of that word rather than one packed
      // transfer.
      auto elemLoop = affine::AffineForOp::create(rewriter, loc, 0, length);
      rewriter.setInsertionPointToStart(elemLoop.getBody());
      operands.push_back(elemLoop.getInductionVar());
      elemExpr = getAffineDimExpr(numDims++, ctx);
      if (lanes > 1) {
        int64_t run = std::gcd(io, lanes);
        auto laneLoop =
            affine::AffineForOp::create(rewriter, loc, 0, lanes, run);
        rewriter.setInsertionPointToStart(laneLoop.getBody());
        operands.push_back(laneLoop.getInductionVar());
        laneExpr = getAffineDimExpr(numDims++, ctx);
        if (run > 1) {
          auto beatLoop = affine::AffineForOp::create(rewriter, loc, 0, run);
          beatLoop->setAttr(kUnrollFactorAttr, rewriter.getI64IntegerAttr(run));
          rewriter.setInsertionPointToStart(beatLoop.getBody());
          operands.push_back(beatLoop.getInductionVar());
          laneExpr = laneExpr + getAffineDimExpr(numDims++, ctx);
        } else {
          laneLoop->setAttr(kUnrollFactorAttr,
                            rewriter.getI64IntegerAttr(lanes));
        }
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
    auto elemLoop = affine::AffineForOp::create(rewriter, loc, 0, length, io);
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

  /// Direct Stockham transform over staged blocks of lines.
  ///
  /// Every block iteration prefetches its lines into an on-chip block, runs
  /// the butterfly stages on chip with the lane loop innermost, and writes
  /// the block back. A slow-axis transfer stages at least one packed word of
  /// adjacent lines even when the compute engine has fewer lanes; the engine
  /// then visits that block in sub-blocks. This preserves unit-stride AXI
  /// sweeps without multiplying every Stockham scratch buffer.
  static void emitPowerOfTwo(PatternRewriter &rewriter, Location loc,
                             MLIRContext *ctx, ModuleOp module, int64_t length,
                             int64_t lines, int64_t rank, int64_t dim,
                             FloatType elemType, PlanePair in, PlanePair out,
                             bool inverse, unsigned stageGroup,
                             unsigned requestedParallelRows,
                             unsigned requestedIoUnroll) {
    auto twiddles =
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
    bool slowAxis = rank == 2 && dim == 0;
    int64_t transferExtent = slowAxis ? lines : length;
    int64_t io =
        std::min<int64_t>(std::max(1u, requestedIoUnroll), transferExtent);
    while (io > 1 && transferExtent % io != 0)
      io >>= 1;
    int64_t transferLanes = slowAxis ? std::max(lanes, io) : lanes;
    // A slow-axis transfer block spans `transferLanes` lines and the compute
    // engine walks it `lanes` at a time, so a lane count that does not
    // divide the block leaves a final sub-block running past its end. The
    // autotuner only offers powers of two, where this holds already; a
    // hand-pinned option need not, so the lanes are reduced to a divisor
    // rather than allowed to index out of bounds.
    while (lanes > 1 && transferLanes % lanes != 0)
      lanes >>= 1;

    int stages = static_cast<int>(radixSchedule(length).size());
    int slots = scratchSlots(stages, stageGroup);
    auto allocBlock = [&](int64_t blockLanes, int64_t elemFactor) -> PlanePair {
      return {allocLineBuffer(rewriter, loc, blockLanes, length, elemType,
                              elemFactor),
              allocLineBuffer(rewriter, loc, blockLanes, length, elemType,
                              elemFactor)};
    };
    // Only the transfer blocks bank their element dimension, and only as
    // far as their sweep needs (see `transferBanks`). The butterfly scratch
    // keeps one bank per lane: a radix-4 stage writes sixteen values per
    // lane per iteration, so at the unrolled lane count the write demand
    // exceeds what any cyclic factor can serve from two-port blocks -- the
    // achieved II of 2 is that density, not a bank conflict. Measured on
    // the omega-K engine at 1024, banking the scratch as well leaves the II
    // unchanged and costs 28% more block RAM. The lanes absorb the II
    // instead.
    // A slow-axis sweep widens over adjacent lines, so complete lane banking
    // is sufficient. Banking its element dimension as well only fragments
    // the line storage without serving a concurrent access.
    int64_t transferElemFactor = slowAxis ? 0 : transferBanks(io);
    PlanePair srcBlock = allocBlock(transferLanes, transferElemFactor);
    PlanePair dstBlock = allocBlock(transferLanes, transferElemFactor);
    SmallVector<PlanePair> scratch;
    for (int i = 0; i < slots; ++i)
      scratch.push_back(allocBlock(lanes, 0));

    auto blockLoop =
        affine::AffineForOp::create(rewriter, loc, 0, lines, transferLanes);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(blockLoop.getBody());
    Value blockIV = blockLoop.getInductionVar();

    emitTransfer(rewriter, loc, ctx, /*toLocal=*/true, rank, dim, length,
                 transferLanes, io, blockIV, in, srcBlock);
    if (transferLanes == lanes) {
      emitStockham(rewriter, loc, ctx, length, lanes, srcBlock, scratch,
                   dstBlock, twiddles, inverse, scale, stageGroup);
    } else {
      auto laneBlockLoop =
          affine::AffineForOp::create(rewriter, loc, 0, transferLanes, lanes);
      OpBuilder::InsertionGuard laneGuard(rewriter);
      rewriter.setInsertionPointToStart(laneBlockLoop.getBody());
      emitStockham(rewriter, loc, ctx, length, lanes, srcBlock, scratch,
                   dstBlock, twiddles, inverse, scale, stageGroup,
                   laneBlockLoop.getInductionVar());
    }
    emitTransfer(rewriter, loc, ctx, /*toLocal=*/false, rank, dim, length,
                 transferLanes, io, blockIV, out, dstBlock);
  }

  /// Bluestein's chirp-z reduction for non-power-of-two sizes.
  ///
  /// The inverse transform reuses the forward path through the conjugate
  /// identity ifft(x) = conj(fft(conj(x))) / n, so only one chirp table and
  /// one kernel spectrum are needed per size.
  static void
  emitBluestein(PatternRewriter &rewriter, Location loc, MLIRContext *ctx,
                ModuleOp module, int64_t length, int64_t dim, int64_t lines,
                FloatType elemType, RankedTensorType tensorType, Value inRe,
                Value inIm, Value outBufRe, Value outBufIm, bool inverse,
                llvm::function_ref<AffineMap(AffineExpr)> storageMap,
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

    auto twiddles =
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
                 spec, twiddles,
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
                 conv, twiddles,
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
