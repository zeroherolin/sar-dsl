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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "sar/Analysis/DisplacementRange.h"
#include "sar/Conversion/Passes.h"
#include "sar/Dialect/SAR/IR/SARDialect.h"
#include "sar/Dialect/SAR/IR/SAROps.h"
#include "sar/Support/HLSHints.h"

#include "llvm/ADT/StringSwitch.h"

#include <cmath>
#include <optional>

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
  StringRef boundary;
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

/// Emits the kernel-based gather at fractional position `posF64`. Returns the
/// (re, im) accumulators in f64. Taps are statically unrolled; the boundary
/// policy resolves out-of-range taps with selects, keeping the body free of
/// control flow.
///
/// `loadAt` supplies the source sample for an already-clamped i64 column
/// index, so the full-plane and banded paths share the tap arithmetic.
using SourceLoader = llvm::function_ref<std::pair<Value, Value>(Value)>;
using PositionBuilder = llvm::function_ref<Value(Value, Value)>;

static std::pair<Value, Value> emitInterpGather(ScalarBuilder &s,
                                                const InterpSpec &spec,
                                                Value posF64, int64_t cols,
                                                SourceLoader loadAt) {
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

    // The sample index the boundary policy resolves to. `zero` and `edge`
    // both hand the raw index to the loader, which clamps it; they differ
    // only in whether the tap keeps its weight below. `reflect` mirrors the
    // index here, so the loader's clamp becomes a no-op.
    Value fetchIdx = idxK;
    if (spec.boundary == "reflect") {
      if (cols == 1) {
        fetchIdx = zeroI;
      } else {
        Value period =
            arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(2 * cols));
        Value folded = arith::RemSIOp::create(b, loc, idxK, period);
        Value negative = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::slt, folded, zeroI);
        folded = arith::SelectOp::create(
            b, loc, negative, arith::AddIOp::create(b, loc, folded, period),
            folded);
        Value firstHalf = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::slt, folded, colsI);
        Value mirrored = arith::SubIOp::create(
            b, loc,
            arith::ConstantOp::create(b, loc,
                                      b.getI64IntegerAttr(2 * cols - 1)),
            folded);
        fetchIdx = arith::SelectOp::create(b, loc, firstHalf, folded, mirrored);
      }
    }

    auto [vRe, vIm] = loadAt(fetchIdx);

    Value weight;
    if (spec.kernel == "nearest") {
      weight = one;
    } else if (spec.kernel == "linear") {
      weight = k == 0 ? s.sub(one, frac) : frac;
    } else if (spec.kernel == "cubic") {
      weight = cubicW[k + 1];
    } else {
      // w(d) = sinc(d) * window(d / (taps/2)), numpy sinc convention. The
      // distance stays measured against the unresolved tap index: the policy
      // changes which sample is fetched, not where the kernel is centred.
      Value dist = s.sub(posF64, arith::SIToFPOp::create(b, loc, s.f64, idxK));
      Value pd = s.mul(dist, s.cst(M_PI));
      Value small = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                          math::AbsFOp::create(b, loc, dist),
                                          s.cst(1e-12));
      Value safePd = arith::SelectOp::create(b, loc, small, one, pd);
      Value sincRaw = s.div(s.sin(pd), safePd);
      Value sinc = arith::SelectOp::create(b, loc, small, one, sincRaw);
      Value t = s.mul(dist, s.cst(1.0 / double(spec.taps / 2)));
      weight = s.mul(sinc, emitSincWindow(s, spec, t));
    }
    if (spec.boundary == "zero")
      weight = arith::SelectOp::create(b, loc, inBounds, weight, zero);

    accRe = s.add(accRe, s.mul(vRe, weight));
    accIm = s.add(accIm, s.mul(vIm, weight));
  }
  return {accRe, accIm};
}

/// Clamps an i64 column index into [0, cols-1] and returns it as an index.
static Value clampToColumn(ScalarBuilder &s, Value idxI64, int64_t cols) {
  OpBuilder &b = s.b;
  Location loc = s.loc;
  Value zeroI = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(0));
  Value maxI = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(cols - 1));
  Value clamped = arith::MinSIOp::create(
      b, loc, arith::MaxSIOp::create(b, loc, idxI64, zeroI), maxI);
  return arith::IndexCastOp::create(b, loc, b.getIndexType(), clamped);
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

static bool canScalarize(Value value) {
  if (isa<BlockArgument>(value))
    return isa<RankedTensorType>(value.getType());
  Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  if (isa<ConstantOp>(op))
    return true;
  if (auto broadcast = dyn_cast<BroadcastOp>(op))
    return canScalarize(broadcast.getInput());

  auto allOperands = [&] {
    return llvm::all_of(op->getOperands(),
                        [&](Value operand) { return canScalarize(operand); });
  };
  return isa<AddOp, SubOp, MulOp, DivOp, AddScalarOp, MulScalarOp, SqrtOp,
             CosOp, SinOp, ExpOp, LogOp, AbsOp, CmpOp, WhereOp, CastOp>(op) &&
         allOperands();
}

static FailureOr<Value> scalarize(OpBuilder &b, Location loc, Value value,
                                  ValueRange indices) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType || tensorType.getRank() != (int64_t)indices.size())
    return failure();

  if (isa<BlockArgument>(value))
    return tensor::ExtractOp::create(b, loc, value, indices).getResult();
  Operation *op = value.getDefiningOp();
  if (!op)
    return failure();

  if (auto constant = dyn_cast<ConstantOp>(op)) {
    auto elements = dyn_cast<DenseElementsAttr>(constant.getValue());
    if (!elements)
      return failure();
    if (elements.isSplat()) {
      auto scalar = cast<TypedAttr>(elements.getSplatValue<Attribute>());
      return arith::ConstantOp::create(b, loc, scalar).getResult();
    }
    return tensor::ExtractOp::create(b, loc, value, indices).getResult();
  }
  if (auto broadcast = dyn_cast<BroadcastOp>(op)) {
    Value index = indices[broadcast.getDim()];
    return scalarize(b, loc, broadcast.getInput(), ValueRange{index});
  }

  auto binary = [&](Value lhs, Value rhs, auto create) -> FailureOr<Value> {
    FailureOr<Value> left = scalarize(b, loc, lhs, indices);
    FailureOr<Value> right = scalarize(b, loc, rhs, indices);
    if (failed(left) || failed(right))
      return failure();
    return create(*left, *right);
  };
  if (auto valueOp = dyn_cast<AddOp>(op))
    return binary(valueOp.getLhs(), valueOp.getRhs(),
                  [&](Value lhs, Value rhs) {
                    return arith::AddFOp::create(b, loc, lhs, rhs).getResult();
                  });
  if (auto valueOp = dyn_cast<SubOp>(op))
    return binary(valueOp.getLhs(), valueOp.getRhs(),
                  [&](Value lhs, Value rhs) {
                    return arith::SubFOp::create(b, loc, lhs, rhs).getResult();
                  });
  if (auto valueOp = dyn_cast<MulOp>(op))
    return binary(valueOp.getLhs(), valueOp.getRhs(),
                  [&](Value lhs, Value rhs) {
                    return arith::MulFOp::create(b, loc, lhs, rhs).getResult();
                  });
  if (auto valueOp = dyn_cast<DivOp>(op))
    return binary(valueOp.getLhs(), valueOp.getRhs(),
                  [&](Value lhs, Value rhs) {
                    return arith::DivFOp::create(b, loc, lhs, rhs).getResult();
                  });

  auto unary = [&](Value input, auto create) -> FailureOr<Value> {
    FailureOr<Value> scalar = scalarize(b, loc, input, indices);
    if (failed(scalar))
      return failure();
    return create(*scalar);
  };
  if (auto valueOp = dyn_cast<AddScalarOp>(op))
    return unary(valueOp.getInput(), [&](Value input) {
      auto type = cast<FloatType>(input.getType());
      Value scalar = arith::ConstantOp::create(
          b, loc, b.getFloatAttr(type, valueOp.getScalar().convertToDouble()));
      return arith::AddFOp::create(b, loc, input, scalar).getResult();
    });
  if (auto valueOp = dyn_cast<MulScalarOp>(op))
    return unary(valueOp.getInput(), [&](Value input) {
      auto type = cast<FloatType>(input.getType());
      Value scalar = arith::ConstantOp::create(
          b, loc, b.getFloatAttr(type, valueOp.getScalar().convertToDouble()));
      return arith::MulFOp::create(b, loc, input, scalar).getResult();
    });
  if (auto valueOp = dyn_cast<SqrtOp>(op))
    return unary(valueOp.getInput(), [&](Value input) {
      return math::SqrtOp::create(b, loc, input).getResult();
    });
  if (auto valueOp = dyn_cast<CosOp>(op))
    return unary(valueOp.getInput(), [&](Value input) {
      return math::CosOp::create(b, loc, input).getResult();
    });
  if (auto valueOp = dyn_cast<SinOp>(op))
    return unary(valueOp.getInput(), [&](Value input) {
      return math::SinOp::create(b, loc, input).getResult();
    });
  if (auto valueOp = dyn_cast<ExpOp>(op))
    return unary(valueOp.getInput(), [&](Value input) {
      return math::ExpOp::create(b, loc, input).getResult();
    });
  if (auto valueOp = dyn_cast<LogOp>(op))
    return unary(valueOp.getInput(), [&](Value input) {
      return math::LogOp::create(b, loc, input).getResult();
    });
  if (auto valueOp = dyn_cast<AbsOp>(op))
    return unary(valueOp.getInput(), [&](Value input) {
      return math::AbsFOp::create(b, loc, input).getResult();
    });

  if (auto cmp = dyn_cast<CmpOp>(op)) {
    auto predicate = llvm::StringSwitch<std::optional<arith::CmpFPredicate>>(
                         cmp.getPredicate())
                         .Case("eq", arith::CmpFPredicate::OEQ)
                         .Case("ne", arith::CmpFPredicate::UNE)
                         .Case("lt", arith::CmpFPredicate::OLT)
                         .Case("le", arith::CmpFPredicate::OLE)
                         .Case("gt", arith::CmpFPredicate::OGT)
                         .Case("ge", arith::CmpFPredicate::OGE)
                         .Default(std::nullopt);
    if (!predicate)
      return failure();
    return binary(cmp.getLhs(), cmp.getRhs(), [&](Value lhs, Value rhs) {
      Value held = arith::CmpFOp::create(b, loc, *predicate, lhs, rhs);
      auto elementType = cast<FloatType>(tensorType.getElementType());
      Value one =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(elementType, 1.0));
      Value zero =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(elementType, 0.0));
      return arith::SelectOp::create(b, loc, held, one, zero).getResult();
    });
  }
  if (auto where = dyn_cast<WhereOp>(op)) {
    FailureOr<Value> mask = scalarize(b, loc, where.getMask(), indices);
    FailureOr<Value> lhs = scalarize(b, loc, where.getLhs(), indices);
    FailureOr<Value> rhs = scalarize(b, loc, where.getRhs(), indices);
    if (failed(mask) || failed(lhs) || failed(rhs))
      return failure();
    auto maskType = cast<FloatType>(
        cast<RankedTensorType>(where.getMask().getType()).getElementType());
    Value zero =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(maskType, 0.0));
    Value held =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UNE, *mask, zero);
    return arith::SelectOp::create(b, loc, held, *lhs, *rhs).getResult();
  }
  if (auto castOp = dyn_cast<CastOp>(op)) {
    FailureOr<Value> input = scalarize(b, loc, castOp.getInput(), indices);
    if (failed(input))
      return failure();
    auto source = dyn_cast<FloatType>((*input).getType());
    auto target = dyn_cast<FloatType>(tensorType.getElementType());
    if (!source || !target)
      return failure();
    if (source == target)
      return *input;
    if (source.getWidth() < target.getWidth())
      return arith::ExtFOp::create(b, loc, target, *input).getResult();
    return arith::TruncFOp::create(b, loc, target, *input).getResult();
  }
  return failure();
}

/// Returns the smallest power of two >= n (n > 0).
static int64_t nextPow2(int64_t n) {
  int64_t p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

/// Emits the full-plane gather loop nest inside an already-positioned builder.
/// `emitInterpGather` receives the raw column index before clamping; it also
/// needs the global column bounds for its in-bounds mask.
static void emitFullPlaneBody(OpBuilder &b, Location loc, ScalarBuilder &s,
                              const InterpSpec &spec, Value reBuf, Value imBuf,
                              PositionBuilder positionAt, Value outRe,
                              Value outIm, int64_t rows, int64_t cols,
                              Type elemType) {
  auto rowLoop = affine::AffineForOp::create(b, loc, 0, rows);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(rowLoop.getBody());
    auto colLoop = affine::AffineForOp::create(b, loc, 0, cols);
    b.setInsertionPointToStart(colLoop.getBody());
    Value i = rowLoop.getInductionVar();
    Value j = colLoop.getInductionVar();
    Value pos = positionAt(i, j);

    // Named lambda: a function_ref bound to a temporary would dangle.
    auto loader = [&](Value idxI64) -> std::pair<Value, Value> {
      Value col = clampToColumn(s, idxI64, cols);
      Value vRe =
          s.toF64(memref::LoadOp::create(b, loc, reBuf, ValueRange{i, col}));
      Value vIm =
          s.toF64(memref::LoadOp::create(b, loc, imBuf, ValueRange{i, col}));
      return {vRe, vIm};
    };

    auto [accRe, accIm] = emitInterpGather(s, spec, pos, cols, loader);
    affine::AffineStoreOp::create(b, loc, s.fromF64(accRe, elemType), outRe,
                                  ValueRange{i, j});
    affine::AffineStoreOp::create(b, loc, s.fromF64(accIm, elemType), outIm,
                                  ValueRange{i, j});
  }
}

/// Emits the banded gather loop nest.
///
/// The window for output column j is source columns [j + bandLo, j + bandHi],
/// exactly `bandW` wide (a power of two), so it slides by one as j advances
/// and one new sample per column keeps it current. That is the line-buffer
/// shape: each source row is read once, sequentially, instead of `taps`
/// scattered reads per output element.
///
/// Residency invariant: at the gather in iteration j, `band[t & (bandW-1)]`
/// holds source column t for every t in [j + bandLo, j + bandHi]. bandW being
/// a power of two makes those bandW indices occupy distinct slots, and the
/// push for column j overwrites exactly the index leaving the window.
///
/// Out-of-row source indices are clamped on staging, exactly as the
/// full-plane path clamps them; those taps still receive zero weight in the
/// gather, so a clamped value never reaches the accumulator.
static void emitBandedBody(OpBuilder &b, Location loc, ScalarBuilder &s,
                           const InterpSpec &spec, Value reBuf, Value imBuf,
                           PositionBuilder positionAt, Value outRe, Value outIm,
                           int64_t rows, int64_t cols, Type elemType,
                           int64_t bandLo, int64_t bandW) {
  assert(bandW > 0 && (bandW & (bandW - 1)) == 0 && "bandW must be a pow2");
  int64_t bandHi = bandLo + bandW - 1;
  Type srcElem = cast<MemRefType>(reBuf.getType()).getElementType();
  MemRefType bandType = MemRefType::get({bandW}, srcElem);

  Value bandRe = memref::AllocOp::create(b, loc, bandType);
  Value bandIm = memref::AllocOp::create(b, loc, bandType);
  // Every unrolled tap reads the band in the same cycle through a
  // data-dependent index no affine analysis can bank; registers are the
  // only layout that serves them all. Wide bands would burn a register
  // per element, so they keep the default banking and pay the port limit.
  if (bandW <= 128)
    for (Value band : {bandRe, bandIm}) {
      auto *alloc = band.getDefiningOp();
      alloc->setAttr(kPartitionKindsAttr, b.getStrArrayAttr({"complete"}));
      alloc->setAttr(kPartitionFactorsAttr, b.getI64ArrayAttr({bandW}));
    }

  Value maskI =
      arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(bandW - 1));
  Value bandHiI =
      arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(bandHi));
  Value bandLoI =
      arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(bandLo));

  // Stages source column `srcColI64` (clamped into the row) into its slot.
  auto stageColumn = [&](Value i, Value srcColI64) {
    Value col = clampToColumn(s, srcColI64, cols);
    Value slot = arith::IndexCastOp::create(
        b, loc, b.getIndexType(),
        arith::AndIOp::create(b, loc, srcColI64, maskI));
    memref::StoreOp::create(
        b, loc, memref::LoadOp::create(b, loc, reBuf, ValueRange{i, col}),
        bandRe, ValueRange{slot});
    memref::StoreOp::create(
        b, loc, memref::LoadOp::create(b, loc, imBuf, ValueRange{i, col}),
        bandIm, ValueRange{slot});
  };

  auto rowLoop = affine::AffineForOp::create(b, loc, 0, rows);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(rowLoop.getBody());
    Value i = rowLoop.getInductionVar();

    // Prime slots for source columns [bandLo, bandHi); the push in iteration
    // j = 0 supplies the last one.
    if (bandW > 1) {
      auto primeLoop = affine::AffineForOp::create(b, loc, 0, bandW - 1);
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(primeLoop.getBody());
      Value t = arith::IndexCastOp::create(b, loc, b.getI64Type(),
                                           primeLoop.getInductionVar());
      stageColumn(i, arith::AddIOp::create(b, loc, t, bandLoI));
    }

    auto colLoop = affine::AffineForOp::create(b, loc, 0, cols);
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(colLoop.getBody());
      Value j = colLoop.getInductionVar();
      Value jI64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), j);

      // Slide the window: bring in source column j + bandHi.
      stageColumn(i, arith::AddIOp::create(b, loc, jI64, bandHiI));

      Value pos = positionAt(i, j);
      // Named lambda: a function_ref bound to a temporary would dangle.
      auto loader = [&](Value idxI64) -> std::pair<Value, Value> {
        Value slot = arith::IndexCastOp::create(
            b, loc, b.getIndexType(),
            arith::AndIOp::create(b, loc, idxI64, maskI));
        Value vRe =
            s.toF64(memref::LoadOp::create(b, loc, bandRe, ValueRange{slot}));
        Value vIm =
            s.toF64(memref::LoadOp::create(b, loc, bandIm, ValueRange{slot}));
        return {vRe, vIm};
      };

      auto [accRe, accIm] = emitInterpGather(s, spec, pos, cols, loader);
      affine::AffineStoreOp::create(b, loc, s.fromF64(accRe, elemType), outRe,
                                    ValueRange{i, j});
      affine::AffineStoreOp::create(b, loc, s.fromF64(accIm, elemType), outIm,
                                    ValueRange{i, j});
    }
  }
}

struct Interp1DSplitLowering : OpRewritePattern<Interp1DSplitOp> {
  bool enableBandedGather;
  int64_t profitThreshold;

  Interp1DSplitLowering(MLIRContext *ctx, bool enableBanded, int64_t thresh)
      : OpRewritePattern(ctx), enableBandedGather(enableBanded),
        profitThreshold(thresh) {}

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
    Value outRe = memref::AllocOp::create(rewriter, loc, bufferType);
    Value outIm = memref::AllocOp::create(rewriter, loc, bufferType);

    InterpSpec spec{op.getKernel(), static_cast<int64_t>(op.getTaps()),
                    op.getWindow(), op.getBeta().convertToDouble(),
                    op.getBoundary()};

    // Derive tap offsets from the spec for band width accounting.
    int64_t tapLo = 1 - spec.taps / 2, tapHi = spec.taps / 2;
    if (spec.kernel == "nearest") {
      tapLo = tapHi = 0;
    } else if (spec.kernel == "linear") {
      tapLo = 0;
      tapHi = 1;
    } else if (spec.kernel == "cubic") {
      tapLo = -1;
      tapHi = 2;
    }

    // Attempt banded path when enabled. `reflect` is excluded: a mirrored
    // index jumps back inside the row, outside the sliding window the band
    // holds, so the residency invariant would not cover it. `edge` is safe --
    // staging already clamps, so a slot outside the row holds the edge sample.
    bool useBanded = false;
    int64_t bandLo = 0, bandW = 0;
    if (enableBandedGather && spec.boundary != "reflect") {
      auto range = computeDisplacementRange(op.getPositions(), /*dim=*/1);
      if (range) {
        int64_t dLo = static_cast<int64_t>(std::floor(range->lo));
        int64_t dHi = static_cast<int64_t>(std::ceil(range->hi));
        // Band must cover the displacement range plus the tap support.
        int64_t rawLo = dLo + tapLo;
        int64_t rawW = (dHi - dLo) + (tapHi - tapLo + 1);
        int64_t w = nextPow2(std::max<int64_t>(rawW, 1));
        if (cols / w >= profitThreshold) {
          bandLo = rawLo;
          bandW = w;
          useBanded = true;
        }
      }
    }

    FloatType f64 = rewriter.getF64Type();
    ScalarBuilder s{rewriter, loc, f64};
    bool sinkPositions = canScalarize(op.getPositions());
    Value posBuf;
    if (!sinkPositions)
      posBuf = toBuffer(rewriter, loc, op.getPositions());
    auto positionAt = [&](Value i, Value j) -> Value {
      if (!sinkPositions)
        return affine::AffineLoadOp::create(rewriter, loc, posBuf,
                                            ValueRange{i, j});
      FailureOr<Value> position =
          scalarize(rewriter, loc, op.getPositions(), ValueRange{i, j});
      assert(succeeded(position) && "preflight accepted position expression");
      return *position;
    };

    if (useBanded)
      emitBandedBody(rewriter, loc, s, spec, reBuf, imBuf, positionAt, outRe,
                     outIm, rows, cols, elemType, bandLo, bandW);
    else
      emitFullPlaneBody(rewriter, loc, s, spec, reBuf, imBuf, positionAt, outRe,
                        outIm, rows, cols, elemType);

    rewriter.replaceOp(op, {toResultTensor(rewriter, loc, tensorType, outRe),
                            toResultTensor(rewriter, loc, tensorType, outIm)});
    return success();
  }
};

/// Banded lowering for `sar.gather2d_split`: when the displacement of the
/// row coordinate off the output row is provably bounded, the gather runs
/// against a sliding band of whole source rows instead of the resident
/// plane. The band advances one row per output row (`coeff == 1` is what
/// the analysis proves), so each source row is read exactly once.
///
/// The residency invariant is the row-axis analogue of the interpolation
/// band: at output row i, `band[t & (bandW-1)]` holds source row
/// clamp(t) for every t in [i + bandLo, i + bandHi]. Because staging
/// clamps and the gather masks out-of-plane taps by weight (under the
/// `zero` boundary) or wants the clamped sample anyway (`edge`), the
/// numerics match the full-plane path bit for bit.
///
/// The column coordinate stays arbitrary -- whole rows are staged -- so
/// only the row axis needs a bound. When the analysis cannot bound it,
/// the pattern declines and the op takes the full-plane linalg path.
struct Gather2DSplitBandedLowering : OpRewritePattern<Gather2DSplitOp> {
  bool enableBandedGather;
  int64_t profitThreshold;

  Gather2DSplitBandedLowering(MLIRContext *ctx, bool enableBanded,
                              int64_t thresh)
      : OpRewritePattern(ctx), enableBandedGather(enableBanded),
        profitThreshold(thresh) {}

  LogicalResult matchAndRewrite(Gather2DSplitOp op,
                                PatternRewriter &rewriter) const override {
    if (!enableBandedGather)
      return failure();
    // Bilinear taps reach [floor(r), floor(r)+1]; nearest reaches the
    // single rounded row.
    StringRef kernel = op.getKernel();
    int64_t tapLo = 0, tapHi = kernel == "nearest" ? 0 : 1;

    auto range = computeDisplacementRange(op.getRows(), /*dim=*/0);
    if (!range)
      return failure();

    auto dataType = cast<RankedTensorType>(op.getRe().getType());
    auto outType = cast<RankedTensorType>(op.getOutRe().getType());
    int64_t srcRows = dataType.getDimSize(0);
    int64_t srcCols = dataType.getDimSize(1);
    int64_t outRows = outType.getDimSize(0);
    int64_t outCols = outType.getDimSize(1);

    int64_t dLo = static_cast<int64_t>(std::floor(range->lo));
    int64_t dHi = static_cast<int64_t>(std::ceil(range->hi));
    int64_t bandLo = dLo + tapLo;
    int64_t bandW =
        nextPow2(std::max<int64_t>((dHi - dLo) + (tapHi - tapLo + 1), 1));
    if (srcRows / bandW < profitThreshold)
      return failure();

    Location loc = op.getLoc();
    Type elemType = outType.getElementType();
    Type srcElem = dataType.getElementType();
    Value reBuf = toBuffer(rewriter, loc, op.getRe());
    Value imBuf = toBuffer(rewriter, loc, op.getIm());
    Value rowsBuf = toBuffer(rewriter, loc, op.getRows());
    Value colsBuf = toBuffer(rewriter, loc, op.getCols());
    auto outBufType = MemRefType::get(outType.getShape(), elemType);
    Value outRe = memref::AllocOp::create(rewriter, loc, outBufType);
    Value outIm = memref::AllocOp::create(rewriter, loc, outBufType);

    OpBuilder &b = rewriter;
    FloatType f64 = b.getF64Type();
    ScalarBuilder s{b, loc, f64};

    MemRefType bandType = MemRefType::get({bandW, srcCols}, srcElem);
    Value bandRe = memref::AllocOp::create(b, loc, bandType);
    Value bandIm = memref::AllocOp::create(b, loc, bandType);

    auto i64c = [&](int64_t v) {
      return arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(v))
          .getResult();
    };
    Value maskI = i64c(bandW - 1);
    Value bandLoI = i64c(bandLo);
    Value bandHiI = i64c(bandLo + bandW - 1);
    Value zeroI = i64c(0), srcRowMax = i64c(srcRows - 1);

    // Stages source row clamp(rowI64) into its slot: one contiguous
    // row copy, the streaming-friendly access the band exists for.
    auto stageRow = [&](Value rowI64) {
      Value clamped = arith::MinSIOp::create(
          b, loc, arith::MaxSIOp::create(b, loc, rowI64, zeroI), srcRowMax);
      Value src = arith::IndexCastOp::create(b, loc, b.getIndexType(), clamped);
      Value slot = arith::IndexCastOp::create(
          b, loc, b.getIndexType(),
          arith::AndIOp::create(b, loc, rowI64, maskI));
      auto copyLoop = affine::AffineForOp::create(b, loc, 0, srcCols);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(copyLoop.getBody());
      Value c = copyLoop.getInductionVar();
      Value vRe = memref::LoadOp::create(b, loc, reBuf, ValueRange{src, c});
      Value vIm = memref::LoadOp::create(b, loc, imBuf, ValueRange{src, c});
      memref::StoreOp::create(b, loc, vRe, bandRe, ValueRange{slot, c});
      memref::StoreOp::create(b, loc, vIm, bandIm, ValueRange{slot, c});
    };

    // Reads the band at an unclamped source row (slot by mask) and a
    // clamped column.
    auto loadAt = [&](Value rowI64, Value colIdx) -> std::pair<Value, Value> {
      Value slot = arith::IndexCastOp::create(
          b, loc, b.getIndexType(),
          arith::AndIOp::create(b, loc, rowI64, maskI));
      Value vRe = s.toF64(
          memref::LoadOp::create(b, loc, bandRe, ValueRange{slot, colIdx}));
      Value vIm = s.toF64(
          memref::LoadOp::create(b, loc, bandIm, ValueRange{slot, colIdx}));
      return {vRe, vIm};
    };

    // Prime rows [bandLo, bandHi); the push at i = 0 supplies the last.
    if (bandW > 1) {
      auto primeLoop = affine::AffineForOp::create(b, loc, 0, bandW - 1);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(primeLoop.getBody());
      Value t = arith::IndexCastOp::create(b, loc, b.getI64Type(),
                                           primeLoop.getInductionVar());
      stageRow(arith::AddIOp::create(b, loc, t, bandLoI));
    }

    auto rowLoop = affine::AffineForOp::create(b, loc, 0, outRows);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(rowLoop.getBody());
      Value i = rowLoop.getInductionVar();
      Value iI64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), i);
      stageRow(arith::AddIOp::create(b, loc, iI64, bandHiI));

      auto colLoop = affine::AffineForOp::create(b, loc, 0, outCols);
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(colLoop.getBody());
      Value j = colLoop.getInductionVar();

      Value rowPos = s.toF64(
          affine::AffineLoadOp::create(b, loc, rowsBuf, ValueRange{i, j}));
      Value colPos = s.toF64(
          affine::AffineLoadOp::create(b, loc, colsBuf, ValueRange{i, j}));

      // Tap accumulation, matching the full-plane semantics: weights
      // masked out of plane under `zero`, clamped sample under `edge`.
      Value zeroF = s.cst(0.0), oneF = s.cst(1.0);
      Value srcRowsI = i64c(srcRows), srcColsI = i64c(srcCols);
      Value srcColMax = i64c(srcCols - 1);
      Value accRe = zeroF, accIm = zeroF;
      auto inRange = [&](Value idx, Value extent) {
        Value lo = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, idx,
                                         zeroI);
        Value hi = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, idx,
                                         extent);
        return arith::AndIOp::create(b, loc, lo, hi).getResult();
      };
      auto tap = [&](Value rowIdx, Value colIdx, Value weight) {
        if (op.getBoundary() == "zero") {
          Value inb = arith::AndIOp::create(b, loc, inRange(rowIdx, srcRowsI),
                                            inRange(colIdx, srcColsI));
          weight = arith::SelectOp::create(b, loc, inb, weight, zeroF);
        }
        Value colClamped = arith::IndexCastOp::create(
            b, loc, b.getIndexType(),
            arith::MinSIOp::create(
                b, loc, arith::MaxSIOp::create(b, loc, colIdx, zeroI),
                srcColMax));
        auto [re, im] = loadAt(rowIdx, colClamped);
        accRe = s.add(accRe, s.mul(re, weight));
        accIm = s.add(accIm, s.mul(im, weight));
      };

      if (kernel == "nearest") {
        Value half = s.cst(0.5);
        Value rowIdx = emitFloorI64(s, s.add(rowPos, half));
        Value colIdx = emitFloorI64(s, s.add(colPos, half));
        tap(rowIdx, colIdx, oneF);
      } else {
        Value r0 = emitFloorI64(s, rowPos);
        Value c0 = emitFloorI64(s, colPos);
        Value fr = s.sub(rowPos, arith::SIToFPOp::create(b, loc, f64, r0));
        Value fc = s.sub(colPos, arith::SIToFPOp::create(b, loc, f64, c0));
        Value oneMinusFr = s.sub(oneF, fr);
        Value oneMinusFc = s.sub(oneF, fc);
        Value oneI = i64c(1);
        Value r1 = arith::AddIOp::create(b, loc, r0, oneI);
        Value c1 = arith::AddIOp::create(b, loc, c0, oneI);
        tap(r0, c0, s.mul(oneMinusFr, oneMinusFc));
        tap(r0, c1, s.mul(oneMinusFr, fc));
        tap(r1, c0, s.mul(fr, oneMinusFc));
        tap(r1, c1, s.mul(fr, fc));
      }

      affine::AffineStoreOp::create(b, loc, s.fromF64(accRe, elemType), outRe,
                                    ValueRange{i, j});
      affine::AffineStoreOp::create(b, loc, s.fromF64(accIm, elemType), outIm,
                                    ValueRange{i, j});
    }

    rewriter.replaceOp(op, {toResultTensor(rewriter, loc, outType, outRe),
                            toResultTensor(rewriter, loc, outType, outIm)});
    return success();
  }
};

struct ConvertSARInterpToAffinePass
    : sar::impl::ConvertSARInterpToAffineBase<ConvertSARInterpToAffinePass> {
  using ConvertSARInterpToAffineBase::ConvertSARInterpToAffineBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // The banded gather2d pattern runs greedily first: the op stays legal
    // either way (the full-plane linalg path picks up whatever the band
    // cannot prove), so a conversion driver would never visit it.
    {
      RewritePatternSet banded(context);
      banded.add<Gather2DSplitBandedLowering>(context, enableBandedGather,
                                              bandedProfitThreshold);
      (void)applyPatternsGreedily(getOperation(), std::move(banded));
    }

    ConversionTarget target(*context);
    target.addIllegalOp<Interp1DSplitOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    patterns.add<Interp1DSplitLowering>(context, enableBandedGather,
                                        bandedProfitThreshold);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
