//===- DisplacementRange.cpp - Bounds on resampling displacement ----------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Conservative bounds on `|positions[i, j] - j|`, the displacement a
// resampling stage applies. A bounded displacement is what lets the gather
// read a sliding band instead of the resident plane; why SAR geometry makes
// the bound exist is in docs/architecture.md.
//
// Two analyses, tried in order:
//
//   1. An affine domain. Each value is abstracted as `coeff * k + offset`,
//      where `k` indexes the resampled axis and `offset` is an interval over
//      everything that does not vary with it. The displacement is then
//      `(coeff - 1) * j + offset`, bounded exactly when `coeff == 1`. This is
//      symbolic, so it holds for fields built from kernel arguments -- but
//      any nonlinearity over the ramp (`sqrt`, a product of two ramps) forces
//      `coeff` to zero and an unbounded verdict, and `sqrt(f^2 + g^2)` is
//      exactly the shape a frequency-domain remapping takes.
//
//   2. Element-wise constant folding. Such a field is nonetheless usually
//      constant: the frequency axes are acquisition metadata the frontend
//      bakes in, so every leaf is a `sar.constant`. Folding the plane reads
//      the displacement off exactly, which is tighter than any interval and
//      covers Stolt and RCMC.
//
// A field that is neither affine nor constant stays unbounded and the
// lowering falls back to the full-plane gather.
//
//===----------------------------------------------------------------------===//

#include "sar/Analysis/DisplacementRange.h"

#include "llvm/ADT/DenseMap.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "sar/Dialect/SAR/IR/SAROps.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace mlir;
using namespace mlir::sar;

namespace mlir {
namespace sar {

//===----------------------------------------------------------------------===//
// Interval arithmetic
//===----------------------------------------------------------------------===//

Interval Interval::join(const Interval &other) const {
  if (!bounded || !other.bounded)
    return unbounded();
  return range(std::min(lo, other.lo), std::max(hi, other.hi));
}

Interval Interval::operator+(const Interval &rhs) const {
  if (!bounded || !rhs.bounded)
    return unbounded();
  return range(lo + rhs.lo, hi + rhs.hi);
}

Interval Interval::operator-(const Interval &rhs) const {
  if (!bounded || !rhs.bounded)
    return unbounded();
  return range(lo - rhs.hi, hi - rhs.lo);
}

Interval Interval::operator*(const Interval &rhs) const {
  if (!bounded || !rhs.bounded)
    return unbounded();
  double a = lo * rhs.lo, b = lo * rhs.hi;
  double c = hi * rhs.lo, d = hi * rhs.hi;
  return range(std::min({a, b, c, d}), std::max({a, b, c, d}));
}

Interval Interval::operator/(const Interval &rhs) const {
  if (!bounded || !rhs.bounded)
    return unbounded();
  // A denominator straddling zero admits arbitrarily large quotients.
  if (rhs.lo <= 0.0 && rhs.hi >= 0.0)
    return unbounded();
  double a = lo / rhs.lo, b = lo / rhs.hi;
  double c = hi / rhs.lo, d = hi / rhs.hi;
  return range(std::min({a, b, c, d}), std::max({a, b, c, d}));
}

} // namespace sar
} // namespace mlir

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Ramp axis sentinel: the value is not tracked along any ramp axis, only
/// its interval matters.
constexpr int64_t kNoRamp = -1;

struct Slice {
  int64_t axis = -1;
  int64_t begin = 0;
  int64_t end = 0;

  bool active() const { return axis >= 0; }
};

static bool finite(double v) { return std::isfinite(v); }

static bool intervalFinite(const Interval &iv) {
  return iv.bounded && finite(iv.lo) && finite(iv.hi);
}

/// True when the abstract value carries no ramp, i.e. it is a plain interval.
static bool isPlain(const AffineInterval &a) { return a.coeff == 0.0; }

/// True when the abstract value is a single compile-time number.
static bool isPoint(const AffineInterval &a) {
  return a.coeff == 0.0 && a.offset.bounded && a.offset.lo == a.offset.hi;
}

static AffineInterval unbounded() { return AffineInterval::unbounded(); }

static Interval squareInterval(const Interval &value) {
  if (!intervalFinite(value))
    return Interval::unbounded();
  double lo = value.lo * value.lo;
  double hi = value.hi * value.hi;
  double maximum = std::max(lo, hi);
  double minimum = value.lo <= 0.0 && value.hi >= 0.0 ? 0.0 : std::min(lo, hi);
  return Interval::range(minimum, maximum);
}

//===----------------------------------------------------------------------===//
// Abstract operations on `coeff * k + offset`
//===----------------------------------------------------------------------===//

static AffineInterval scaleBy(const AffineInterval &a, double s) {
  return AffineInterval{a.coeff * s, a.offset * Interval::point(s)};
}

static AffineInterval addAI(const AffineInterval &a, const AffineInterval &b) {
  return AffineInterval{a.coeff + b.coeff, a.offset + b.offset};
}

static AffineInterval subAI(const AffineInterval &a, const AffineInterval &b) {
  return AffineInterval{a.coeff - b.coeff, a.offset - b.offset};
}

/// A product stays affine only when at most one factor carries a ramp, and
/// the other factor is a single number (multiplying a ramp by a non-degenerate
/// interval yields a set that is not of the form `coeff*k + offset`).
static AffineInterval mulAI(const AffineInterval &a, const AffineInterval &b) {
  if (isPlain(a) && isPlain(b))
    return AffineInterval::constant(a.offset * b.offset);
  if (isPlain(a)) {
    if (!isPoint(a))
      return unbounded();
    return scaleBy(b, a.offset.lo);
  }
  if (isPlain(b)) {
    if (!isPoint(b))
      return unbounded();
    return scaleBy(a, b.offset.lo);
  }
  // Both carry ramps: the product contains k^2.
  return unbounded();
}

static AffineInterval divAI(const AffineInterval &a, const AffineInterval &b) {
  // Dividing by anything that varies is outside the domain.
  if (!isPlain(b))
    return unbounded();
  if (isPlain(a))
    return AffineInterval::constant(a.offset / b.offset);
  if (!isPoint(b) || b.offset.lo == 0.0)
    return unbounded();
  return scaleBy(a, 1.0 / b.offset.lo);
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

/// Coordinate along `axis` of the row-major linear index `linear`.
static int64_t coordinateAt(int64_t linear, ArrayRef<int64_t> shape,
                            int64_t axis) {
  int64_t stride = 1;
  for (int64_t d = axis + 1; d < (int64_t)shape.size(); ++d)
    stride *= shape[d];
  return (linear / stride) % shape[axis];
}

static bool inSlice(int64_t linear, ArrayRef<int64_t> shape, Slice slice) {
  if (!slice.active())
    return true;
  if (slice.axis >= (int64_t)shape.size())
    return false;
  int64_t coordinate = coordinateAt(linear, shape, slice.axis);
  return coordinate >= slice.begin && coordinate < slice.end;
}

/// Plain min/max hull over the elements of a dense attribute that fall
/// inside `slice`.
static Interval hullOfDenseAttr(ElementsAttr attr, Slice slice) {
  if (!isa<FloatType>(attr.getShapedType().getElementType()))
    return Interval::unbounded();
  double lo = kInf, hi = -kInf;
  int64_t linear = 0;
  ArrayRef<int64_t> shape = attr.getShapedType().getShape();
  for (APFloat v : attr.getValues<APFloat>()) {
    if (!inSlice(linear++, shape, slice))
      continue;
    double d = v.convertToDouble();
    if (!finite(d))
      return Interval::unbounded();
    lo = std::min(lo, d);
    hi = std::max(hi, d);
  }
  if (lo > hi) // empty tensor
    return Interval::unbounded();
  return Interval::range(lo, hi);
}

/// Decomposes a dense constant into one common slope along `rampAxis` plus an
/// interval enclosing every residual. Intercepts and floating-point step
/// roundoff are absorbed by the residual, so the result is conservative for
/// arbitrary finite values rather than relying on a tolerance test.
static std::optional<AffineInterval>
progressionOfDenseAttr(ElementsAttr attr, int64_t rampAxis, Slice slice) {
  auto shaped = attr.getShapedType();
  if (!isa<FloatType>(shaped.getElementType()))
    return std::nullopt;
  if (rampAxis < 0 || rampAxis >= shaped.getRank())
    return std::nullopt;
  int64_t n = shaped.getDimSize(rampAxis);
  // A single-element axis carries no slope information; treat it as plain.
  if (n < 2)
    return std::nullopt;

  // Materialize the values in row-major order.
  SmallVector<double> vals;
  vals.reserve(shaped.getNumElements());
  for (APFloat v : attr.getValues<APFloat>()) {
    double d = v.convertToDouble();
    if (!finite(d))
      return std::nullopt;
    vals.push_back(d);
  }
  if (static_cast<int64_t>(vals.size()) != shaped.getNumElements())
    return std::nullopt;

  ArrayRef<int64_t> shape = shaped.getShape();
  // Row-major stride of the ramp axis, and the number of independent lines.
  int64_t stride = 1;
  for (int64_t d = rampAxis + 1; d < shaped.getRank(); ++d)
    stride *= shape[d];
  int64_t outer = 1;
  for (int64_t d = 0; d < rampAxis; ++d)
    outer *= shape[d];

  int64_t slopeBase = -1;
  for (int64_t o = 0; o < outer && slopeBase < 0; ++o)
    for (int64_t s = 0; s < stride; ++s) {
      int64_t base = o * n * stride + s;
      if (inSlice(base, shape, slice) && inSlice(base + stride, shape, slice)) {
        slopeBase = base;
        break;
      }
    }
  if (slopeBase < 0)
    return std::nullopt;
  double slope = vals[slopeBase + stride] - vals[slopeBase];
  if (!finite(slope))
    return std::nullopt;
  double residualLo = kInf, residualHi = -kInf;

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t s = 0; s < stride; ++s) {
      int64_t base = o * n * stride + s;
      for (int64_t k = 0; k < n; ++k) {
        int64_t linear = base + k * stride;
        if (!inSlice(linear, shape, slice))
          continue;
        double got = vals[linear];
        double residual = got - slope * static_cast<double>(k);
        if (!finite(residual))
          return std::nullopt;
        residualLo = std::min(residualLo, residual);
        residualHi = std::max(residualHi, residual);
      }
    }
  }
  if (!finite(residualLo) || !finite(residualHi))
    return std::nullopt;
  return AffineInterval{slope, Interval::range(residualLo, residualHi)};
}

//===----------------------------------------------------------------------===//
// SSA walk
//===----------------------------------------------------------------------===//

struct Analyzer {
  using Key = std::tuple<Value, int64_t, int64_t, int64_t, int64_t>;
  DenseMap<Key, AffineInterval> memo;
  /// Guards against cycles (SSA in a region can be cyclic through blocks).
  DenseSet<Key> inFlight;

  AffineInterval analyze(Value v, int64_t rampAxis, Slice slice = {});
  AffineInterval analyzeOp(Operation *op, int64_t rampAxis, Slice slice);
};

/// Evaluates a position field numerically when every leaf it depends on is a
/// compile-time constant, writing the plane into `out` (row-major, shaped
/// like the op's result).
///
/// The affine form above gives up as soon as a nonlinearity touches the ramp:
/// `sqrt(a^2 + b^2)` -- the Stolt mapping, and the shape most frequency-domain
/// remappings take -- collapses the coefficient to zero and reports unbounded,
/// even though the field is fully determined at compile time. Folding it
/// yields the exact displacement instead of a conservative one, and costs a
/// pass over a plane the compiler already knows.
struct ConstantFolder {
  /// Bounds the work: a field wider than this is left to the affine path.
  // Keep the worst-case memoized plane to 2 MiB. Larger fields use interval
  // analysis; materializing several 2048-square SSA intermediates here can
  // otherwise consume hundreds of MiB during compilation.
  static constexpr int64_t kMaxElements = 1 << 18;

  DenseMap<Value, std::optional<SmallVector<double>>> memo;

  static ArrayRef<int64_t> shapeOf(Value v) {
    return cast<RankedTensorType>(v.getType()).getShape();
  }

  /// Element-wise apply, broadcasting nothing: SAR element-wise ops already
  /// carry matching shapes.
  template <typename Fn>
  static std::optional<SmallVector<double>>
  zip(const std::optional<SmallVector<double>> &a,
      const std::optional<SmallVector<double>> &b, Fn fn) {
    if (!a || !b || a->size() != b->size())
      return std::nullopt;
    SmallVector<double> out(a->size());
    for (size_t i = 0; i < a->size(); ++i)
      out[i] = fn((*a)[i], (*b)[i]);
    return out;
  }

  template <typename Fn>
  static std::optional<SmallVector<double>>
  map(const std::optional<SmallVector<double>> &a, Fn fn) {
    if (!a)
      return std::nullopt;
    SmallVector<double> out(a->size());
    for (size_t i = 0; i < a->size(); ++i)
      out[i] = fn((*a)[i]);
    return out;
  }

  std::optional<SmallVector<double>> eval(Value v) {
    auto it = memo.find(v);
    if (it != memo.end())
      return it->second;
    memo.insert({v, std::nullopt}); // cycle guard

    auto type = dyn_cast<RankedTensorType>(v.getType());
    if (!type || !type.hasStaticShape() ||
        !isa<FloatType>(type.getElementType()) ||
        type.getNumElements() > kMaxElements)
      return std::nullopt;

    std::optional<SmallVector<double>> result = evalOp(v);
    memo[v] = result;
    return result;
  }

  std::optional<SmallVector<double>> evalOp(Value v) {
    Operation *op = v.getDefiningOp();
    if (!op)
      return std::nullopt; // kernel argument: not compile-time known

    if (auto cst = dyn_cast<ConstantOp>(op)) {
      auto attr = dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!attr)
        return std::nullopt;
      SmallVector<double> out;
      out.reserve(attr.getNumElements());
      for (APFloat f : attr.getValues<APFloat>()) {
        bool lossy = false;
        f.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &lossy);
        out.push_back(f.convertToDouble());
      }
      return out;
    }

    if (auto bc = dyn_cast<BroadcastOp>(op)) {
      auto src = eval(bc.getInput());
      if (!src)
        return std::nullopt;
      ArrayRef<int64_t> shape = shapeOf(bc.getResult());
      if (shape.size() != 2)
        return std::nullopt;
      SmallVector<double> out(shape[0] * shape[1]);
      bool alongRows = bc.getDim() == 0;
      for (int64_t i = 0; i < shape[0]; ++i)
        for (int64_t j = 0; j < shape[1]; ++j)
          out[i * shape[1] + j] = (*src)[alongRows ? i : j];
      return out;
    }

    // Arithmetic in f32 rounds after every operation. Evaluating it in the
    // host's double type could prove a narrower band than the runtime value;
    // leave it to the full-plane path unless the value was a constant or a
    // pure broadcast handled above.
    if (auto type = dyn_cast<RankedTensorType>(v.getType()))
      if (type.getElementType().isF32())
        return std::nullopt;

    if (auto o = dyn_cast<AddOp>(op))
      return zip(eval(o.getLhs()), eval(o.getRhs()),
                 [](double a, double b) { return a + b; });
    if (auto o = dyn_cast<SubOp>(op))
      return zip(eval(o.getLhs()), eval(o.getRhs()),
                 [](double a, double b) { return a - b; });
    if (auto o = dyn_cast<MulOp>(op))
      return zip(eval(o.getLhs()), eval(o.getRhs()),
                 [](double a, double b) { return a * b; });
    if (auto o = dyn_cast<DivOp>(op))
      return zip(eval(o.getLhs()), eval(o.getRhs()),
                 [](double a, double b) { return a / b; });
    if (auto o = dyn_cast<AddScalarOp>(op)) {
      double s = o.getScalar().convertToDouble();
      return map(eval(o.getInput()), [s](double a) { return a + s; });
    }
    if (auto o = dyn_cast<MulScalarOp>(op)) {
      double s = o.getScalar().convertToDouble();
      return map(eval(o.getInput()), [s](double a) { return a * s; });
    }
    if (auto o = dyn_cast<SqrtOp>(op)) {
      auto src = eval(o.getInput());
      if (!src)
        return std::nullopt;
      // A negative operand yields NaN at run time; refuse rather than
      // invent a finite value the hardware will not produce.
      for (double a : *src)
        if (a < 0.0)
          return std::nullopt;
      return map(src, [](double a) { return std::sqrt(a); });
    }
    if (auto o = dyn_cast<CosOp>(op))
      return map(eval(o.getInput()), [](double a) { return std::cos(a); });
    if (auto o = dyn_cast<SinOp>(op))
      return map(eval(o.getInput()), [](double a) { return std::sin(a); });
    if (auto o = dyn_cast<ExpOp>(op))
      return map(eval(o.getInput()), [](double a) { return std::exp(a); });
    if (auto o = dyn_cast<LogOp>(op)) {
      auto src = eval(o.getInput());
      if (!src || llvm::any_of(*src, [](double a) { return a <= 0.0; }))
        return std::nullopt;
      return map(src, [](double a) { return std::log(a); });
    }
    if (auto o = dyn_cast<Atan2Op>(op))
      return zip(eval(o.getY()), eval(o.getX()),
                 [](double y, double x) { return std::atan2(y, x); });
    if (auto o = dyn_cast<CastOp>(op)) {
      auto inTy = dyn_cast<RankedTensorType>(o.getInput().getType());
      auto outTy = dyn_cast<RankedTensorType>(o.getResult().getType());
      if (!inTy || !outTy || !isa<FloatType>(inTy.getElementType()) ||
          !isa<FloatType>(outTy.getElementType()))
        return std::nullopt;
      return eval(o.getInput());
    }
    if (auto o = dyn_cast<WhereOp>(op)) {
      auto mask = eval(o.getMask());
      auto l = eval(o.getLhs());
      auto r = eval(o.getRhs());
      if (!mask || !l || !r || mask->size() != l->size() ||
          l->size() != r->size())
        return std::nullopt;
      SmallVector<double> out(l->size());
      for (size_t i = 0; i < l->size(); ++i)
        out[i] = (*mask)[i] != 0.0 ? (*l)[i] : (*r)[i];
      return out;
    }
    if (auto o = dyn_cast<CmpOp>(op)) {
      auto l = eval(o.getLhs());
      auto r = eval(o.getRhs());
      StringRef pred = o.getPredicate();
      if (!l || !r || l->size() != r->size())
        return std::nullopt;
      SmallVector<double> out(l->size());
      for (size_t i = 0; i < l->size(); ++i) {
        double a = (*l)[i], b = (*r)[i];
        bool t = pred == "eq"   ? a == b
                 : pred == "ne" ? a != b
                 : pred == "lt" ? a < b
                 : pred == "le" ? a <= b
                 : pred == "gt" ? a > b
                 : pred == "ge" ? a >= b
                                : false;
        if (pred != "eq" && pred != "ne" && pred != "lt" && pred != "le" &&
            pred != "gt" && pred != "ge")
          return std::nullopt;
        out[i] = t ? 1.0 : 0.0;
      }
      return out;
    }
    return std::nullopt;
  }
};

/// Exact displacement bound from a folded position plane, or nullopt when the
/// field is not fully constant.
static std::optional<Interval> foldedDisplacement(Value positions,
                                                  int64_t dim) {
  auto type = dyn_cast<RankedTensorType>(positions.getType());
  if (!type || !type.hasStaticShape() || dim < 0 || dim >= type.getRank())
    return std::nullopt;

  ConstantFolder folder;
  auto values = folder.eval(positions);
  if (!values || static_cast<int64_t>(values->size()) != type.getNumElements())
    return std::nullopt;

  int64_t stride = 1;
  for (int64_t d = dim + 1; d < type.getRank(); ++d)
    stride *= type.getDimSize(d);
  int64_t extent = type.getDimSize(dim);

  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  for (int64_t linear = 0; linear < type.getNumElements(); ++linear) {
    int64_t coordinate = (linear / stride) % extent;
    double displacement = (*values)[linear] - static_cast<double>(coordinate);
    if (!std::isfinite(displacement))
      return std::nullopt;
    lo = std::min(lo, displacement);
    hi = std::max(hi, displacement);
  }
  return Interval::range(lo, hi);
}

AffineInterval Analyzer::analyzeOp(Operation *op, int64_t rampAxis,
                                   Slice slice) {
  // sar.constant -- the only producer of a ramp coefficient.
  if (auto cst = dyn_cast<ConstantOp>(op)) {
    if (rampAxis != kNoRamp) {
      if (auto prog = progressionOfDenseAttr(cst.getValue(), rampAxis, slice))
        return *prog;
      // Not a progression: sound to fall back to the plain hull, which
      // simply reports no ramp.
    }
    return AffineInterval::constant(hullOfDenseAttr(cst.getValue(), slice));
  }

  // sar.broadcast -- result axis `dim` is the source's only axis.
  if (auto bc = dyn_cast<BroadcastOp>(op)) {
    int64_t srcAxis =
        (rampAxis == static_cast<int64_t>(bc.getDim())) ? 0 : kNoRamp;
    Slice srcSlice;
    if (slice.active() && slice.axis == (int64_t)bc.getDim())
      srcSlice = Slice{0, slice.begin, slice.end};
    AffineInterval src = analyze(bc.getInput(), srcAxis, srcSlice);
    if (srcAxis == kNoRamp && !isPlain(src))
      return unbounded(); // defensive: no ramp may survive replication
    return src;
  }

  // The affine domain models real arithmetic. An f32 operation rounds after
  // every step and can destroy an identity ramp at large magnitudes, so it
  // cannot safely establish a band. Fully constant fields still use the
  // exact-folder fallback above, which declines f32 arithmetic for the same
  // reason.
  if (op->getNumResults() == 1)
    if (auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType()))
      if (type.getElementType().isF32())
        return unbounded();

  // Element-wise binaries: shapes match, so the ramp axis passes through.
  if (auto o = dyn_cast<AddOp>(op))
    return addAI(analyze(o.getLhs(), rampAxis, slice),
                 analyze(o.getRhs(), rampAxis, slice));
  if (auto o = dyn_cast<SubOp>(op))
    return subAI(analyze(o.getLhs(), rampAxis, slice),
                 analyze(o.getRhs(), rampAxis, slice));
  if (auto o = dyn_cast<MulOp>(op)) {
    if (o.getLhs() == o.getRhs()) {
      AffineInterval value = analyze(o.getLhs(), rampAxis, slice);
      if (isPlain(value))
        return AffineInterval::constant(squareInterval(value.offset));
    }
    return mulAI(analyze(o.getLhs(), rampAxis, slice),
                 analyze(o.getRhs(), rampAxis, slice));
  }
  if (auto o = dyn_cast<DivOp>(op))
    return divAI(analyze(o.getLhs(), rampAxis, slice),
                 analyze(o.getRhs(), rampAxis, slice));

  // Scalar shift / scale.
  if (auto o = dyn_cast<AddScalarOp>(op)) {
    AffineInterval src = analyze(o.getInput(), rampAxis, slice);
    double s = o.getScalar().convertToDouble();
    return AffineInterval{src.coeff, src.offset + Interval::point(s)};
  }
  if (auto o = dyn_cast<MulScalarOp>(op))
    return scaleBy(analyze(o.getInput(), rampAxis, slice),
                   o.getScalar().convertToDouble());

  // sar.sqrt -- monotone, so endpoints map. A generic ramp does not survive
  // it; the recognized sqrt(x*x + q) form below is the one exception.
  if (auto o = dyn_cast<SqrtOp>(op)) {
    // Preserve a positive ramp through sqrt(x*x + q) by bounding the
    // residual q / (sqrt(x*x + q) + x). This is a general stable form for
    // remapping a coordinate ramp by a non-negative orthogonal term.
    auto trySum = [&](Value sum) -> std::optional<AffineInterval> {
      auto add = sum.getDefiningOp<AddOp>();
      if (!add)
        return std::nullopt;
      auto tryRampSquare =
          [&](Value square, Value orthogonal) -> std::optional<AffineInterval> {
        auto mul = square.getDefiningOp<MulOp>();
        if (!mul || mul.getLhs() != mul.getRhs())
          return std::nullopt;
        Value baseValue = mul.getLhs();
        AffineInterval base = analyze(baseValue, rampAxis, slice);
        AffineInterval q = analyze(orthogonal, rampAxis, slice);
        AffineInterval baseHull = analyze(baseValue, kNoRamp, slice);
        if (base.coeff == 0.0 || !isPlain(q) || !isPlain(baseHull) ||
            !intervalFinite(q.offset) || !intervalFinite(baseHull.offset) ||
            q.offset.lo < 0.0 || baseHull.offset.lo <= 0.0)
          return std::nullopt;

        auto residual = [](double x, double q) {
          return q / (std::sqrt(x * x + q) + x);
        };
        double lo = residual(baseHull.offset.hi, q.offset.lo);
        double hi = residual(baseHull.offset.lo, q.offset.hi);
        return AffineInterval{
            base.coeff,
            base.offset + Interval::range(std::min(lo, hi), std::max(lo, hi))};
      };
      if (auto result = tryRampSquare(add.getLhs(), add.getRhs()))
        return *result;
      if (auto result = tryRampSquare(add.getRhs(), add.getLhs()))
        return *result;
      return std::nullopt;
    };

    SmallVector<Value> candidates{o.getInput()};
    if (auto outer = o.getInput().getDefiningOp<WhereOp>()) {
      candidates.push_back(outer.getLhs());
      candidates.push_back(outer.getRhs());
      if (auto inner = outer.getLhs().getDefiningOp<WhereOp>()) {
        candidates.push_back(inner.getLhs());
        candidates.push_back(inner.getRhs());
      }
      if (auto inner = outer.getRhs().getDefiningOp<WhereOp>()) {
        candidates.push_back(inner.getLhs());
        candidates.push_back(inner.getRhs());
      }
    }
    for (Value candidate : candidates)
      if (auto result = trySum(candidate))
        return *result;

    AffineInterval src = analyze(o.getInput(), rampAxis, slice);
    if (!isPlain(src) || !intervalFinite(src.offset))
      return unbounded();
    // A wholly negative domain yields NaN; refuse rather than invent a value.
    if (src.offset.hi < 0.0)
      return unbounded();
    double lo = src.offset.lo > 0.0 ? std::sqrt(src.offset.lo) : 0.0;
    return AffineInterval::constant(
        Interval::range(lo, std::sqrt(src.offset.hi)));
  }

  // sar.cast between float widths preserves the value range. Casts that
  // round (float -> int) are not modelled.
  if (auto o = dyn_cast<CastOp>(op)) {
    auto inTy = dyn_cast<RankedTensorType>(o.getInput().getType());
    auto outTy = dyn_cast<RankedTensorType>(o.getResult().getType());
    if (!inTy || !outTy || !isa<FloatType>(inTy.getElementType()) ||
        !isa<FloatType>(outTy.getElementType()))
      return unbounded();
    AffineInterval src = analyze(o.getInput(), rampAxis, slice);
    // Narrowing rounds; a ramp is no longer exact, but the interval still
    // holds to within the rounding the hardware itself performs.
    return src;
  }

  // sar.where -- the frontend's maximum/minimum lower to cmp + where, so
  // this covers clamping without naming it. The result is one of the two
  // arms element-wise, hence contained in their hull. Hulling is only valid
  // in the ramp domain when both arms share a coefficient.
  if (auto o = dyn_cast<WhereOp>(op)) {
    AffineInterval l = analyze(o.getLhs(), rampAxis, slice);
    AffineInterval r = analyze(o.getRhs(), rampAxis, slice);
    if (l.coeff != r.coeff)
      return unbounded();
    return AffineInterval{l.coeff, l.offset.join(r.offset)};
  }

  // Everything else -- including transposes, FFTs, reductions and any op
  // added later -- is conservatively unknown.
  return unbounded();
}

AffineInterval Analyzer::analyze(Value v, int64_t rampAxis, Slice slice) {
  auto key = std::make_tuple(v, rampAxis, slice.axis, slice.begin, slice.end);
  auto it = memo.find(key);
  if (it != memo.end())
    return it->second;
  // A value reached while it is still being computed is part of a cycle.
  if (!inFlight.insert(key).second)
    return unbounded();

  AffineInterval result = unbounded();
  if (!isa<BlockArgument>(v)) {
    if (Operation *def = v.getDefiningOp())
      result = analyzeOp(def, rampAxis, slice);
  }
  // Block arguments -- kernel inputs above all -- carry no compile-time
  // range, so they keep the unbounded default.

  inFlight.erase(key);
  memo.insert({key, result});
  return result;
}

} // namespace

namespace mlir {
namespace sar {

static AffineInterval analyzePositions(Value positions, int64_t dim,
                                       Slice slice = {}) {
  Analyzer analyzer;
  return analyzer.analyze(positions, dim, slice);
}

static std::optional<Interval>
displacementFromAffineInterval(AffineInterval ai, RankedTensorType type,
                               int64_t dim) {
  if (!type || !type.hasStaticShape() || dim < 0 || dim >= type.getRank() ||
      !intervalFinite(ai.offset))
    return std::nullopt;
  double end = (ai.coeff - 1.0) * (type.getDimSize(dim) - 1);
  return ai.offset + Interval::range(std::min(0.0, end), std::max(0.0, end));
}

std::optional<Interval> computeDisplacementRange(Value positions, int64_t dim) {
  AffineInterval ai = analyzePositions(positions, dim);
  // displacement = positions - j = (coeff - 1) * j + offset. Kernel shapes
  // are static on this path, so include the finite ramp contribution rather
  // than requiring an exactly unit coefficient.
  auto type = dyn_cast<RankedTensorType>(positions.getType());
  if (auto range = displacementFromAffineInterval(ai, type, dim))
    return range;
  // The symbolic form gives up on any nonlinearity over the ramp. When the
  // field is nonetheless fully determined at compile time, evaluating it
  // gives the exact bound.
  return foldedDisplacement(positions, dim);
}

} // namespace sar
} // namespace mlir
