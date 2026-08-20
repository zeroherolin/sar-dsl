//===- DisplacementRange.h - Resampling displacement bounds -----*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Conservative interval analysis over the SSA graph that produces the
// `positions` operand of a resampling op.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_ANALYSIS_DISPLACEMENTRANGE_H
#define SAR_ANALYSIS_DISPLACEMENTRANGE_H

#include "mlir/IR/Value.h"

#include <optional>

namespace mlir {
namespace sar {

/// A closed interval of values, or the unbounded interval.
///
/// `unbounded()` is the top element: any operation whose result cannot be
/// constrained yields it, and it absorbs everything it is combined with.
struct Interval {
  double lo = 0.0;
  double hi = 0.0;
  bool bounded = false;

  static Interval unbounded() { return Interval{0.0, 0.0, false}; }
  static Interval point(double v) { return Interval{v, v, true}; }
  static Interval range(double lo, double hi) { return Interval{lo, hi, true}; }

  /// Union hull, used to merge the arms of a select.
  Interval join(const Interval &other) const;

  Interval operator+(const Interval &rhs) const;
  Interval operator-(const Interval &rhs) const;
  Interval operator*(const Interval &rhs) const;
  Interval operator/(const Interval &rhs) const;
};

/// `positions` decomposed as `coeff * j + offset`, where `j` is the index
/// along the resampled axis and `offset` is an interval covering everything
/// else. Displacement `positions - j` is bounded exactly when `coeff == 1`
/// and `offset` is bounded, in which case the displacement interval is
/// `offset` itself.
struct AffineInterval {
  /// Coefficient of the identity ramp along the resampled axis.
  double coeff = 0.0;
  /// Everything that does not vary as the identity ramp.
  Interval offset;

  static AffineInterval constant(const Interval &v) {
    return AffineInterval{0.0, v};
  }
  static AffineInterval unbounded() {
    return AffineInterval{0.0, Interval::unbounded()};
  }

  bool isBounded() const { return offset.bounded; }
};

/// Computes a conservative interval on `positions[i, j] - j`, the
/// displacement the resampler applies along axis `dim` of `positions`.
///
/// Returns `std::nullopt` when no finite bound can be proven -- an
/// unrecognized producer, a kernel argument, or a position field that is not
/// an identity ramp plus a bounded perturbation.
std::optional<Interval> computeDisplacementRange(Value positions, int64_t dim);

} // namespace sar
} // namespace mlir

#endif // SAR_ANALYSIS_DISPLACEMENTRANGE_H
