//===- Passes.h - SAR conversion passes --------------------------*- C++-*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_CONVERSION_PASSES_H
#define SAR_CONVERSION_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace affine {
class AffineDialect;
} // namespace affine
namespace arith {
class ArithDialect;
} // namespace arith
namespace bufferization {
class BufferizationDialect;
} // namespace bufferization
namespace complex {
class ComplexDialect;
} // namespace complex
namespace func {
class FuncDialect;
} // namespace func
namespace linalg {
class LinalgDialect;
} // namespace linalg
namespace math {
class MathDialect;
} // namespace math
namespace memref {
class MemRefDialect;
} // namespace memref
namespace tensor {
class TensorDialect;
} // namespace tensor

class RewritePatternSet;

namespace sar {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "sar/Conversion/Passes.h.inc"

/// Populates patterns lowering SAR structural / element-wise ops to linalg.
void populateSARToLinalgConversionPatterns(RewritePatternSet &patterns);

} // namespace sar
} // namespace mlir

#endif // SAR_CONVERSION_PASSES_H
