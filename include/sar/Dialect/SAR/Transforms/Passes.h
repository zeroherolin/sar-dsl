//===- Passes.h - SAR dialect transformation passes --------------*- C++-*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_SAR_TRANSFORMS_PASSES_H
#define SAR_DIALECT_SAR_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncDialect;
} // namespace func

namespace sar {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "sar/Dialect/SAR/Transforms/Passes.h.inc"

} // namespace sar
} // namespace mlir

#endif // SAR_DIALECT_SAR_TRANSFORMS_PASSES_H
