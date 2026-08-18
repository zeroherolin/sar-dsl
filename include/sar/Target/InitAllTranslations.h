//===- InitAllTranslations.h - Register SAR translations --------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// One place listing every target SAR can translate out to. A new target
// adds its registration here and nothing else changes.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_TARGET_INITALLTRANSLATIONS_H
#define SAR_TARGET_INITALLTRANSLATIONS_H

#include "sar/Target/HLS/EmitHLSCpp.h"
#include "sar/Target/KernelFacts.h"

namespace mlir {
namespace sar {

inline void registerAllTranslations() {
  registerEmitHLSCppTranslation();
  registerKernelFactsTranslation();
}

} // namespace sar
} // namespace mlir

#endif // SAR_TARGET_INITALLTRANSLATIONS_H
