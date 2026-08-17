//===- RuntimeEnums.h - Interpolation enums shared with the runtime ------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
// The integer encodings of `sar.interp1d`'s `kernel`, `window` and
// `boundary` attributes
// as passed across the runtime C ABI. Included by both the lowering
// (lib/Conversion/SARSignalToRuntime) and the runtime implementation
// (runtime/SARRuntime.cpp), which is dependency-free C++ -- so this header
// stays plain C++ with no MLIR includes.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_RUNTIME_RUNTIMEENUMS_H
#define SAR_RUNTIME_RUNTIMEENUMS_H

#include <cstdint>

namespace sar_rt {

enum InterpKernel : int64_t {
  kNearest = 0,
  kLinear = 1,
  kCubic = 2,
  kSinc = 3,
};

enum InterpWindow : int64_t {
  kRect = 0,
  kHann = 1,
  kHamming = 2,
  kKaiser = 3,
};

enum InterpBoundary : int64_t {
  kZero = 0,
  kEdge = 1,
  kReflect = 2,
};

} // namespace sar_rt

#endif // SAR_RUNTIME_RUNTIMEENUMS_H
