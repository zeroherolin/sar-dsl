//===- HLSHints.h - hint-attribute contract with the HLS backend -*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
// Discardable attributes through which a lowering hands the HLS passes
// knowledge that cannot be recovered from the IR by local analysis. The
// CPU validation pipeline ignores all of them, so a lowering may attach
// them unconditionally.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_SUPPORT_HLSHINTS_H
#define SAR_SUPPORT_HLSHINTS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sar {

/// On an `affine.for`: the loop is a compact stand-in for `factor` parallel
/// lanes. The emitter prints an unroll directive instead of the IR being
/// cloned, the pipelining pass pipelines the enclosing loop, and the loop
/// passes leave the surrounding band's shape alone.
constexpr llvm::StringLiteral kUnrollFactorAttr{"hls.unroll_factor"};

/// On a `memref.alloc` (carried onto the buffer it becomes): the banking
/// the access pattern was designed for, as parallel string/integer arrays
/// with one partition kind ("complete", "cyclic", "block" or "none") and
/// factor per dimension. The banking pass applies these verbatim and skips
/// its own search.
constexpr llvm::StringLiteral kPartitionKindsAttr{"hls.partition_kinds"};
constexpr llvm::StringLiteral kPartitionFactorsAttr{"hls.partition_factors"};

} // namespace sar
} // namespace mlir

#endif // SAR_SUPPORT_HLSHINTS_H
