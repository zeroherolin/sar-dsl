//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_TRANSFORMS_UTILS_H
#define SAR_DIALECT_HLS_TRANSFORMS_UTILS_H

#include "sar/Dialect/HLS/IR/Utils.h"

namespace mlir {
namespace sar {

using namespace hls;

/// Apply loop perfection. Try to sink all operations between loop statements
/// into the innermost loop of the input loop band.
bool applyAffineLoopPerfection(AffineLoopBand &band);

/// Optimize loop order. Loops associated with memory access dependencies are
/// moved to an as outer as possible location of the input loop band. If
/// "reverse" is true, as inner as possible.
bool applyAffineLoopOrderOpt(AffineLoopBand &band,
                             ArrayRef<unsigned> permMap = {},
                             bool reverse = false);

/// Try to rectangularize the input band.
bool applyRemoveVariableBound(AffineLoopBand &band);

/// Apply loop tiling to the input loop band and sink all intra-tile loops to
/// the innermost loop with the original loop order.
bool applyLoopTiling(AffineLoopBand &band, FactorList tileList,
                     bool loopNormalize = true, bool annotatePointLoop = true);

/// Apply loop pipelining to the pipelineLoc of the input loop band, all inner
/// loops are automatically fully unrolled.
bool applyLoopPipelining(AffineLoopBand &band, unsigned pipelineLoc,
                         unsigned targetII);

/// Apply unroll and jam to the loop band with the given overall unroll factor.
bool applyLoopUnrollJam(AffineLoopBand &band, unsigned unrollFactor);

/// Apply unroll and jam to the loop band with the given unroll factors.
bool applyLoopUnrollJam(AffineLoopBand &band, FactorList unrollFactors);

/// Fully unroll all loops insides of a loop block.
bool applyFullyLoopUnrolling(Block &block, unsigned maxIterNum = 10);

/// Applies `factors`/`kinds` to `array`. A bank holding fewer than
/// `lutramMaxBits` bits is placed in distributed RAM, so long as
/// `lutramBitsBudget` (0 = unbounded) still has room; `lutramBitsUsed`
/// carries the running total across every array in a design.
bool applyArrayPartition(Value array, ArrayRef<unsigned> factors,
                         ArrayRef<hls::PartitionKind> kinds,
                         bool updateFuncSignature = true,
                         unsigned lutramMaxBits = 1024,
                         uint64_t lutramBitsBudget = 0,
                         uint64_t *lutramBitsUsed = nullptr);

/// Find the suitable array partition factors and kinds for all arrays in the
/// targeted function.
bool applyAutoArrayPartition(func::FuncOp func, unsigned lutramMaxBits = 1024,
                             unsigned maxFactor = 32,
                             uint64_t lutramBitsBudget = 0,
                             uint64_t *lutramBitsUsed = nullptr);

bool applyFuncPreprocess(func::FuncOp func, bool topFunc);

} // namespace sar
} // namespace mlir

#endif // SAR_DIALECT_HLS_TRANSFORMS_UTILS_H
