//===- Utils.h - HLS transform helpers --------------------------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_TRANSFORMS_UTILS_H
#define SAR_DIALECT_HLS_TRANSFORMS_UTILS_H

#include "sar/Dialect/HLS/IR/Utils.h"

namespace mlir {
namespace sar {

/// Given a tiled loop band, return true and get the tile (tile-space) loop
/// band and the point (intra-tile) loop band. If failed, return false.
bool getTileAndPointLoopBand(const AffineLoopBand &band,
                             AffineLoopBand &tileBand,
                             AffineLoopBand &pointBand);

/// Get the whole loop band given the outermost or innermost loop and return it
/// in "band". Meanwhile, the return value is the innermost or outermost loop of
/// this loop band.
affine::AffineForOp getLoopBandFromOutermost(affine::AffineForOp forOp,
                                             AffineLoopBand &band);
affine::AffineForOp getLoopBandFromInnermost(affine::AffineForOp forOp,
                                             AffineLoopBand &band);

/// Collect all loop bands in the "block" and return them in "bands". If
/// "allowHavingChilds" is true, loop bands containing more than 1 other loop
/// bands are also collected. Otherwise, only loop bands that contains no child
/// loops are collected.
void getLoopBands(Block &block, AffineLoopBands &bands,
                  bool allowHavingChilds = false);

/// Rewrites the buffer forms a dataflow node cannot carry across its
/// boundary. Defined in this library, not the dialect one.
void populateBufferConversionPatterns(RewritePatternSet &patterns);

/// Optimize loop order. Loops associated with memory access dependencies are
/// moved to an as outer as possible location of the input loop band. If
/// "reverse" is true, as inner as possible.
bool applyAffineLoopOrderOpt(AffineLoopBand &band,
                             ArrayRef<unsigned> permMap = {},
                             bool reverse = false);

/// Apply loop tiling to the input loop band and sink all intra-tile loops to
/// the innermost loop with the original loop order.
bool applyLoopTiling(AffineLoopBand &band, FactorList tileList,
                     bool loopNormalize = true, bool annotatePointLoop = true);

/// Apply loop pipelining to the pipelineLoc of the input loop band, all inner
/// loops are automatically fully unrolled.
bool applyLoopPipelining(AffineLoopBand &band, unsigned pipelineLoc,
                         unsigned targetII);

/// Applies `factors`/`kinds` to `array`. A bank holding at most
/// `lutramMaxBits` bits is placed in distributed RAM, so long as
/// `lutramBitsBudget` (0 = unbounded) still has room; `lutramBitsUsed`
/// carries the running total across every array in a design. The bound is
/// inclusive because it is one bus beat, and a bank holding exactly one
/// beat is the canonical distributed-RAM case.
bool applyArrayPartition(Value array, ArrayRef<unsigned> factors,
                         ArrayRef<hls::PartitionKind> kinds,
                         bool updateFuncSignature = true,
                         unsigned lutramMaxBits = 512,
                         uint64_t lutramBitsBudget = 0,
                         uint64_t *lutramBitsUsed = nullptr);

/// Find the suitable array partition factors and kinds for all arrays in the
/// targeted function.
bool applyAutoArrayPartition(func::FuncOp func, unsigned lutramMaxBits = 512,
                             unsigned maxFactor = 32,
                             uint64_t lutramBitsBudget = 0,
                             uint64_t *lutramBitsUsed = nullptr);

LogicalResult applyFuncPreprocess(func::FuncOp func, bool topFunc,
                                  bool rewriteOps = true);

} // namespace sar
} // namespace mlir

#endif // SAR_DIALECT_HLS_TRANSFORMS_UTILS_H
