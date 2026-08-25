//===- Utils.h - HLS IR helpers ---------------------------------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_IR_UTILS_H
#define SAR_DIALECT_HLS_IR_UTILS_H

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "sar/Dialect/HLS/IR/HLS.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
namespace sar {

using AffineLoopBand = SmallVector<affine::AffineForOp, 6>;
using AffineLoopBands = std::vector<AffineLoopBand>;
using FactorList = SmallVector<unsigned, 8>;

//===----------------------------------------------------------------------===//
// HLS attribute utils
//===----------------------------------------------------------------------===//

hls::MemoryKind getMemoryKind(MemRefType type);

//===----------------------------------------------------------------------===//
// Dataflow utils
//===----------------------------------------------------------------------===//

/// Wrap the operations in the block with dispatch op.
hls::DispatchOp dispatchBlock(Block *block);

/// Fuse the given operations into a new task. The new task will be created
/// before the first operation or last operation and each operation will be
/// inserted in order. This method always succeeds even if the resulting IR is
/// invalid.
hls::TaskOp fuseOpsIntoTask(ArrayRef<Operation *> ops,
                            PatternRewriter &rewriter);

/// Fuse multiple nodes into a new node.
hls::NodeOp fuseNodeOps(ArrayRef<hls::NodeOp> nodes, PatternRewriter &rewriter);

/// Get the consumer/producer nodes of the given buffer except the given op.
SmallVector<hls::NodeOp> getConsumersExcept(Value buffer, hls::NodeOp except);
SmallVector<hls::NodeOp> getProducersExcept(Value buffer, hls::NodeOp except);
SmallVector<hls::NodeOp> getProducers(Value buffer);
SmallVector<hls::NodeOp> getDependentConsumers(Value buffer, hls::NodeOp node);

/// Get the depth of a buffer or stream channel. Note that only if the defining
/// operation of the buffer is not a hls::BufferOp or stream types, the returned
/// result will be 1.
unsigned getBufferDepth(Value memref);

/// Find buffer value or buffer op across the dataflow hierarchy.
Value findBuffer(Value memref);

bool isExtBuffer(Value memref);

/// Check whether the given use has read/write semantics.
bool isRead(OpOperand &use);
bool isWritten(OpOperand &use);

//===----------------------------------------------------------------------===//
// Memory and loop analysis utils
//===----------------------------------------------------------------------===//

/// Return a pair which indicates whether the if statement is always true or
/// false, respectively. The returned result is one-hot.
std::pair<bool, bool> ifAlwaysTrueOrFalse(mlir::affine::AffineIfOp ifOp);

/// Check whether the two given if statements have the same condition.
bool checkSameIfStatement(affine::AffineIfOp lhsOp, affine::AffineIfOp rhsOp);

/// Parse array attributes.
SmallVector<int64_t, 8> getIntArrayAttrValue(Operation *op, StringRef name);

/// For storing all affine memory access operations (including
/// affine::AffineLoadOp, and affine::AffineStoreOp) indexed by the
/// corresponding memref. A MapVector, so iteration follows the program order
/// the block was walked in rather than pointer values -- consumers allocate
/// budgets while iterating, and the outcome has to be the same on every run.
using MemAccessesMap = llvm::MapVector<Value, SmallVector<Operation *, 16>>;

/// Collect all load and store operations in the block and return them in "map".
void getMemAccessesMap(Block &block, MemAccessesMap &map,
                       bool includeVectorTransfer = false);

bool crossRegionDominates(Operation *a, Operation *b);

/// Calculate the upper and lower bound of the affine map if possible.
std::optional<std::pair<int64_t, int64_t>>
getBoundOfAffineMap(AffineMap map, ValueRange operands);

/// Calculate partition factors through analyzing the "memrefType" and return
/// them in "factors". Meanwhile, the overall partition number is calculated and
/// returned as well.
int64_t getPartitionFactors(MemRefType memrefType,
                            SmallVectorImpl<int64_t> *factors = nullptr);

bool isFullyPartitioned(MemRefType memrefType);

/// Whether `operation` is the sole access in a zero-based, unit-step affine
/// loop nest that visits every element of `memref` once in row-major order.
bool isCompleteRowMajorSweep(Operation *operation, Value memref);

/// Whether `operation` is nested in any loop-like operation in its function.
bool isNestedInLoop(Operation *operation);

func::FuncOp getTopFunc(ModuleOp module, std::string topFuncName = "");

} // namespace sar
} // namespace mlir

#endif // SAR_DIALECT_HLS_IR_UTILS_H
