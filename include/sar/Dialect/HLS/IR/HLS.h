//===- HLS.h - the HLS dialect ----------------------------------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_IR_HLS_H
#define SAR_DIALECT_HLS_IR_HLS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "sar/Dialect/HLS/IR/HLSOpsDialect.h.inc"
#include "sar/Dialect/HLS/IR/HLSOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "sar/Dialect/HLS/IR/HLSOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "sar/Dialect/HLS/IR/HLSOpsAttributes.h.inc"

namespace mlir {
namespace sar {
namespace hls {

#include "sar/Dialect/HLS/IR/HLSOpsInterfaces.h.inc"

/// Kind of dataflow.node operands.
enum class OperandKind { INPUT, OUTPUT, PARAM };

//===----------------------------------------------------------------------===//
// Tile layout attribute utils.
//===----------------------------------------------------------------------===//

TileLayoutAttr getTileLayout(Operation *op);
void setTileLayout(Operation *op, TileLayoutAttr tileLayout);

TileLayoutAttr getTileLayout(Value memref);
void setTileLayout(Value memref, TileLayoutAttr tileLayout);
void setTileLayout(Value memref, ArrayRef<int64_t> tileShape,
                   ArrayRef<int64_t> vectorShape);
void setTileLayout(Value memref, ArrayRef<int64_t> tileShape);

//===----------------------------------------------------------------------===//
// HLS directive attributes
//===----------------------------------------------------------------------===//

/// Loop directives attribute utils.
LoopDirectiveAttr getLoopDirective(Operation *op);
void setLoopDirective(Operation *op, LoopDirectiveAttr loopDirective);
void setLoopDirective(Operation *op, bool pipeline, int64_t targetII);

/// Parallel and point loop attribute utils.
bool hasParallelAttr(Operation *op);
void setParallelAttr(Operation *op);
bool hasPointAttr(Operation *op);
void setPointAttr(Operation *op);

/// Function directives attribute utils.
FuncDirectiveAttr getFuncDirective(Operation *op);
void setFuncDirective(Operation *op, FuncDirectiveAttr funcDirective);
void setFuncDirective(Operation *op, bool pipeline, int64_t targetInterval,
                      bool dataflow);

/// Top and runtime function attribute utils.
bool hasTopFuncAttr(Operation *op);
void setTopFuncAttr(Operation *op);
bool hasRuntimeAttr(Operation *op);
void setRuntimeAttr(Operation *op);

class NodeOp;

} // namespace hls
} // namespace sar
} // namespace mlir

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class DataflowBufferLike : public TraitBase<ConcreteType, DataflowBufferLike> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    if (op->getNumResults() != 1 ||
        !isa<sar::hls::StreamType, MemRefType>(op->getResult(0).getType()))
      return failure();
    return success();
  }
};

} // namespace OpTrait
} // namespace mlir

#define GET_OP_CLASSES
#include "sar/Dialect/HLS/IR/HLSOps.h.inc"

#endif // SAR_DIALECT_HLS_IR_HLS_H
