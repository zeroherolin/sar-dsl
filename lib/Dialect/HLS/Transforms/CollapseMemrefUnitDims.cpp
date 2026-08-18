//===- CollapseMemrefUnitDims.cpp - collapse memref unit dims -------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_COLLAPSEMEMREFUNITDIMS
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

static LogicalResult collapseMemref(Value memref) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type)
    return failure();

  // Only identity-layout buffers with a unit dim qualify: a non-identity
  // layout (partitioned or tiled) encodes banking whose meaning would
  // change if a dimension vanished from under it.
  if (!type.getLayout().getAffineMap().isIdentity() ||
      !llvm::any_of(type.getShape(),
                    [](int64_t dimSize) { return dimSize == 1; }))
    return failure();

  // Every user's access map must be rewritable; only the affine
  // read/write interfaces expose one.
  if (llvm::any_of(memref.getUsers(), [](Operation *op) {
        return !isa<AffineReadOpInterface, AffineWriteOpInterface>(op);
      }))
    return failure();

  // Construct new shape.
  SmallVector<int64_t> newShape;
  SmallVector<unsigned> remainDims;
  for (auto dimSize : llvm::enumerate(type.getShape()))
    if (dimSize.value() != 1) {
      newShape.push_back(dimSize.value());
      remainDims.push_back(dimSize.index());
    }
  // A rank-0 memref emits as a scalar C++ value. Dataflow extraction may
  // later pass it to a helper by value, losing writes to the channel.
  if (newShape.empty())
    return failure();

  // Set the buffer to a new type.
  auto newType = MemRefType::get(newShape, type.getElementType(), AffineMap(),
                                 type.getMemorySpace());
  memref.setType(newType);

  // Update buffer users.
  for (auto user : memref.getUsers()) {
    AffineMap map;
    if (auto read = dyn_cast<mlir::affine::AffineReadOpInterface>(user))
      map = read.getAffineMap();
    else if (auto write = dyn_cast<mlir::affine::AffineWriteOpInterface>(user))
      map = write.getAffineMap();

    SmallVector<AffineExpr> newResults;
    for (auto dim : remainDims)
      newResults.push_back(map.getResult(dim));
    auto newMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                 newResults, map.getContext());
    // Both affine.load and affine.store spell their map attribute this way;
    // take the name from the op rather than hardcoding it.
    user->setAttr(affine::AffineLoadOp::getMapAttrStrName(),
                  AffineMapAttr::get(newMap));
  }

  // Update tile layout - remove the collapsed dimensions.
  if (auto layout = getTileLayout(memref)) {
    SmallVector<int64_t> newTileShape;
    SmallVector<int64_t> newVectorShape;

    for (auto dim : remainDims) {
      newTileShape.push_back(layout.getTileShape()[dim]);
      newVectorShape.push_back(layout.getVectorShape()[dim]);
    }
    setTileLayout(memref, newTileShape, newVectorShape);
  }
  return success();
}

namespace {
struct CollapseFuncMemref : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter &rewriter) const override {
    bool hasChanged = false;
    func.walk([&](hls::BufferLikeInterface buffer) {
      if (succeeded(collapseMemref(buffer.getMemref()))) {
        hasChanged = true;
        if (auto constBuffer = dyn_cast<ConstBufferOp>(buffer.getOperation())) {
          auto memrefType = buffer.getMemrefType();
          auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                                  memrefType.getElementType());

          SmallVector<Attribute> attrs;
          for (auto attr : constBuffer.getValue().getValues<Attribute>())
            attrs.push_back(attr);
          constBuffer.setValueAttr(DenseElementsAttr::get(tensorType, attrs));
        }
      }
    });

    func.setType(rewriter.getFunctionType(
        func.front().getArgumentTypes(),
        func.front().getTerminator()->getOperandTypes()));
    return success(hasChanged);
  }
};
} // namespace

namespace {
struct CollapseMemrefUnitDims
    : public sar::impl::CollapseMemrefUnitDimsBase<CollapseMemrefUnitDims> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<CollapseFuncMemref>(context);
    (void)applyOpPatternsGreedily(ArrayRef<Operation *>{func.getOperation()},
                                  std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> sar::createCollapseMemrefUnitDimsPass() {
  return std::make_unique<CollapseMemrefUnitDims>();
}
