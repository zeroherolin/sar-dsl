//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_CREATELOCALBUFFER
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir


using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

namespace {
struct CreateLocalBuffer
    : public sar::impl::CreateLocalBufferBase<CreateLocalBuffer> {
  CreateLocalBuffer() = default;
  CreateLocalBuffer(bool argExternalBufferOnly, bool argRegisterOnly) {
    externalBufferOnly = argExternalBufferOnly;
    registerOnly = argRegisterOnly;
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto builder = OpBuilder(func);

    func.walk([&](memref::SubViewOp subview) {
      if (externalBufferOnly && !isExtBuffer(subview.getSource()))
        return WalkResult::advance();

      if (registerOnly && subview.getType().getNumElements() != 1)
        return WalkResult::advance();

      // Check the read/write status of the memref.
      auto readFlag = llvm::any_of(subview->getUses(), isRead);
      auto writeFlag = llvm::any_of(subview->getUses(), isWritten);
      if (!readFlag && !writeFlag)
        return WalkResult::advance();

      // We strip the original layout map and memory kind when constructing the
      // local buffer's memref type.
      auto subviewType = subview.getType();
      auto bufType = MemRefType::get(
          subviewType.getShape(), subviewType.getElementType(), AffineMap(),
          MemoryKindAttr::get(subview.getContext(), MemoryKind::BRAM_T2P));

      // Allocate an on-chip buffer and replace all its uses.
      auto loc = builder.getUnknownLoc();
      builder.setInsertionPointAfter(subview);
      auto buf = builder.create<BufferOp>(loc, bufType);
      subview.getResult().replaceAllUsesWith(buf);

      // If the global buffer has initial value, set it to the local buffer.
      auto globalBuf = findBufferOp(subview.getSource());
      if (globalBuf && globalBuf.getBufferInitValue())
        buf.setInitValueAttr(globalBuf.getBufferInitValue().value());

      // Create explicit copy from/to the local buffer.
      if (readFlag)
        builder.create<memref::CopyOp>(loc, subview, buf);
      if (writeFlag) {
        builder.setInsertionPoint(subview->getBlock()->getTerminator());
        builder.create<memref::CopyOp>(loc, buf, subview);
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

std::unique_ptr<Pass>
sar::createCreateLocalBufferPass(bool externalBufferOnly,
                                      bool registerOnly) {
  return std::make_unique<CreateLocalBuffer>(externalBufferOnly, registerOnly);
}
