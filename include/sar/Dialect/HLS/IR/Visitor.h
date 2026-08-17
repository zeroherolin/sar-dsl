//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_IR_VISITOR_H
#define SAR_DIALECT_HLS_IR_VISITOR_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace sar {

using namespace hls;

/// This class is a visitor for SSACFG operation nodes.
template <typename ConcreteType, typename ResultType, typename... ExtraArgs>
class HLSVisitorBase {
public:
  ResultType dispatchVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // HLS dialect operations.
            BufferOp, ConstBufferOp, StreamOp, StreamReadOp, StreamWriteOp,
            AxiBundleOp, AxiPortOp, AxiPackOp, hls::AffineSelectOp,

            // Function operations.
            func::CallOp, func::ReturnOp,

            // SCF statements.
            scf::ForOp, scf::IfOp, scf::ParallelOp, scf::ReduceOp,
            scf::ReduceReturnOp, scf::YieldOp,

            // Affine statements.
            affine::AffineForOp, affine::AffineIfOp, affine::AffineParallelOp,
            affine::AffineApplyOp, affine::AffineMaxOp, affine::AffineMinOp,
            affine::AffineLoadOp, affine::AffineStoreOp,
            affine::AffineVectorLoadOp, affine::AffineVectorStoreOp,
            affine::AffineYieldOp,

            // Vector statements.
            vector::InsertOp, vector::ExtractOp, vector::TransferReadOp,
            vector::TransferWriteOp, vector::BroadcastOp,

            // Memref statements.
            memref::AllocOp, memref::AllocaOp, memref::LoadOp, memref::StoreOp,
            memref::DeallocOp, memref::CopyOp,

            // Unary expressions.
            math::AbsIOp, math::AbsFOp, math::CeilOp, math::CosOp, math::SinOp,
            math::TanhOp, math::SqrtOp, math::RsqrtOp, math::ExpOp,
            math::Exp2Op, math::LogOp, math::Log2Op, math::Log10Op,
            arith::NegFOp,

            // Float binary expressions.
            arith::CmpFOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,
            arith::DivFOp, arith::RemFOp, arith::MaximumFOp, arith::MinimumFOp,
            math::PowFOp, math::Atan2Op,

            // Integer binary expressions.
            arith::CmpIOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
            arith::DivSIOp, arith::RemSIOp, arith::DivUIOp, arith::RemUIOp,
            arith::XOrIOp, arith::AndIOp, arith::OrIOp, arith::ShLIOp,
            arith::ShRSIOp, arith::ShRUIOp, arith::MaxSIOp, arith::MinSIOp,
            arith::MaxUIOp, arith::MinUIOp,

            // Special expressions.
            arith::SelectOp, arith::ConstantOp, arith::TruncIOp,
            arith::TruncFOp, arith::ExtUIOp, arith::ExtSIOp, arith::ExtFOp,
            arith::IndexCastOp, arith::UIToFPOp, arith::SIToFPOp,
            arith::FPToSIOp, arith::FPToUIOp>([&](auto opNode) -> ResultType {
          return thisCast->visitOp(opNode, args...);
        })
        .Default([&](auto opNode) -> ResultType {
          return thisCast->visitInvalidOp(op, args...);
        });
  }

  /// This callback is invoked on operations no visitor lists at all. It
  /// reports "not mine" like visitUnhandledOp does: the dispatcher that
  /// tried every visitor is the one that knows the op is unsupported, and
  /// its diagnostic is the one that carries a failure state.
  ResultType visitInvalidOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

  /// This callback is invoked on any operations that are not handled by the
  /// concrete visitor.
  ResultType visitUnhandledOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitOp(OPTYPE op, ExtraArgs... args) {                           \
    return static_cast<ConcreteType *>(this)->visitUnhandledOp(op, args...);   \
  }

  // HLS dialect operations.
  HANDLE(BufferOp);
  HANDLE(ConstBufferOp);
  HANDLE(StreamOp);
  HANDLE(StreamReadOp);
  HANDLE(StreamWriteOp);
  HANDLE(AxiBundleOp);
  HANDLE(AxiPortOp);
  HANDLE(AxiPackOp);
  HANDLE(hls::AffineSelectOp);

  // Control flow operations.
  HANDLE(func::CallOp);
  HANDLE(func::ReturnOp);

  // SCF statements.
  HANDLE(scf::ForOp);
  HANDLE(scf::IfOp);
  HANDLE(scf::ParallelOp);
  HANDLE(scf::ReduceOp);
  HANDLE(scf::ReduceReturnOp);
  HANDLE(scf::YieldOp);

  // Affine statements.
  HANDLE(affine::AffineForOp);
  HANDLE(affine::AffineIfOp);
  HANDLE(affine::AffineParallelOp);
  HANDLE(affine::AffineApplyOp);
  HANDLE(affine::AffineMaxOp);
  HANDLE(affine::AffineMinOp);
  HANDLE(affine::AffineLoadOp);
  HANDLE(affine::AffineStoreOp);
  HANDLE(affine::AffineVectorLoadOp);
  HANDLE(affine::AffineVectorStoreOp);
  HANDLE(affine::AffineYieldOp);

  // Vector statements.
  HANDLE(vector::InsertOp);
  HANDLE(vector::ExtractOp);
  HANDLE(vector::TransferReadOp);
  HANDLE(vector::TransferWriteOp);
  HANDLE(vector::BroadcastOp);

  // Memref statements.
  HANDLE(memref::AllocOp);
  HANDLE(memref::AllocaOp);
  HANDLE(memref::LoadOp);
  HANDLE(memref::StoreOp);
  HANDLE(memref::DeallocOp);
  HANDLE(memref::CopyOp);

  // Unary expressions.
  HANDLE(math::AbsIOp);
  HANDLE(math::AbsFOp);
  HANDLE(math::CeilOp);
  HANDLE(math::CosOp);
  HANDLE(math::SinOp);
  HANDLE(math::TanhOp);
  HANDLE(math::SqrtOp);
  HANDLE(math::RsqrtOp);
  HANDLE(math::ExpOp);
  HANDLE(math::Exp2Op);
  HANDLE(math::LogOp);
  HANDLE(math::Log2Op);
  HANDLE(math::Log10Op);
  HANDLE(arith::NegFOp);

  // Float binary expressions.
  HANDLE(arith::CmpFOp);
  HANDLE(arith::AddFOp);
  HANDLE(arith::SubFOp);
  HANDLE(arith::MulFOp);
  HANDLE(arith::DivFOp);
  HANDLE(arith::RemFOp);
  HANDLE(arith::MaximumFOp);
  HANDLE(arith::MinimumFOp);
  HANDLE(math::PowFOp);
  HANDLE(math::Atan2Op);

  // Integer binary expressions.
  HANDLE(arith::CmpIOp);
  HANDLE(arith::AddIOp);
  HANDLE(arith::SubIOp);
  HANDLE(arith::MulIOp);
  HANDLE(arith::DivSIOp);
  HANDLE(arith::RemSIOp);
  HANDLE(arith::DivUIOp);
  HANDLE(arith::RemUIOp);
  HANDLE(arith::XOrIOp);
  HANDLE(arith::AndIOp);
  HANDLE(arith::OrIOp);
  HANDLE(arith::ShLIOp);
  HANDLE(arith::ShRSIOp);
  HANDLE(arith::ShRUIOp);
  HANDLE(arith::MaxSIOp);
  HANDLE(arith::MinSIOp);
  HANDLE(arith::MaxUIOp);
  HANDLE(arith::MinUIOp);

  // Special expressions.
  HANDLE(arith::SelectOp);
  HANDLE(arith::ConstantOp);
  HANDLE(arith::TruncIOp);
  HANDLE(arith::TruncFOp);
  HANDLE(arith::ExtUIOp);
  HANDLE(arith::ExtSIOp);
  HANDLE(arith::ExtFOp);
  HANDLE(arith::IndexCastOp);
  HANDLE(arith::UIToFPOp);
  HANDLE(arith::SIToFPOp);
  HANDLE(arith::FPToUIOp);
  HANDLE(arith::FPToSIOp);
#undef HANDLE
};
} // namespace sar
} // namespace mlir

#endif // SAR_DIALECT_HLS_IR_VISITOR_H
