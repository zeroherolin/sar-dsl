//===- HLSInterfaces.td - HLS dialect interfaces --------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_IR_HLSINTERFACES_TD
#define SAR_DIALECT_HLS_IR_HLSINTERFACES_TD

include "mlir/IR/OpBase.td"

def BufferLikeInterface : OpInterface<"BufferLikeInterface"> {
  let description = [{
    Models an operation that owns a dataflow buffer.
  }];

  let methods = [
    InterfaceMethod<"Returns the memref result of the buffer",
      "mlir::Value", "getMemref", (ins), [{ return $_op.getMemref(); }]
    >,
    InterfaceMethod<"Returns the memref type of the buffer",
      "mlir::MemRefType", "getMemrefType", (ins), [{
        return llvm::cast<mlir::MemRefType>($_op.getMemref().getType());
      }]
    >,
    InterfaceMethod<"Returns the depth of the buffer", "int32_t",
                    "getBufferDepth">,
    InterfaceMethod<"Returns the initial value of the buffer",
                    "std::optional<mlir::TypedAttr>", "getBufferInitValue">,
  ];
}

def StageLikeInterface : OpInterface<"StageLikeInterface"> {
  let description = [{
    Models an operation that owns a dataflow stage.
  }];

  let methods = [
    InterfaceMethod<"Returns the body region of the stage",
      "mlir::Region &", "getBody", (ins), "return $_op.getBody();">,
    InterfaceMethod<"Returns whether the stage contains another stage",
      "bool", "hasHierarchy", (ins), [{
        return $_op.walk([&](StageLikeInterface stage) {
          if (stage != $_op)
            return WalkResult::interrupt();
          return WalkResult::advance();
        }).wasInterrupted();
      }]>,
  ];
}

#endif // SAR_DIALECT_HLS_IR_HLSINTERFACES_TD
