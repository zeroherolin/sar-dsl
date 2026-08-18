//===- HLSOps.td - HLS dialect op definitions -----------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_IR_HLSOPS_TD
#define SAR_DIALECT_HLS_IR_HLSOPS_TD

include "sar/Dialect/HLS/IR/HLSInterfaces.td"
include "sar/Dialect/HLS/IR/HLSTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

class HLSOp<string mnemonic, list<Trait> traits = []> :
    Op<HLSDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Traits and Type Constraints
//===----------------------------------------------------------------------===//

def DataflowBufferLike : NativeOpTrait<"DataflowBufferLike">;

class StreamOf<list<Type> allowedTypes> : Type<And<[
    CPred<"isa<mlir::sar::hls::StreamType>($_self)">,
    Concat<"[](mlir::Type elementType) { return ",
      SubstLeaves<"$_self", "elementType", AnyTypeOf<allowedTypes>.predicate>,
      "; }(cast<mlir::sar::hls::StreamType>($_self).getElementType())">
  ]>,
  "stream of " # AnyTypeOf<allowedTypes>.summary # " values",
  "mlir::sar::hls::StreamType"
>;

class AxiOf<list<Type> allowedTypes> : Type<And<[
    CPred<"isa<mlir::sar::hls::AxiType>($_self)">,
    Concat<"[](mlir::Type elementType) { return ",
      SubstLeaves<"$_self", "elementType", AnyTypeOf<allowedTypes>.predicate>,
      "; }(cast<mlir::sar::hls::AxiType>($_self).getElementType())">
  ]>,
  "axi of " # AnyTypeOf<allowedTypes>.summary # " values",
  "mlir::sar::hls::AxiType"
>;

def AnyBuffer : StaticShapeMemRefOf<[AnyType]>;
def AnyStream : StreamOf<[AnyType]>;
def AnyBufferOrStream : Type<Or<[AnyBuffer.predicate, AnyStream.predicate]>,
    "memref or stream values">;
def AnyAxi : AxiOf<[AnyType]>;

//===----------------------------------------------------------------------===//
// High Dataflow Operations
//===----------------------------------------------------------------------===//

def DispatchOp : HLSOp<"dataflow.dispatch", [RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"YieldOp">,
      ParentOneOf<["func::FuncOp", "affine::AffineForOp"]>]> {
  let summary = "Represent a dataflow dispatch";
  let description = [{
    Dispatch op has a transparent region that contains a list of task ops to be
    dispatched. This op is designed to organize and manipulate task ops at a
    high level and will be lowered to schedule op for dataflow scheduling.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword ( `:` type($results)^ )? $body";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Get the terminator yield op.
    YieldOp getYieldOp();
  }];
}

def TaskOp : HLSOp<"dataflow.task", [
      DeclareOpInterfaceMethods<StageLikeInterface>, RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"YieldOp">, HasParent<"DispatchOp">]> {
  let summary = "Represent a dataflow task";
  let description = [{
    Task op has a transparent region that contains a list of ops to be executed
    in the task. This op is designed to represent the task partition at a high
    level and will be lowered to node op for dataflow scheduling.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword ( `:` type($results)^ )? $body";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Get the parent dispatch op.
    DispatchOp getDispatchOp();

    /// Get the terminator yield op.
    YieldOp getYieldOp();

    bool hasHierarchy() {
      return cast<StageLikeInterface>(this->getOperation()).hasHierarchy();
    }
  }];
}

def YieldOp : HLSOp<"dataflow.yield", [NoMemoryEffect, ReturnLike, Terminator,
      ParentOneOf<["DispatchOp", "TaskOp"]>]> {
  let summary = "Terminate and yield results of a dispatch or task op";

  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "$results attr-dict `:` type($results)";

  let builders = [OpBuilder<(ins), "build($_builder, $_state, ::mlir::ValueRange());">];
}

def ToStreamOp : HLSOp<"dataflow.to_stream", [NoMemoryEffect]> {
  let summary = "Convert a scalar value to a stream channel";

  let arguments = (ins AnyType:$value);
  let results = (outs AnyStream:$stream);
  let assemblyFormat = "$value attr-dict `:` functional-type($value, $stream)";

  let hasVerifier = 1;
  let hasFolder = 1;
}

def ToValueOp : HLSOp<"dataflow.to_value", [NoMemoryEffect]> {
  let summary = "Convert a stream to a scalar value";

  let arguments = (ins AnyStream:$stream);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$stream attr-dict `:` functional-type($stream, $value)";

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Low Dataflow Operations
//===----------------------------------------------------------------------===//

def ScheduleOp : HLSOp<"dataflow.schedule", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>, IsolatedFromAbove,
      AffineScope, SingleBlock, NoTerminator,
      ParentOneOf<["func::FuncOp", "affine::AffineForOp"]>]> {
  let summary = "Represent a dataflow schedule";
  let description = [{
    Schedule op has an isolated region to contain a list of dataflow node ops to
    be scheduled. This op can be explicitly marked as legal when all the
    dataflow violations have been resolved and all the nodes has been scheduled.
  }];

  let arguments = (ins Variadic<AnyType>:$operands, UnitAttr:$isLegal);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    ( `legal` $isLegal^ )? ( `(` $operands^ `)` )? attr-dict-with-keyword
    ( `:` type($operands)^ )? $body
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Whether no execution of the schedule can depend on a previous one:
    /// true when the schedule runs once (function-level) or when every
    /// enclosing-loop iteration is independent (parallel-marked affine
    /// loop). Conservative -- any other position answers false.
    bool isDependenceFree();

    /// Update the signature of the schedule op recursively.
    void updateSignatureRecursively();
  }];
}

def NodeOp : HLSOp<"dataflow.node", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      DeclareOpInterfaceMethods<StageLikeInterface>, IsolatedFromAbove,
      AffineScope, SingleBlock, NoTerminator, AttrSizedOperandSegments,
      HasParent<"ScheduleOp">]> {
  let summary = "Represent a dataflow node";
  let description = [{
    A node owns an isolated region and accepts only buffers or streams as data
    inputs and outputs. Scalar parameters may enter the region but do not
    participate in dataflow scheduling. `inputTaps` records the hierarchy level
    of each input buffer or stream tap.
  }];

  let arguments = (ins Variadic<AnyBufferOrStream>:$inputs,
      Variadic<AnyBufferOrStream>:$outputs, Variadic<AnyType>:$params,
      I32ArrayAttr:$inputTaps, OptionalAttr<I32Attr>:$level);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(` $inputs `)` `->` `(` $outputs `)` ( `[` $params^ `]` )? attr-dict `:`
    functional-type($inputs, $outputs) ( `[` type($params)^ `]` )? $body
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs,
        "mlir::ValueRange":$params, "ArrayRef<unsigned>":$inputTaps,
        "mlir::IntegerAttr":$level), [{
          auto newInputTaps = SmallVector<int32_t>(
            llvm::map_range(inputTaps, [](unsigned a) { return (int32_t)a; }));
          build($_builder, $_state, inputs, outputs, params,
                $_builder.getI32ArrayAttr(newInputTaps), level);
        }]>,

    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs,
        "mlir::ValueRange":$params, "ArrayRef<unsigned>":$inputTaps), [{
          build($_builder, $_state, inputs, outputs, params, inputTaps, nullptr);
        }]>,
    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs,
        "ArrayRef<unsigned>":$inputTaps), [{
          build($_builder, $_state, inputs, outputs, ValueRange(), inputTaps);
        }]>,

    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs,
        "mlir::ValueRange":$params), [{
          build($_builder, $_state, inputs, outputs, params,
                SmallVector<unsigned>(inputs.size(), 0));
        }]>,
    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs),
        "build($_builder, $_state, inputs, outputs, ValueRange());">
  ];

  let extraClassDeclaration = [{
    /// Get the parent schedule op.
    ScheduleOp getScheduleOp();

    /// Get input taps.
    void setInputTap(unsigned idx, unsigned tap);
    unsigned getInputTap(unsigned idx);
    SmallVector<unsigned> getInputTapsAsInt();

    /// Get the number of inputs, outputs, and params.
    unsigned getNumInputs();
    unsigned getNumOutputs();
    unsigned getNumParams();

    /// Get the type of operand: input, output, or param.
    OperandKind getOperandKind(OpOperand &operand);
    OperandKind getOperandKind(unsigned operandIdx);

    /// Get the input, output, and param arguments.
    iterator_range<Block::args_iterator> getInputArgs();
    iterator_range<Block::args_iterator> getOutputArgs();
    iterator_range<Block::args_iterator> getParamArgs();

    bool hasHierarchy() {
      return cast<StageLikeInterface>(this->getOperation()).hasHierarchy();
    }

    /// Update the signature of the node op recursively.
    void updateSignatureRecursively();
  }];
}

def BufferOp : HLSOp<"dataflow.buffer", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      DeclareOpInterfaceMethods<BufferLikeInterface>]> {
  let summary = "Represent a dataflow buffer";

  let arguments = (ins I32Attr:$depth,
                       OptionalAttr<TypedAttrInterface>:$init_value);
  let results = (outs AnyBuffer:$memref);
  let assemblyFormat = "attr-dict `:` type($memref)";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let builders = [
    OpBuilder<(ins "mlir::Type":$memref),
        "build($_builder, $_state, memref, 1, nullptr);">
  ];
}

def ConstBufferOp : HLSOp<"dataflow.const_buffer", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      DeclareOpInterfaceMethods<BufferLikeInterface>]> {
  let summary = "Represent a constant dataflow buffer";

  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyBuffer:$memref);
  let assemblyFormat = "attr-dict `:` type($memref)";

  let hasVerifier = 1;
}

def StreamOp : HLSOp<"dataflow.stream", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Declare a stream channel";

  let arguments = (ins I32Attr:$depth);
  let results = (outs StreamOf<[AnyType]>:$channel);
  let assemblyFormat = "attr-dict `:` type($channel)";

  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "mlir::Type":$channel),
        "build($_builder, $_state, channel, 1);">
  ];
}

def StreamReadOp : HLSOp<"dataflow.stream_read", []> {
  let summary = "Read a stream channel";
  let description = [{
    Read/pop a value from a stream channel. The absence of the result indicates
    the popped value is dropped. Each stream channel can be read multiple times.
  }];

  let arguments = (ins StreamOf<[AnyType]>:$channel);
  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = [{
    $channel attr-dict `:` functional-type($channel, $result)
  }];

  let hasVerifier = 1;
}

def StreamWriteOp : HLSOp<"dataflow.stream_write", []> {
  let summary = "Write a stream channel";
  let description = [{
    Write/push a value to a stream channel. Each stream channel can only be
    written once.
  }];

  let arguments = (ins StreamOf<[AnyType]>:$channel, AnyType:$value);
  let assemblyFormat = [{
    $channel `,` $value attr-dict `:` type($channel) `,` type($value)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AXI-related Operations
//===----------------------------------------------------------------------===//

def AxiBundleOp : HLSOp<"axi.bundle", [NoMemoryEffect,
    HasParent<"func::FuncOp">]> {
  let summary = "Represent a bundle of AXI ports";

  let arguments = (ins StrAttr:$name);
  let results = (outs BundleType:$bundle);
  let assemblyFormat = "$name attr-dict `:` type($bundle)";
}

// Deliberately not `NoMemoryEffect`: a hardware interface declaration must
// survive DCE even when its placeholder arena is unused. Erasing it would
// leave a bare argument without the pragma required by the interface contract.
def AxiPortOp : HLSOp<"axi.port", [HasParent<"func::FuncOp">]> {
  let summary = "Represent an AXI port";

  let arguments = (ins BundleType:$bundle, AnyAxi:$axi);
  let results = (outs AnyType:$element);
  let assemblyFormat = [{
    $bundle `,` $axi attr-dict `:` type($bundle) `,`
    functional-type($axi, $element)
  }];

  let hasVerifier = 1;
  let extraClassDeclaration = [{
    AxiType getAxiType() {
      return cast<AxiType>(getAxi().getType());
    }
    BundleType getBundleType() {
      return cast<BundleType>(getBundle().getType());
    }
    StringRef getBundleName() {
      return getBundle().getDefiningOp<AxiBundleOp>().getName();
    }
  }];
}

def AxiPackOp : HLSOp<"axi.pack", [NoMemoryEffect, HasParent<"func::FuncOp">]> {
  let summary = "Pack a memref or scalar into an AXI type";

  let arguments = (ins AnyType:$element);
  let results = (outs AnyAxi:$axi);
  let assemblyFormat = "$element attr-dict `:` functional-type($element, $axi)";

  let hasVerifier = 1;
  let extraClassDeclaration = [{
    AxiType getAxiType() {
      return cast<AxiType>(getAxi().getType());
    }
  }];
}

def AffineSelectOp : HLSOp<"affine.select", [NoMemoryEffect]> {
  let summary = "if-then-else operation";

  let arguments = (ins Variadic<AnyType>:$args, AnyType:$true_value,
                       AnyType:$false_value);
  let results = (outs AnyType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "IntegerSet":$set, "ValueRange":$args, "Value":$trueValue,
                   "Value":$falseValue)>
  ];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    static StringRef getConditionAttrStrName() { return "condition"; }

    IntegerSet getIntegerSet();
    void setIntegerSet(IntegerSet newSet);

    /// Sets the integer set with its operands.
    void setConditional(IntegerSet set, ValueRange operands);
  }];
}

#endif // SAR_DIALECT_HLS_IR_HLSOPS_TD
