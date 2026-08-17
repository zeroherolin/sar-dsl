//===- Passes.td - HLS dialect transformation passes -------*- tablegen -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_TRANSFORMS_PASSES_TD
#define SAR_DIALECT_HLS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def FuncPreprocess : Pass<"hls-func-preprocess", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Preprocess functions for the HLS pipeline";
  let constructor = "mlir::sar::createFuncPreprocessPass()";

  let options = [
    Option<"topFunc", "top-func", "std::string", /*default=*/"\"main\"",
           "The top function for HLS synthesis">
  ];
}

//===----------------------------------------------------------------------===//
// Dataflow-related Passes
//===----------------------------------------------------------------------===//

def BalanceDataflowNode :
      Pass<"hls-balance-dataflow-node", "func::FuncOp"> {
  let summary = "Balance dataflow nodes";
  let constructor = "mlir::sar::createBalanceDataflowNodePass()";
}

def ConvertDataflowToFunc :
      Pass<"hls-convert-dataflow-to-func", "ModuleOp"> {
  let summary = "Convert dataflow to function dialect";
  let constructor = "mlir::sar::createConvertDataflowToFuncPass()";

  let options = [
    Option<"splitExternalAccess", "split-external-access", "bool",
           /*default=*/"true", "whether split external memory accesses">
  ];
}

def CreateDataflowFromAffine :
      Pass<"hls-create-dataflow-from-affine", "func::FuncOp"> {
  let summary = "Create dataflow hierarchy from affine loops";
  let constructor = "mlir::sar::createCreateDataflowFromAffinePass()";
}

def CreateTokenStream : Pass<"hls-create-token-stream", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Create token stream channels for DRAM buffers";
  let constructor = "mlir::sar::createCreateTokenStreamPass()";
}

def EliminateMultiConsumer :
      Pass<"hls-eliminate-multi-consumer", "func::FuncOp"> {
  let summary = "Eliminate multi-consumer violations";
  let description = [{
    This pass will insert explicit fork nodes to eliminate multi-consumers.
  }];
  let constructor = "mlir::sar::createEliminateMultiConsumerPass()";
}

def EliminateMultiProducer :
      Pass<"hls-eliminate-multi-producer", "func::FuncOp"> {
  let summary = "Try to eliminate multi-producer violations";
  let constructor = "mlir::sar::createEliminateMultiProducerPass()";
}

def LegalizeDataflow : Pass<"hls-legalize-dataflow", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Legalize dataflow by merging dataflow nodes";
  let constructor = "mlir::sar::createLegalizeDataflowPass()";
}

def LowerDataflow : Pass<"hls-lower-dataflow", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Lower dataflow from task level to node level";
  let constructor = "mlir::sar::createLowerDataflowPass()";
}

def ParallelizeDataflowNode :
      Pass<"hls-parallelize-dataflow-node", "func::FuncOp"> {
  let summary = "Unroll affine loop nests based on the dataflow structure";
  let description = [{
    This pass calculates the overall loop unroll factor of each dataflow node
    based on the amount of associated computations. Then, unroll and jam from
    the outermost loop until the overall unroll factor reaches the calculated
    factor. Optionally, optimize the loop order after the unrolling.
  }];
  let constructor = "mlir::sar::createParallelizeDataflowNodePass()";

  let options = [
    Option<"maxUnrollFactor", "max-unroll-factor", "unsigned", /*default=*/"1",
           "Positive number: the maximum factor of unrolling">,
    Option<"pointLoopOnly", "point-loop-only", "bool", /*default=*/"false",
           "Only apply unroll and jam to point loop band">,
    Option<"complexityAware", "complexity-aware", "bool", /*default=*/"true",
           "Whether to consider node complexity in the transform">,
    Option<"correlationAware", "correlation-aware", "bool", /*default=*/"true",
           "Whether to consider node correlation in the transform">
  ];
}

def PlaceDataflowBuffer :
      Pass<"hls-place-dataflow-buffer", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Place dataflow buffers";
  let description = [{
    Bind each buffer to an on-chip memory tier, or to DRAM when no tier has
    room. A buffer's tier follows its measured size: one small enough to
    sit in distributed RAM without competing with the logic goes to LUTRAM,
    a contiguous plane large enough to fill ultra RAM blocks goes to URAM,
    the rest go to block RAM. Tiers are charged in whole primitives -- a
    36Kb block holding one kilobyte is spent -- so the budgets cannot
    over-promise. A tier budget of 0 means that tier is unbounded; the
    lutram-max/uram-min thresholds steer between tiers either way.

    Buffers at or above `threshold` elements move to DRAM -- that is what
    lets scene size grow past the device -- except constant buffers, which
    are the ROM tables the design needs on entry and never stream.
    Function arguments are placed too and count against the same budgets:
    an interface array occupies the same primitives as a local buffer.
  }];
  let constructor = "mlir::sar::createPlaceDataflowBufferPass()";

  let options = [
    Option<"threshold", "threshold", "unsigned", /*default=*/"1024",
           "Element count at or above which a buffer is placed in DRAM">,
    Option<"bramBytes", "bram-bytes", "unsigned", /*default=*/"0",
           "Block RAM the design may occupy, in bytes (0 is unbounded)">,
    Option<"uramBytes", "uram-bytes", "unsigned", /*default=*/"0",
           "Ultra RAM the design may occupy, in bytes (0 is unbounded)">,
    Option<"lutramBytes", "lutram-bytes", "unsigned", /*default=*/"0",
           "Distributed RAM the design may occupy, in bytes (0 is "
           "unbounded)">,
    Option<"lutramMaxBytes", "lutram-max-bytes", "unsigned",
           /*default=*/"256",
           "Buffer size, in bytes, at or below which distributed RAM is "
           "preferred">,
    Option<"uramMinBytes", "uram-min-bytes", "unsigned", /*default=*/"36864",
           "Buffer size, in bytes, at or above which ultra RAM is preferred">
  ];
}

def ScheduleDataflowNode :
      Pass<"hls-schedule-dataflow-node", "func::FuncOp"> {
  let summary = "Schedule dataflow nodes";
  let constructor = "mlir::sar::createScheduleDataflowNodePass()";

  let options = [
    Option<"ignoreViolations", "ignore-violations", "bool",
           /*default=*/"false", "Ignore multi-consumer or producer violations">
  ];
}

def StreamDataflowTask : Pass<"hls-stream-dataflow-task", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Stream dataflow tasks";
  let constructor = "mlir::sar::createStreamDataflowTaskPass()";
}

//===----------------------------------------------------------------------===//
// Loop-related Passes
//===----------------------------------------------------------------------===//

def AffineLoopOrderOpt :
      Pass<"hls-affine-loop-order-opt", "func::FuncOp"> {
  let summary = "Optimize the order of affine loop nests";
  let description = [{
    This pass will optimize the order of perfect affine loop nests through
    polyhedral-based dependency analysis. The objective of this pass is to
    maximize the distance of loop-carried dependencies.
  }];
  let constructor = "mlir::sar::createAffineLoopOrderOptPass()";
}

def AffineLoopPerfection :
      Pass<"hls-affine-loop-perfection", "func::FuncOp"> {
  let summary = "Try to perfect a nested loop";
  let description = [{
    This pass will try to perfect all affine loops. Specifically, this pass will
    move operations under non-innermost loops into innermost loop and create if
    statements to ensure the correctness of the transformation.
  }];
  let constructor = "mlir::sar::createAffineLoopPerfectionPass()";
}

def AffineLoopTile : Pass<"hls-affine-loop-tile", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Tile affine loop nests and annotate point loops";
  let description = [{
    Tile affine loop nests, choosing per dimension whether a tile can pay.
    A dimension carrying no reuse, or the one every external access sweeps
    contiguously, is kept whole so its sweep stays one long AXI burst. A
    band whose accesses disagree on layout -- a transpose -- is tiled on
    every dimension, with an edge sized to "tile-buffer-bytes".
  }];
  let constructor = "mlir::sar::createAffineLoopTilePass()";

  let options = [
    Option<"tileSize", "tile-size", "unsigned", /*default=*/"1",
           "Use this tile size for all loops">,
    Option<"tileBufferBytes", "tile-buffer-bytes", "unsigned",
           /*default=*/"524288",
           "On-chip bytes a transposing band may stage per block">
  ];
}

def RemoveVariableBound :
      Pass<"hls-remove-variable-bound", "func::FuncOp"> {
  let summary = "Try to remove variable loop bounds";
  let description = [{
    This pass will try to remove the variable loop bounds. Specifically, this is
    only feasible when the loop bound is an affine expression of induction
    variables of other loops with constant lower and upper bound.
  }];
  let constructor = "mlir::sar::createRemoveVariableBoundPass()";
}

//===----------------------------------------------------------------------===//
// Memory-related Passes
//===----------------------------------------------------------------------===//

def AffineStoreForward : Pass<"hls-affine-store-forward", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Forward store to load, including conditional stores";
  let description = [{
    This pass is similar to scal-repl, but supports forwarding stores located
    in if statements.
  }];
  let constructor = "mlir::sar::createAffineStoreForwardPass()";
}

def CollapseMemrefUnitDims :
      Pass<"hls-collapse-memref-unit-dims", "func::FuncOp"> {
  let summary = "Collapse memref's unit dimensions";
  let constructor = "mlir::sar::createCollapseMemrefUnitDimsPass()";
}

def LowerCopyToAffine : Pass<"hls-lower-copy-to-affine", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Convert copy and assign to affine loops";
  let constructor = "mlir::sar::createLowerCopyToAffinePass()";

  let options = [
    Option<"internalCopyOnly", "internal-copy-only", "bool",
           /*default=*/"false", "only convert copy between internal buffers">
  ];
}

def RaiseAffineToCopy : Pass<"hls-raise-affine-to-copy", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Raise copy in affine loops to memref.copy";
  let constructor = "mlir::sar::createRaiseAffineToCopyPass()";
}

def ReduceInitialInterval :
      Pass<"hls-reduce-initial-interval", "func::FuncOp"> {
  let summary = "Try to reduce the initiation interval";
  let description = [{
    This pass tries to reduce the II by optimizing the commutative operator
    trees and iteration variables.
  }];
  let constructor = "mlir::sar::createReduceInitialIntervalPass()";
}

def SimplifyAffineIf : Pass<"hls-simplify-affine-if", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Simplify affine if operations";
  let description = [{
    This pass will eliminate all redundant affine if operations which will
    always take then or else branch, and merge all affine if operations that
    having the same statement.
  }];
  let constructor = "mlir::sar::createSimplifyAffineIfPass()";
}

def SimplifyCopy : Pass<"hls-simplify-copy", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Simplify memref copy ops";
  let constructor = "mlir::sar::createSimplifyCopyPass()";
}

//===----------------------------------------------------------------------===//
// Directive-related Passes
//===----------------------------------------------------------------------===//

def ArrayPartition : Pass<"hls-array-partition", "ModuleOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Apply optimized array partition strategy";
  let description = [{
    This pass will automatically search for the best array partition solution
    for each on-chip memory instance and apply the solution through changing the
    layout of the corresponding memref.
  }];
  let constructor = "mlir::sar::createArrayPartitionPass()";

  let options = [
    Option<"lutramMaxBits", "lutram-max-bits", "unsigned",
           /*default=*/"1024",
           "Bank size, in bits, below which distributed RAM is used">,
    Option<"lutramBytes", "lutram-bytes", "unsigned", /*default=*/"0",
           "Distributed RAM the design may occupy, in bytes (0 is "
           "unbounded). Banking stops promoting into it once spent, so a "
           "bank that would overrun the budget keeps the tier placement "
           "already chose for it">,
    Option<"maxFactor", "max-factor", "unsigned", /*default=*/"32",
           "Positive number: the largest array partition factor to apply">
  ];
}

def CreateAxiInterface : Pass<"hls-create-axi-interface", "ModuleOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Create AXI interfaces for the top function";
  let description = [{
    This pass will create a new "main" function calling the original top
    function. All constant tensors are instantiated in the new "main" function
    and passed into the original top function as arguments after the transform.
  }];
  let constructor = "mlir::sar::createCreateAxiInterfacePass()";

  let options = [
    Option<"topFunc", "top-func", "std::string", /*default=*/"\"main\"",
           "The top function for HLS synthesis">,
    Option<"streamInterface", "stream-interface", "bool", /*default=*/"false",
           "Bundle top-level ports as AXI4-Stream (axis) rather than AXI4 "
           "memory-mapped masters">
  ];
}

def LoopPipelining : Pass<"hls-loop-pipelining", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Apply loop pipelining";
  let description = [{
    This pass will insert pipeline pragma to the target loop level, and
    automatically unroll all inner loops.
  }];
  let constructor = "mlir::sar::createLoopPipeliningPass()";

  let options = [
    Option<"pipelineLevel", "pipeline-level", "unsigned", /*default=*/"0",
           "Positive number: loop level to be pipelined (from innermost)">,
    Option<"targetII", "target-ii", "unsigned", /*default=*/"1",
           "Positive number: the targeted II to achieve">
  ];
}

def CreateMemrefSubview :
      Pass<"hls-create-memref-subview", "func::FuncOp"> {
  let summary = "Create subviews based on loop analysis";
  let description = [{
    Through loop analysis, this pass can identify the memory partition that each
    sub-function is accessing. Then, by creating subview operations, the program
    in each sub-function can access the memory subview rather than the original
    memory.
  }];
  let constructor = "mlir::sar::createCreateMemrefSubviewPass()";

  let options = [
    Option<"createSubviewMode", "mode", "enum CreateSubviewMode",
           /*default=*/"CreateSubviewMode::Point",
           "loop band mode to create subviews", "llvm::cl::values("
           "clEnumValN(CreateSubviewMode::Point, \"point\", "
           "\"Create subviews on point loop band\"), "
           "clEnumValN(CreateSubviewMode::Reduction, \"reduction\", "
           "\"Create subviews on reduction loop band\"))">
  ];
}

#endif // SAR_DIALECT_HLS_TRANSFORMS_PASSES_TD
