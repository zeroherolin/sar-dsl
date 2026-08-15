//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#ifndef SCALEHLS_TRANSFORMS_PASSES_TD
#define SCALEHLS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"



def FuncPreprocess : Pass<"hls-func-preprocess", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Preprocess the functions subsequent ScaleHLS optimizations";
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

  let options = [
    Option<"splitExternalAccess", "split-external-access", "bool",
           /*default=*/"true", "whether split external memory accesses">
  ];
}

def ParallelizeDataflowNode :
      Pass<"hls-parallelize-dataflow-node", "func::FuncOp"> {
  let summary = "Unroll affine loop nests based on the dataflow structure";
  let description = [{
    This pass calculates the overall loop unroll factor of each dataflow node
    based on the amount of associated computations. Then, unroll and jam from
    the outermost loop until the overall unroll factor reaches the caculated
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
  let summary = "Place dataflow buffers";
  let constructor = "mlir::sar::createPlaceDataflowBufferPass()";

  let options = [
    Option<"threshold", "threshold", "unsigned", /*default=*/"1024",
           "Positive number: the threshold of placing external buffers">,
    Option<"placeExternalBuffer", "place-external-buffer", "bool",
           /*default=*/"true", "Place buffers in external buffers">
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
// Tensor-related Passes
//===----------------------------------------------------------------------===//






//===----------------------------------------------------------------------===//
// Loop-related Passes
//===----------------------------------------------------------------------===//

def AffineLoopFusion : Pass<"hls-affine-loop-fusion", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Fuse affine loop nests";
  let description = [{
    This pass performs fusion of loop nests using a slicing-based approach. It
    combines two fusion strategies: producer-consumer fusion and sibling fusion.
    Producer-consumer fusion is aimed at fusing pairs of loops where the first
    one writes to a memref that the second reads. Sibling fusion targets pairs
    of loops that share no dependences between them but that load from the same
    memref. The fused loop nests, when possible, are rewritten to access
    significantly smaller local buffers instead of the original memref's, and
    the latter are often either completely optimized away or contracted. This
    transformation leads to enhanced locality and lower memory footprint through
    the elimination or contraction of temporaries/intermediate memref's. These
    benefits are sometimes achieved at the expense of redundant computation
    through a cost model that evaluates available choices such as the depth at
    which a source slice should be materialized in the designation slice.
  }];
  let constructor = "mlir::sar::createAffineLoopFusionPass()";

  let options = [
    Option<"computeToleranceThreshold", "fusion-compute-tolerance", "double",
           /*default=*/"0.30f", "Fractional increase in additional computation "
                                "tolerated while fusing">,
    Option<"fastMemorySpace", "fusion-fast-mem-space", "unsigned",
           /*default=*/"0",
           "Faster memory space number to promote fusion buffers to">,
    Option<"localBufSizeThreshold", "fusion-local-buf-threshold", "uint64_t",
           /*default=*/"0", "Threshold size (KiB) for promoting local buffers "
                            "to fast memory space">,
    Option<"maximalFusion", "fusion-maximal", "bool", /*default=*/"false",
           "Enables maximal loop fusion">,
    Option<"affineFusionMode", "mode", "enum AffineFusionMode",
           "AffineFusionMode::Greedy", "fusion mode to attempt",
           "llvm::cl::values(clEnumValN(AffineFusionMode::Greedy,"
           " \"greedy\", \"Perform greedy fusion\"), "
           "clEnumValN( AffineFusionMode::ProducerConsumer, "
           "\"producer\", \"Perform only producer-consumer fusion\"), "
           "clEnumValN( AffineFusionMode::Sibling, "
           "\"sibling\", \"Perform only sibling fusion\"))">
    ];
}

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
    This pass is similar to scal-repl, but supports to forward stores located in
    if statements.
  }];
  let constructor = "mlir::sar::createAffineStoreForwardPass()";
}


def CollapseMemrefUnitDims :
      Pass<"hls-collapse-memref-unit-dims", "func::FuncOp"> {
  let summary = "Collapse memref's unit dimensions";
  let constructor = "mlir::sar::createCollapseMemrefUnitDimsPass()";
}


def CreateLocalBuffer : Pass<"hls-create-local-buffer", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Promote external buffer to on-chip buffer";
  let constructor = "mlir::sar::createCreateLocalBufferPass()";

  let options = [
    Option<"externalBufferOnly", "external-buffer-only", "bool",
           /*default=*/"true", "only handle external buffers">,
    Option<"registerOnly", "register-only", "bool",
           /*default=*/"false", "only registers or single-element buffers">
  ];
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
  let summary = "Try to reduce the intiail interval";
  let description = [{
    This pass try to reduce the II by optimizing the commutative operator trees
    and iteration variables.
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
    Option<"maxFactor", "max-factor", "unsigned", /*default=*/"32",
           "Positive number: the largest array partition factor to apply">
  ];
}

def CreateHLSPrimitive : Pass<"hls-create-hls-primitive", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect", "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "mlir::bufferization::BufferizationDialect"];

  let summary = "Create HLS C++ multiplification primitives";
  let description = [{
    This pass will convert 8-bits multiplifications to HLS C++ primitives in
    order to utilize DSP instances in FPGA.
  }];
  let constructor = "mlir::sar::createCreateHLSPrimitivePass()";
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
           "The top function for HLS synthesis">
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



#endif // SCALEHLS_TRANSFORMS_PASSES_TD

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


