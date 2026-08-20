//===- Passes.td - HLS transform pass definitions -------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_TRANSFORMS_PASSES_TD
#define SAR_DIALECT_HLS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def FuncPreprocess : Pass<"hls-func-preprocess", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::vector::VectorDialect",
    "mlir::linalg::LinalgDialect"];

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
  // The balancing copies are memref.copy ops.
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];
}

def ConvertDataflowToFunc :
      Pass<"hls-convert-dataflow-to-func", "ModuleOp"> {
  let summary = "Convert dataflow to function dialect";
  let constructor = "mlir::sar::createConvertDataflowToFuncPass()";

  let options = [
    Option<"splitExternalAccess", "split-external-access", "bool",
           /*default=*/"true", "whether split external memory accesses">
  ];
  // Creates hls dataflow ops (and copies through memref).
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];
}

def CreateDataflowFromAffine :
      Pass<"hls-create-dataflow-from-affine", "func::FuncOp"> {
  let summary = "Create dataflow hierarchy from affine loops";
  let constructor = "mlir::sar::createCreateDataflowFromAffinePass()";
  // Creates hls dataflow ops (and copies through memref).
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::scf::SCFDialect"];
}

def CreateTokenStream : Pass<"hls-create-token-stream", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

  let summary = "Create token stream channels for DRAM buffers";
  let constructor = "mlir::sar::createCreateTokenStreamPass()";
}

def EliminateMultiConsumer :
      Pass<"hls-eliminate-multi-consumer", "func::FuncOp"> {
  // Fork nodes copy buffers with memref.copy.
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

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
  // Producer privatization copies through memref ops.
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];
}

def LegalizeDataflow : Pass<"hls-legalize-dataflow", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

  let summary = "Merge dataflow nodes, then mark schedules dataflow-legal";
  let description = [{
    Fuses the node shapes a `#pragma HLS dataflow` region cannot express -- a
    multi-consumer buffer and a bypass path around a node -- and then marks
    every schedule whose nodes all carry a level and share no external buffer.
    That mark is what makes the emitter attach the pragma; a schedule left
    unmarked is emitted as ordinary sequential calls.
  }];
  let constructor = "mlir::sar::createLegalizeDataflowPass()";
}

def LowerDataflow : Pass<"hls-lower-dataflow", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::linalg::LinalgDialect"];

  let summary = "Lower dataflow from task level to node level";
  let constructor = "mlir::sar::createLowerDataflowPass()";
}

def PlaceDataflowBuffer :
      Pass<"hls-place-dataflow-buffer", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"];

  let summary = "Place dataflow buffers";
  let description = [{
    Bind each buffer to an on-chip memory tier, or to DRAM when no tier has
    room. A buffer's tier follows its measured size: one small enough to
    sit in distributed RAM without competing with the logic goes to LUTRAM,
    a contiguous plane at or above one ultra RAM block goes to URAM, the
    rest go to block RAM, and either block tier overflows into the other
    before anything spills. Tiers are charged in whole primitives -- a
    36Kb block holding one kilobyte is spent -- and the budgets are hard
    caps: a buffer that must stay on chip (a constant table, a
    per-iteration scratch) but fits no tier fails the pass rather than
    overcommitting the device.

    Buffers at or above `threshold` elements move to DRAM -- that is what
    lets scene size grow past the device -- except constant buffers, which
    are the ROM tables the design needs on entry and never stream.
    Function arguments are placed too and count against the same budgets:
    an interface array occupies the same primitives as a local buffer.

    With `rebalance-only` the pass re-runs on an already-placed scheduled
    graph (the fork/balance passes copy buffers at the placed type, past
    what the first placement budgeted): tier assignments are recharged and
    URAM overflow demotes to block RAM, but nothing newly moves to DRAM --
    the external set and its ports were fixed by the first placement.
  }];
  let constructor = "mlir::sar::createPlaceDataflowBufferPass()";

  let options = [
    Option<"threshold", "threshold", "unsigned", /*default=*/"1024",
           "Element count at or above which a buffer is placed in DRAM">,
    Option<"bramBytes", "bram-bytes", "unsigned", /*default=*/"9907200",
           "Block RAM the design may occupy, in bytes (hard cap; 0 forbids "
           "the tier)">,
    Option<"uramBytes", "uram-bytes", "unsigned", /*default=*/"37748736",
           "Ultra RAM the design may occupy, in bytes (hard cap; 0 forbids "
           "the tier)">,
    Option<"lutramBytes", "lutram-bytes", "unsigned", /*default=*/"2883584",
           "Distributed RAM the design may occupy, in bytes (hard cap; 0 "
           "forbids the tier)">,
    Option<"lutramMaxBytes", "lutram-max-bytes", "unsigned",
           /*default=*/"64",
           "Buffer size, in bytes, at or below which distributed RAM is "
           "preferred (one bus beat at the default 512-bit bus)">,
    Option<"allowDram", "allow-dram", "bool", /*default=*/"true",
           "Permit buffers that do not fit on chip to spill to external "
           "DRAM; disable for ap_memory interfaces">,
    Option<"rebalanceOnly", "rebalance-only", "bool", /*default=*/"false",
           "Re-place an already-scheduled graph: recharge tiers, demote "
           "URAM overflow to BRAM, never newly move a buffer to DRAM">
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
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

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
  // Buffers sunk results as hls buffers.
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];
}

def AffineLoopTile : Pass<"hls-affine-loop-tile", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

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
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

  let summary = "Forward store to load, including conditional stores";
  let description = [{
    This pass is similar to scal-repl, but supports forwarding stores located
    in if statements.
  }];
  let constructor = "mlir::sar::createAffineStoreForwardPass()";
}

def FuseSiblingLoops :
      Pass<"hls-fuse-sibling-loops", "func::FuncOp"> {
  let summary = "Fuse independent affine sweeps with identical iteration";
  let description = [{
    Merges adjacent perfect affine loop nests when they have identical bounds
    and no cross-loop memory dependence. Shared read-only expressions then
    become common subexpressions, allowing split real/imaginary phase outputs
    to share trigonometric and coordinate arithmetic.
  }];
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];
  let constructor = "mlir::sar::createFuseSiblingLoopsPass()";
}

def CollapseMemrefUnitDims :
      Pass<"hls-collapse-memref-unit-dims", "func::FuncOp"> {
  let summary = "Collapse memref's unit dimensions";
  let constructor = "mlir::sar::createCollapseMemrefUnitDimsPass()";
}

def LowerCopyToAffine : Pass<"hls-lower-copy-to-affine", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

  let summary = "Convert copy and assign to affine loops";
  let constructor = "mlir::sar::createLowerCopyToAffinePass()";

  let options = [
    Option<"internalCopyOnly", "internal-copy-only", "bool",
           /*default=*/"false", "only convert copy between internal buffers">
  ];
}

def RaiseAffineToCopy : Pass<"hls-raise-affine-to-copy", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

  let summary = "Raise copy in affine loops to memref.copy";
  let constructor = "mlir::sar::createRaiseAffineToCopyPass()";
}

def ReduceInitiationInterval :
      Pass<"hls-reduce-initiation-interval", "func::FuncOp"> {
  let summary = "Try to reduce the initiation interval";
  let description = [{
    This pass tries to reduce the II by optimizing the commutative operator
    trees and iteration variables.
  }];
  let constructor = "mlir::sar::createReduceInitiationIntervalPass()";
}

def SimplifyAffineIf : Pass<"hls-simplify-affine-if", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

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
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

  let summary = "Simplify memref copy ops";
  let constructor = "mlir::sar::createSimplifyCopyPass()";
}

def ShareEquivalentFunctions :
      Pass<"hls-share-equivalent-functions", "ModuleOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

  let summary = "Merge structurally equivalent HLS helper functions";
  let description = [{
    Redirects calls to one canonical helper after final type and banking
    decisions. Repeated hierarchical helpers are marked for a single Vitis
    function instance; leaf helpers remain inline so stage-level dataflow is
    not serialized.
  }];
  let constructor = "mlir::sar::createShareEquivalentFunctionsPass()";
}

//===----------------------------------------------------------------------===//
// Directive-related Passes
//===----------------------------------------------------------------------===//

def ArrayPartition : Pass<"hls-array-partition", "ModuleOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::vector::VectorDialect"];

  let summary = "Apply optimized array partition strategy";
  let description = [{
    This pass will automatically search for the best array partition solution
    for each on-chip memory instance and apply the solution through changing the
    layout of the corresponding memref.
  }];
  let constructor = "mlir::sar::createArrayPartitionPass()";

  let options = [
    Option<"lutramMaxBits", "lutram-max-bits", "unsigned",
           /*default=*/"512",
           "Bank size, in bits, at or below which distributed RAM is used "
           "(one bus beat at the default 512-bit bus)">,
    Option<"lutramBytes", "lutram-bytes", "unsigned", /*default=*/"0",
           "Distributed RAM the design may occupy, in bytes (0 forbids the "
           "tier). Banking stops promoting into it once spent, so a "
           "bank that would overrun the budget keeps the tier placement "
           "already chose for it">,
    Option<"bramBytes", "bram-bytes", "unsigned", /*default=*/"9907200",
           "Final block RAM budget in bytes, checked after banking">,
    Option<"uramBytes", "uram-bytes", "unsigned", /*default=*/"37748736",
           "Final UltraRAM budget in bytes, checked after banking">,
    Option<"maxFactor", "max-factor", "unsigned", /*default=*/"32",
           "Positive number: the largest array partition factor to apply">
  ];
}

def CreateAxiInterface : Pass<"hls-create-axi-interface", "ModuleOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

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

def WidenExternalMemory :
      Pass<"hls-widen-external-memory", "ModuleOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::vector::VectorDialect"];

  let summary = "Pack proven contiguous external accesses into wide beats";
  let description = [{
    Rewrites an AXI memref and its complete call chain to vector elements only
    when every access is a statically aligned, complete group along the
    fastest-varying dimension. Unsupported gathers, scatters, tails, layouts
    and aliases remain scalar.
  }];
  let constructor = "mlir::sar::createWidenExternalMemoryPass()";

  let options = [
    Option<"busBits", "bus-bits", "unsigned", /*default=*/"512",
           "Physical AXI data width used to choose the common lane count">,
    Option<"maxLanes", "max-lanes", "unsigned", /*default=*/"8",
           "Maximum power-of-two lanes in one packed element">,
    Option<"minElements", "min-elements", "unsigned", /*default=*/"4096",
           "Minimum logical elements before a port is worth packing">
  ];
}

def LoopPipelining : Pass<"hls-loop-pipelining", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

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

#endif // SAR_DIALECT_HLS_TRANSFORMS_PASSES_TD
