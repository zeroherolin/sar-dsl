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

  let summary = "Normalize function operations and annotate HLS loop traits";
  let description = [{
    Marks the selected top function, refreshes affine-loop parallel traits and
    removes stale loop directives. With `rewrite-ops=true` it also normalizes
    memref accesses, index arithmetic, vector transfers and allocations. The
    HLS pipeline uses the full mode once and the annotation-only mode after
    copy/placement transformations expose additional loops.
  }];
  let constructor = "mlir::sar::createFuncPreprocessPass()";

  let options = [
    Option<"topFunc", "top-func", "std::string", /*default=*/"\"main\"",
           "The top function for HLS synthesis">,
    Option<"rewriteOps", "rewrite-ops", "bool", /*default=*/"true",
           "Normalize memory/index/vector operations in addition to loop "
           "annotation">
  ];
}

//===----------------------------------------------------------------------===//
// Dataflow construction and legalization
//===----------------------------------------------------------------------===//

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
    This pass inserts explicit fork nodes for on-chip channels and for
    read-only on-chip arguments with two or more consumers. Vitis HLS treats
    each m_axi reader as a dataflow process, so even a two-reader argument
    cannot remain a shared pointer; constant ROM tables are the exception.
  }];
  let constructor = "mlir::sar::createEliminateMultiConsumerPass()";
}

def ScheduleDataflowNode :
      Pass<"hls-schedule-dataflow-node", "func::FuncOp"> {
  let summary = "Schedule dataflow nodes";
  let constructor = "mlir::sar::createScheduleDataflowNodePass()";

  let options = [
    Option<"ignoreViolations", "ignore-violations", "bool",
           /*default=*/"false", "Ignore multi-producer or multi-consumer "
           "violations">
  ];
}


def BalanceDataflowNode :
      Pass<"hls-balance-dataflow-node", "func::FuncOp"> {
  let summary = "Balance dataflow nodes";
  let description = [{
    Carries a value produced several levels above its consumer through a
    chain of one-level copies. Without them the schedule has a bypass, and
    the legalization that follows closes a bypass by fusing the levels it
    spans -- which merges independent stages into one function and lengthens
    its critical path.
  }];
  let constructor = "mlir::sar::createBalanceDataflowNodePass()";
  // The balancing copies are memref.copy ops.
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
    URAM overflow demotes to block RAM while preserving the external buffer
    set and its ports from the first placement.
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
    Option<"bramBlockBytes", "bram-block-bytes", "unsigned",
           /*default=*/"4608", "Bytes in one block-RAM primitive">,
    Option<"uramBlockBytes", "uram-block-bytes", "unsigned",
           /*default=*/"36864", "Bytes in one UltraRAM primitive (0 when "
           "the target has no UltraRAM)">,
    Option<"allowDram", "allow-dram", "bool", /*default=*/"true",
           "Permit buffers that do not fit on chip to spill to external "
           "DRAM; disable for ap_memory interfaces">,
    Option<"rebalanceOnly", "rebalance-only", "bool", /*default=*/"false",
           "Re-place an already-scheduled graph: recharge tiers, demote "
           "URAM overflow to BRAM while preserving the established DRAM set">
  ];
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

def ConvertDataflowToFunc :
      Pass<"hls-convert-dataflow-to-func", "ModuleOp"> {
  let summary = "Convert dataflow to function dialect";
  let constructor = "mlir::sar::createConvertDataflowToFuncPass()";

  // Creates hls dataflow ops (and copies through memref).
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];
}

//===----------------------------------------------------------------------===//
// Loop transformations
//===----------------------------------------------------------------------===//

def AffineLoopOrderOpt :
      Pass<"hls-affine-loop-order-opt", "func::FuncOp"> {
  let summary = "Move loop-carried affine dimensions outward when legal";
  let description = [{
    Examines perfect affine loop bands and moves dimensions carrying memory
    dependences as far outward as legality permits. The resulting inner loops
    have a larger recurrence distance and are better candidates for HLS
    pipelining. Fully parallel or non-perfect bands are left unchanged.
  }];
  let constructor = "mlir::sar::createAffineLoopOrderOptPass()";
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

def FuseSiblingLoops :
      Pass<"hls-fuse-sibling-loops", "func::FuncOp"> {
  let summary = "Fuse independent affine sweeps with identical iteration";
  let description = [{
    Merges adjacent perfect affine loop nests when they have identical bounds
    and no cross-loop memory dependence beyond a pointwise producer/consumer
    pair whose store and load indices agree. Shared read-only expressions then
    become common subexpressions, allowing split real/imaginary phase outputs
    to share trigonometric and coordinate arithmetic. With
    `preserve-dataflow-roles`, a public-result writer is kept separate from a
    producer that still feeds an internal successor, matching Vitis dataflow's
    canonical process roles.
  }];
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];
  let constructor = "mlir::sar::createFuseSiblingLoopsPass()";
  let options = [
    Option<"preserveDataflowRoles", "preserve-dataflow-roles", "bool",
           /*default=*/"false",
           "Keep caller-output writers separate from internal producers with "
           "successors">
  ];
}

def ReduceInitiationInterval :
      Pass<"hls-reduce-initiation-interval", "func::FuncOp"> {
  let summary = "Shorten exact integer loop-carried operator chains";
  let description = [{
    Reconnects associative, commutative integer/index chains so loop-invariant
    heads are evaluated outside the carried recurrence. Floating-point chains
    are deliberately excluded because reassociation would change rounding,
    signed-zero and NaN behavior.
  }];
  let constructor = "mlir::sar::createReduceInitiationIntervalPass()";
}

//===----------------------------------------------------------------------===//
// Memory and copy simplification
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
    This pass is similar to scalar replacement, but also forwards stores
    located in if statements.
  }];
  let constructor = "mlir::sar::createAffineStoreForwardPass()";
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

def LowerCopyToAffine : Pass<"hls-lower-copy-to-affine", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

  let summary = "Convert memref copy ops to affine loops";
  let constructor = "mlir::sar::createLowerCopyToAffinePass()";

  let options = [
    Option<"internalCopyOnly", "internal-copy-only", "bool",
           /*default=*/"false", "Only convert copies between internal buffers">
  ];
}

//===----------------------------------------------------------------------===//
// Interface and directive generation
//===----------------------------------------------------------------------===//

def CreateAxiInterface : Pass<"hls-create-axi-interface", "ModuleOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

  let summary = "Build stable AXI ports and typed scratch arenas";
  let description = [{
    Builds the synthesizable top-level interface for the selected function.
    Public arrays become AXI memory-mapped or AXI4-Stream ports according to
    the requested protocol. Internal DRAM buffers are flattened into a bounded
    set of typed scratch arenas, colored so simultaneous reads and writes use
    independent masters where the interface budget permits. A runtime-only
    `main` wrapper retains the host-facing memref signature; the implementation
    function carries the HLS AXI types consumed by the C++ emitter.
  }];
  let constructor = "mlir::sar::createCreateAxiInterfacePass()";

  let options = [
    Option<"topFunc", "top-func", "std::string", /*default=*/"\"main\"",
           "The top function for HLS synthesis">,
    Option<"streamInterface", "stream-interface", "bool", /*default=*/"false",
           "Bundle top-level ports as AXI4-Stream (axis) rather than AXI4 "
           "memory-mapped masters">,
    Option<"maxScratchArenas", "max-scratch-arenas", "unsigned",
           /*default=*/"4",
           "Maximum compiler-owned scratch masters per scalar type">
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
           "Minimum logical elements before a port is worth packing">,
    Option<"packPublicOutputs", "pack-public-outputs", "bool",
           /*default=*/"true",
           "Permit public output arrays to use a packed physical ABI">,
    Option<"computeMaxLanes", "compute-max-lanes", "unsigned",
           /*default=*/"0",
           "Maximum unrolled compute lanes per packed transfer (0 = all)">
  ];
}

def ArrayPartition : Pass<"hls-array-partition", "ModuleOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::vector::VectorDialect"];

  let summary = "Bank on-chip arrays within final memory-tier budgets";
  let description = [{
    Derives cyclic or complete banking from the concurrent affine accesses to
    each on-chip array, applies explicit producer hints where present, and
    caps the factor at `max-factor`. Banking is costed in whole memory
    primitives; BRAM and URAM assignments are rebalanced before their hard
    budgets are checked, and only sufficiently small affordable banks move to
    distributed RAM. External arrays and arrays participating in subviews keep
    their existing layout.
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
    Option<"bramBlockBytes", "bram-block-bytes", "unsigned",
           /*default=*/"4608", "Bytes in one block-RAM primitive">,
    Option<"uramBlockBytes", "uram-block-bytes", "unsigned",
           /*default=*/"36864", "Bytes in one UltraRAM primitive (0 when "
           "the target has no UltraRAM)">,
    Option<"maxFactor", "max-factor", "unsigned", /*default=*/"32",
           "Positive number: the largest array partition factor to apply">
  ];
}

def LoopPipelining : Pass<"hls-loop-pipelining", "func::FuncOp"> {
  let dependentDialects = [
    "mlir::sar::hls::HLSDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"];

  let summary = "Pipeline affine loop bands with bounded inner unrolling";
  let description = [{
    Attaches an HLS pipeline directive to the selected level of each affine
    loop band. Inner loops are fully unrolled only when the estimated cloned
    operation count fits `max-unrolled-ops`; otherwise they receive a bounded
    power-of-two unroll factor no larger than `max-unroll-factor`. Per-loop
    minimum-II attributes take precedence over the requested target.
  }];
  let constructor = "mlir::sar::createLoopPipeliningPass()";

  let options = [
    Option<"pipelineLevel", "pipeline-level", "unsigned", /*default=*/"0",
           "Loop level to be pipelined (0 = innermost)">,
    Option<"targetII", "target-ii", "unsigned", /*default=*/"1",
           "Positive number: the targeted II to achieve">,
    Option<"maxUnrolledOps", "max-unrolled-ops", "unsigned",
           /*default=*/"4096",
           "Maximum estimated operation copies materialized by full unroll">,
    Option<"maxUnrollFactor", "max-unroll-factor", "unsigned",
           /*default=*/"32",
           "Maximum pragma factor used when full unroll exceeds the budget">
  ];
}

//===----------------------------------------------------------------------===//
// Module-level sharing
//===----------------------------------------------------------------------===//

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
    Redirects calls to one canonical helper when their memory bindings are
    compatible. Repeated hierarchical helpers are marked for one Vitis
    function instance; leaf helpers remain inline so stage-level dataflow is
    not serialized.
  }];
  let constructor = "mlir::sar::createShareEquivalentFunctionsPass()";
}

#endif // SAR_DIALECT_HLS_TRANSFORMS_PASSES_TD
