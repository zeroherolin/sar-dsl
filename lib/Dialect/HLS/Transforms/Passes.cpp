//===- Passes.cpp - HLS pipeline and pass registration --------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/SAR/Transforms/Passes.h"

using namespace mlir;
using namespace sar;
using namespace sar::hls;

namespace {
#define GEN_PASS_REGISTRATION
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace

void sar::addSimplifyAffineLoopPasses(OpPassManager &pm) {
  pm.addPass(affine::createAffineLoopNormalizePass());
  pm.addPass(affine::createSimplifyAffineStructuresPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

namespace {
struct HLSPipelineOptions : public PassPipelineOptions<HLSPipelineOptions> {
  Option<std::string> hlsTopFunc{
      *this, "top-func", llvm::cl::init("main"),
      llvm::cl::desc("Specify the top function of the design")};

  Option<unsigned> loopTileSize{
      *this, "loop-tile-size", llvm::cl::init(2),
      llvm::cl::desc("The tile size of each loop (must be at least 1)")};

  Option<unsigned> bramBytes{
      *this, "bram-bytes", llvm::cl::init(9907200),
      llvm::cl::desc("Block RAM the design may occupy, in bytes (hard cap; "
                     "0 forbids the tier)")};

  Option<unsigned> uramBytes{
      *this, "uram-bytes", llvm::cl::init(37748736),
      llvm::cl::desc("Ultra RAM the design may occupy, in bytes (hard cap; "
                     "0 forbids the tier)")};

  Option<unsigned> lutramBytes{
      *this, "lutram-bytes", llvm::cl::init(2883584),
      llvm::cl::desc("Distributed RAM the design may occupy, in bytes (hard "
                     "cap; 0 forbids the tier)")};

  Option<unsigned> lutramMaxBytes{
      *this, "lutram-max-bytes", llvm::cl::init(64),
      llvm::cl::desc("Bank size at or below which distributed RAM is used, "
                     "in bytes (one bus beat at the default 512-bit bus)")};

  Option<unsigned> arrayPartitionMaxFactor{
      *this, "array-partition-max-factor", llvm::cl::init(32),
      llvm::cl::desc("Largest factor automatic array banking may apply")};

  Option<unsigned> externalBufferThreshold{
      *this, "external-buffer-threshold", llvm::cl::init(1024),
      llvm::cl::desc("Element count at or above which a buffer moves to "
                     "DRAM (2^32-1 keeps everything resident)")};

  Option<bool> axiInterface{*this, "axi-interface", llvm::cl::init(true),
                            llvm::cl::desc("Create AXI interface")};

  Option<bool> streamInterface{
      *this, "stream-interface", llvm::cl::init(false),
      llvm::cl::desc("Emit axis (AXI4-Stream) pragmas on top-level ports "
                     "instead of m_axi ones; implies axi-interface=true")};

  Option<unsigned> axiBusBits{
      *this, "axi-bus-bits", llvm::cl::init(512),
      llvm::cl::desc("Physical AXI data width used for external packing")};

  Option<unsigned> externalVectorMaxLanes{
      *this, "external-vector-max-lanes", llvm::cl::init(8),
      llvm::cl::desc("Maximum lanes packed into one external word")};

  Option<unsigned> externalVectorMinElements{
      *this, "external-vector-min-elements", llvm::cl::init(4096),
      llvm::cl::desc("Minimum logical elements before packing an AXI port")};

  Option<bool> externalVectorPackOutputs{
      *this, "external-vector-pack-outputs", llvm::cl::init(true),
      llvm::cl::desc("Permit public output arrays to use a packed physical "
                     "ABI")};

  Option<unsigned> externalVectorComputeLanes{
      *this, "external-vector-compute-lanes", llvm::cl::init(0),
      llvm::cl::desc("Maximum unrolled compute lanes per packed transfer")};

  Option<unsigned> maxScratchArenas{
      *this, "max-scratch-arenas", llvm::cl::init(4),
      llvm::cl::desc("Maximum compiler-owned scratch masters per scalar type")};

  Option<unsigned> maxUnrolledOps{
      *this, "max-unrolled-ops", llvm::cl::init(4096),
      llvm::cl::desc("Maximum operation copies materialized by loop unroll")};

  Option<unsigned> maxUnrollFactor{
      *this, "max-unroll-factor", llvm::cl::init(32),
      llvm::cl::desc("Maximum factor used when full unroll exceeds the "
                     "budget")};
};
} // namespace

/// On-chip bytes one loop-tiling staging block may occupy: an eighth of
/// the total tier allowance, so staged blocks do not compete with the
/// working set they serve when several bands are in flight.
static unsigned getStagingBudget(const HLSPipelineOptions &opts) {
  uint64_t total = uint64_t(opts.bramBytes) + opts.uramBytes + opts.lutramBytes;
  return std::max<uint64_t>(1, total / 8);
}

void sar::registerHLSPipeline() {
  PassPipelineRegistration<HLSPipelineOptions>(
      "hls-pipeline", "Compile SAR affine IR to scheduled HLS IR",
      [](OpPassManager &pm, const HLSPipelineOptions &opts) {
        // Upstream affine fusion cannot model the data-dependent memref loads
        // in interpolation gathers and may reorder stores across them. Fusion
        // here requires gather-aware dependence analysis.
        pm.addPass(sar::createFuncPreprocessPass(opts.hlsTopFunc));
        sar::addSimplifyAffineLoopPasses(pm);
        pm.addPass(sar::createRaiseAffineToCopyPass());
        pm.addPass(sar::createSimplifyCopyPass());
        pm.addPass(sar::createLowerCopyToAffinePass());
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(sar::createFuseSiblingLoopsPass(!opts.axiInterface ||
                                                   opts.streamInterface));
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Fusion may fold a local producer and its final result copy together.
        // If that local value also feeds another result, later forwarding can
        // turn the public output into working storage. Re-establish one-write,
        // no-read result ports against the IR the HLS pipeline will schedule.
        pm.addPass(sar::createSARPrivatizeOutParams());
        pm.addPass(sar::createLowerCopyToAffinePass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Place dataflow buffers.
        pm.addPass(sar::createPlaceDataflowBufferPass(
            opts.externalBufferThreshold, opts.bramBytes, opts.uramBytes,
            opts.lutramBytes, opts.lutramMaxBytes,
            /*rebalanceOnly=*/false,
            /*allowDram=*/opts.axiInterface || opts.streamInterface));

        // Affine loop tiling.
        // Copy/placement transformations may expose new affine loops, but all
        // operation normalization was already completed above. Re-run only
        // the loop-parallel analysis instead of the full greedy rewrite set.
        pm.addPass(sar::createFuncPreprocessPass(opts.hlsTopFunc,
                                                 /*rewriteOps=*/false));
        pm.addPass(sar::createAffineLoopOrderOptPass());
        if (opts.loopTileSize != 1)
          pm.addPass(sar::createAffineLoopTilePass(opts.loopTileSize,
                                                   getStagingBudget(opts)));
        pm.addPass(affine::createSimplifyAffineStructuresPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // No generic tile staging here: a pass staging tiles of external
        // buffers at this point produces views that cross the dataflow
        // tasks formed below, which nothing downstream can express. The
        // staging that matters is decided at the SAR level instead
        // (sar-stage-transposes for corner turns, the banded gather for
        // interpolation), where the access pattern is known.

        // Form dataflow tasks from affine loops.
        pm.addPass(sar::createAffineStoreForwardPass());
        pm.addPass(sar::createCreateDataflowFromAffinePass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Lower and optimize dataflow.
        pm.addPass(sar::createLowerDataflowPass());
        pm.addPass(sar::createEliminateMultiProducerPass());
        pm.addPass(sar::createEliminateMultiConsumerPass());
        pm.addPass(sar::createScheduleDataflowNodePass());
        pm.addPass(sar::createBalanceDataflowNodePass());
        pm.addPass(sar::createLowerCopyToAffinePass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Mark schedules whose nodes are all scheduled as legal, which is
        // what makes the emitter attach `#pragma HLS dataflow`.
        pm.addPass(sar::createLegalizeDataflowPass());
        // Legalization may expose duplicate coordinate or phase computations;
        // remove them after the nodes have been merged.
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Forking and balancing create buffer copies after initial placement.
        // Recharge the tiers, including ping-pong storage, and demote URAM
        // overflow without changing the established DRAM interface.
        pm.addPass(sar::createPlaceDataflowBufferPass(
            opts.externalBufferThreshold, opts.bramBytes, opts.uramBytes,
            opts.lutramBytes, opts.lutramMaxBytes,
            /*rebalanceOnly=*/true,
            /*allowDram=*/opts.axiInterface || opts.streamInterface));

        // Memory optimization.
        pm.addPass(sar::createAffineStoreForwardPass());
        pm.addPass(sar::createReduceInitiationIntervalPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Outline dataflow nodes as functions.
        pm.addPass(sar::createCreateTokenStreamPass());
        pm.addPass(sar::createConvertDataflowToFuncPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Directive-level optimization.
        if (opts.axiInterface || opts.streamInterface)
          pm.addPass(sar::createCreateAxiInterfacePass(
              opts.hlsTopFunc, opts.streamInterface, opts.maxScratchArenas));
        pm.addPass(sar::createLoopPipeliningPass(opts.maxUnrolledOps,
                                                 opts.maxUnrollFactor));
        // Pack only memory-mapped interfaces here. AXI4-Stream needs an
        // explicit hls::stream<vector<...>> ABI rather than a vector array
        // carrying an axis pragma, and is handled separately.
        if (opts.axiInterface && !opts.streamInterface)
          pm.addPass(sar::createWidenExternalMemoryPass(
              opts.axiBusBits, opts.externalVectorMaxLanes,
              opts.externalVectorMinElements, opts.externalVectorPackOutputs,
              opts.externalVectorComputeLanes));
        // The banking tier threshold is one number expressed in two units:
        // the placement pass reasons in bytes, this one in bits, because a
        // bank's cost is its bit count. Converting here keeps the single
        // derived value (one bus beat) authoritative for both.
        pm.addPass(sar::createArrayPartitionPass(
            /*lutramMaxBits=*/opts.lutramMaxBytes * 8, opts.lutramBytes,
            opts.bramBytes, opts.uramBytes, opts.arrayPartitionMaxFactor));
        pm.addPass(sar::createShareEquivalentFunctionsPass());
        pm.addPass(mlir::createCanonicalizerPass());
      });
}

void sar::registerHLSPasses() {
  registerHLSPipeline();
  registerHLSTransformsPasses(); // TableGen'd per-pass registrations
}
