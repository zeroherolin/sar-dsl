//===----------------------------------------------------------------------===//
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

using namespace mlir;
using namespace sar;

namespace {
#define GEN_PASS_REGISTRATION
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace

void sar::addCreateSubviewPasses(OpPassManager &pm, CreateSubviewMode mode) {
  pm.addPass(sar::createCreateMemrefSubviewPass(mode));
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

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
      llvm::cl::desc("The tile size of each loop (must larger equal to 1)")};

  Option<unsigned> onChipBytes{
      *this, "on-chip-bytes", llvm::cl::init(4 * 1024 * 1024),
      llvm::cl::desc("On-chip memory the design may occupy. Decisions that "
                     "spend it -- staging blocks for transposes, the bank "
                     "size below which distributed RAM is used -- follow "
                     "from this one number")};

  Option<unsigned> bramBytes{
      *this, "bram-bytes", llvm::cl::init(0),
      llvm::cl::desc("Block RAM the design may occupy, in bytes (0 is "
                     "unbounded, which keeps the aggregate on-chip rule)")};

  Option<unsigned> uramBytes{
      *this, "uram-bytes", llvm::cl::init(0),
      llvm::cl::desc("Ultra RAM the design may occupy, in bytes (0 is "
                     "unbounded)")};

  Option<unsigned> lutramBytes{
      *this, "lutram-bytes", llvm::cl::init(0),
      llvm::cl::desc("Distributed RAM the design may occupy, in bytes (0 is "
                     "unbounded)")};

  Option<unsigned> lutramMaxBytes{
      *this, "lutram-max-bytes", llvm::cl::init(64),
      llvm::cl::desc("Bank size at or below which distributed RAM is used, "
                     "in bytes (one bus beat at the default 512-bit bus)")};

  Option<unsigned> uramMinBytes{
      *this, "uram-min-bytes", llvm::cl::init(36864),
      llvm::cl::desc("Buffer size at or above which ultra RAM is preferred, "
                     "in bytes")};

  Option<unsigned> loopUnrollFactor{
      *this, "loop-unroll-factor", llvm::cl::init(0),
      llvm::cl::desc("The overall loop unrolling factor (set 0 to disable)")};

  Option<bool> complexityAware{
      *this, "complexity-aware", llvm::cl::init(true),
      llvm::cl::desc("Whether to consider node complexity when parallelizing "
                     "(only effective with loop-unroll-factor > 0)")};

  Option<bool> correlationAware{
      *this, "correlation-aware", llvm::cl::init(true),
      llvm::cl::desc("Whether to consider node correlation when parallelizing "
                     "(only effective with loop-unroll-factor > 0)")};

  Option<unsigned> externalBufferThreshold{
      *this, "external-buffer-threshold", llvm::cl::init(1024),
      llvm::cl::desc("Element count at or above which a buffer moves to "
                     "DRAM (2^32-1 keeps everything resident)")};

  Option<bool> balanceDataflow{
      *this, "balance-dataflow", llvm::cl::init(true),
      llvm::cl::desc("Whether to balance the dataflow")};

  Option<bool> axiInterface{*this, "axi-interface", llvm::cl::init(true),
                            llvm::cl::desc("Create AXI interface")};

  Option<bool> streamInterface{
      *this, "stream-interface", llvm::cl::init(false),
      llvm::cl::desc("Emit axis (AXI4-Stream) pragmas on top-level ports "
                     "instead of m_axi ones; implies axi-interface=true")};
};
} // namespace

/// On-chip bytes one staging block may occupy.
///
/// A band that stages a block also touches the buffers it streams, and the
/// backend may keep several bands in flight, so a block gets a bounded share
/// of the budget rather than all of it. The share is what decides a
/// transpose's tile edge, and through it how long each side's AXI burst
/// runs.
static unsigned getStagingBudget(unsigned onChipBytes) {
  return std::max(1u, onChipBytes / 8);
}

void sar::registerHLSPipeline() {
  PassPipelineRegistration<HLSPipelineOptions>(
      "hls-pipeline", "Compile SAR affine IR to scheduled HLS IR",
      [](OpPassManager &pm, const HLSPipelineOptions &opts) {
        // Loop preparation. Note there is no affine fusion here: the fusion
        // pass this pipeline inherited only worked inside dataflow tasks,
        // which do not exist until CreateDataflowFromAffine runs further
        // down, and has been removed as dead code. Introducing fusion means
        // running it inside the tasks -- a scheduling change that needs its
        // own validation.
        pm.addPass(sar::createFuncPreprocessPass(opts.hlsTopFunc));
        sar::addSimplifyAffineLoopPasses(pm);
        sar::addCreateSubviewPasses(pm);
        pm.addPass(sar::createRaiseAffineToCopyPass());
        pm.addPass(sar::createSimplifyCopyPass());
        pm.addPass(sar::createLowerCopyToAffinePass());
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Place dataflow buffers.
        pm.addPass(sar::createPlaceDataflowBufferPass(
            opts.externalBufferThreshold, opts.bramBytes, opts.uramBytes,
            opts.lutramBytes, opts.lutramMaxBytes, opts.uramMinBytes));

        // Affine loop tiling.
        pm.addPass(sar::createFuncPreprocessPass(opts.hlsTopFunc));
        pm.addPass(sar::createAffineLoopPerfectionPass());
        pm.addPass(sar::createRemoveVariableBoundPass());
        pm.addPass(sar::createAffineLoopOrderOptPass());
        if (opts.loopTileSize != 1)
          pm.addPass(sar::createAffineLoopTilePass(
              opts.loopTileSize, getStagingBudget(opts.onChipBytes)));
        pm.addPass(affine::createSimplifyAffineStructuresPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // No generic tile staging here: a pass staging tiles of external
        // buffers at this point produces views that cross the dataflow
        // tasks formed below, which nothing downstream can express. The
        // staging that matters is decided at the SAR level instead
        // (sar-stage-transposes for corner turns, the banded gather for
        // interpolation), where the access pattern is known.

        // Affine loop dataflowing.
        pm.addPass(sar::createCollapseMemrefUnitDimsPass());
        pm.addPass(sar::createAffineStoreForwardPass());
        pm.addPass(sar::createCreateDataflowFromAffinePass());
        pm.addPass(sar::createStreamDataflowTaskPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Lower and optimize dataflow.
        pm.addPass(sar::createLowerDataflowPass());
        pm.addPass(sar::createEliminateMultiProducerPass());
        pm.addPass(sar::createEliminateMultiConsumerPass());
        pm.addPass(sar::createScheduleDataflowNodePass());
        if (opts.balanceDataflow)
          pm.addPass(sar::createBalanceDataflowNodePass());
        pm.addPass(sar::createLowerCopyToAffinePass());
        pm.addPass(sar::createAffineStoreForwardPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Parallelize dataflow.
        if (opts.loopUnrollFactor) {
          pm.addPass(sar::createParallelizeDataflowNodePass(
              opts.loopUnrollFactor, /*unrollPointLoopOnly=*/true,
              opts.complexityAware, opts.correlationAware));
          pm.addPass(affine::createSimplifyAffineStructuresPass());
          pm.addPass(mlir::createCanonicalizerPass());
        }

        // Mark schedules whose nodes are all scheduled as legal, which is
        // what makes the emitter attach `#pragma HLS dataflow`. Unlike the
        // unrolling above this does not depend on a factor being set: a
        // schedule is legal or not regardless of how its loops are widened.
        pm.addPass(sar::createLegalizeDataflowPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Memory optimization.
        pm.addPass(sar::createSimplifyAffineIfPass());
        pm.addPass(sar::createAffineStoreForwardPass());
        pm.addPass(sar::createReduceInitialIntervalPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Convert dataflow to func.
        pm.addPass(sar::createCreateTokenStreamPass());
        pm.addPass(sar::createConvertDataflowToFuncPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Directive-level optimization.
        if (opts.axiInterface || opts.streamInterface)
          pm.addPass(sar::createCreateAxiInterfacePass(opts.hlsTopFunc,
                                                       opts.streamInterface));
        pm.addPass(sar::createLoopPipeliningPass());
        // The banking tier threshold is one number expressed in two units:
        // the placement pass reasons in bytes, this one in bits, because a
        // bank's cost is its bit count. Converting here keeps the single
        // derived value (one bus beat) authoritative for both.
        pm.addPass(sar::createArrayPartitionPass(
            /*lutramMaxBits=*/opts.lutramMaxBytes * 8, opts.lutramBytes));
        pm.addPass(mlir::createCanonicalizerPass());
      });
}

void sar::registerTransformsPasses() {
  registerHLSPipeline();
  registerHLSTransformsPasses();
}
