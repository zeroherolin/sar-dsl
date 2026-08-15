//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
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
struct HLSPipelineOptions
    : public PassPipelineOptions<HLSPipelineOptions> {
  Option<std::string> hlsTopFunc{
      *this, "top-func", llvm::cl::init("forward"),
      llvm::cl::desc("Specify the top function of the design")};

  // Option<unsigned> optLevel{
  //     *this, "opt-level", llvm::cl::init(1),
  //     llvm::cl::desc("Optimization level from 0 to 7 (default level is 1)")};

  // Option<unsigned> dataflowGranularity{
  //     *this, "dataflow-granularity",
  //     llvm::cl::desc("The granularity of dataflow (set 0 to disable)")};

  Option<double> fusionTolerance{
      *this, "fusion-tolerance", llvm::cl::init(100.0),
      llvm::cl::desc("Additional computation tolerated while loop fusing "
                     "(default is 100.0)")};

  Option<unsigned> loopTileSize{
      *this, "loop-tile-size", llvm::cl::init(2),
      llvm::cl::desc("The tile size of each loop (must larger equal to 1)")};

  Option<unsigned> onChipBytes{
      *this, "on-chip-bytes", llvm::cl::init(4 * 1024 * 1024),
      llvm::cl::desc("On-chip memory the design may occupy. Decisions that "
                     "spend it -- staging blocks for transposes, the bank "
                     "size below which distributed RAM is used -- follow "
                     "from this one number")};

  Option<bool> createLocalBuffer{
      *this, "create-local-buffer", llvm::cl::init(false),
      llvm::cl::desc("Stage tiles of external buffers in on-chip buffers. "
                     "Requires views to stay within one dataflow task, which "
                     "the affine flow does not currently guarantee")};

  Option<unsigned> loopUnrollFactor{
      *this, "loop-unroll-factor", llvm::cl::init(0),
      llvm::cl::desc("The overall loop unrolling factor (set 0 to disable)")};

  Option<bool> complexityAware{
      *this, "complexity-aware", llvm::cl::init(true),
      llvm::cl::desc("Whether to consider node complexity in the transform")};

  Option<bool> correlationAware{
      *this, "correlation-aware", llvm::cl::init(true),
      llvm::cl::desc("Whether to consider node correlation in the transform")};

  Option<unsigned> externalBufferThreshold{
      *this, "external-buffer-threshold", llvm::cl::init(1024),
      llvm::cl::desc("The threshold of placing external buffers")};

  Option<bool> placeExternalBuffer{
      *this, "place-external-buffer", llvm::cl::init(true),
      llvm::cl::desc("Place buffers in external memories")};

  Option<bool> balanceDataflow{
      *this, "balance-dataflow", llvm::cl::init(true),
      llvm::cl::desc("Whether to balance the dataflow")};

  Option<bool> axiInterface{*this, "axi-interface", llvm::cl::init(true),
                            llvm::cl::desc("Create AXI interface")};

  Option<bool> vectorize{*this, "vectorize", llvm::cl::init(false),
                         llvm::cl::desc("Vectorize with factor of 2")};

  Option<unsigned> debugPoint{
      *this, "debug-point", llvm::cl::init(0),
      llvm::cl::desc("Stop the pipeline at the given debug point")};
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


/// FIXME: Don't use! This is just for testing purpose.

void sar::registerHLSPipeline() {
  PassPipelineRegistration<HLSPipelineOptions>(
      "hls-pipeline", "Compile C++ to optimized C++",
      [](OpPassManager &pm, const HLSPipelineOptions &opts) {
        // // Affine loop dataflowing.
        // pm.addPass(sar::createCreateDataflowFromAffinePass());
        // pm.addPass(sar::createStreamDataflowTaskPass());
        // pm.addPass(mlir::createCanonicalizerPass());

        // if (opts.debugPoint == 4)
        //   return;

        // Affine loop fusion.
        pm.addPass(sar::createFuncPreprocessPass(opts.hlsTopFunc));
        pm.addPass(sar::createAffineLoopFusionPass(opts.fusionTolerance));
        sar::addSimplifyAffineLoopPasses(pm);
        sar::addCreateSubviewPasses(pm);
        pm.addPass(sar::createRaiseAffineToCopyPass());
        pm.addPass(sar::createSimplifyCopyPass());
        pm.addPass(sar::createLowerCopyToAffinePass());
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 5)
          return;

        // Place dataflow buffers.
        pm.addPass(sar::createPlaceDataflowBufferPass(
            opts.externalBufferThreshold, opts.placeExternalBuffer));

        // if (opts.vectorize) {
        //   pm.addPass(mlir::createSuperVectorizePass({2}));
        //   pm.addPass(mlir::createCanonicalizerPass());
        // }

        if (opts.debugPoint == 6)
          return;

        // Affine loop tiling.
        pm.addPass(sar::createFuncPreprocessPass(opts.hlsTopFunc));
        // pm.addPass(bufferization::createBufferLoopHoistingPass());
        pm.addPass(sar::createAffineLoopPerfectionPass());
        pm.addPass(sar::createRemoveVariableBoundPass());
        pm.addPass(sar::createAffineLoopOrderOptPass());
        if (opts.loopTileSize != 1)
          pm.addPass(sar::createAffineLoopTilePass(
              opts.loopTileSize, getStagingBudget(opts.onChipBytes)));
        pm.addPass(affine::createSimplifyAffineStructuresPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 7)
          return;

        // Local buffer allocation. Tiling leaves the tile-index arithmetic
        // in affine.apply ops, which splits the loop band that
        // CreateMemrefSubview walks; folding them back into the access maps
        // first keeps the band contiguous.
        if (opts.loopTileSize != 1 && opts.createLocalBuffer) {
          pm.addPass(affine::createSimplifyAffineStructuresPass());
          pm.addPass(mlir::createCanonicalizerPass());
          sar::addCreateSubviewPasses(pm);
          pm.addPass(sar::createCreateLocalBufferPass());
          pm.addPass(sar::createLowerCopyToAffinePass());
          pm.addPass(memref::createFoldMemRefAliasOpsPass());
          pm.addPass(affine::createSimplifyAffineStructuresPass());
          pm.addPass(mlir::createCanonicalizerPass());
        }

        if (opts.debugPoint == 8)
          return;

        // Affine loop dataflowing.
        pm.addPass(sar::createCollapseMemrefUnitDimsPass());
        pm.addPass(sar::createAffineStoreForwardPass());
        pm.addPass(sar::createCreateDataflowFromAffinePass());
        pm.addPass(sar::createStreamDataflowTaskPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 9)
          return;

        // Lower and optimize dataflow.
        pm.addPass(sar::createLowerDataflowPass());
        pm.addPass(sar::createEliminateMultiProducerPass());
        pm.addPass(sar::createEliminateMultiConsumerPass());
        pm.addPass(sar::createScheduleDataflowNodePass());
        pm.addPass(sar::createBalanceDataflowNodePass());
        pm.addPass(sar::createLowerCopyToAffinePass());
        pm.addPass(sar::createAffineStoreForwardPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 10)
          return;

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

        if (opts.debugPoint == 11)
          return;

        // Memory optimization.
        pm.addPass(sar::createSimplifyAffineIfPass());
        pm.addPass(sar::createAffineStoreForwardPass());
        pm.addPass(sar::createReduceInitialIntervalPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 12)
          return;

        // Convert dataflow to func.
        pm.addPass(sar::createCreateTokenStreamPass());
        pm.addPass(sar::createConvertDataflowToFuncPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 13)
          return;

        // Directive-level optimization.
        if (opts.axiInterface)
          pm.addPass(sar::createCreateAxiInterfacePass(opts.hlsTopFunc));
        pm.addPass(sar::createLoopPipeliningPass());
        pm.addPass(sar::createArrayPartitionPass());
        pm.addPass(sar::createCreateHLSPrimitivePass());
        pm.addPass(mlir::createCanonicalizerPass());
      });
}

void sar::registerTransformsPasses() {
  registerHLSPipeline();
  registerHLSTransformsPasses();
}
