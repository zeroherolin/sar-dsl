//===- Pipelines.cpp - SAR compilation pipelines --------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "sar/Pipelines/Pipelines.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "sar/Conversion/Passes.h"
#include "sar/Dialect/SAR/Transforms/Passes.h"

using namespace mlir;

void mlir::sar::buildSARToLinalgPipeline(OpPassManager &pm) {
  pm.addPass(sar::createConvertSARToLinalg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void mlir::sar::buildSARToLLVMPipeline(
    OpPassManager &pm, const SARBufferPipelineOptions &options) {
  // Normalize first: folds and rewrites such as interp1d dim = 0 must run
  // before the signal ops are matched.
  pm.addPass(createCanonicalizerPass());

  // SAR dialect -> linalg + runtime calls (tensor level).
  pm.addPass(sar::createConvertSARSignalToRuntime());
  pm.addPass(sar::createConvertSARToLinalg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Generalize before fusing: fusion rewrites `linalg.generic`, so a named
  // op between two of them -- a broadcast of a frequency axis against the
  // raster -- would otherwise block the chain.
  pm.addPass(createLinalgGeneralizeNamedOpsPass());

  // Fuse element-wise chains: long phase-computation sequences collapse
  // into single generics, eliminating whole intermediate tensors (the
  // dominant memory-bandwidth cost for large scenes).
  //
  // The SAR pass rather than the upstream one: upstream only fuses a
  // producer with a single consumer, which leaves every value the phase
  // expression reuses -- the broadcast frequency axes and the terms built
  // from them -- standing as its own full-raster plane. Recomputing those
  // costs arithmetic the machine has to spare and saves the traffic it
  // does not.
  pm.addNestedPass<func::FuncOp>(
      sar::createSARFuseElementwise({options.recomputeMinElements}));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Tensors -> buffers, destination-passing style at function boundaries.
  bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.bufferizeFunctionBoundaries = true;
  bufferizeOptions.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  // A compiled loop's body yields freshly computed planes; without this
  // the strict yield-equivalence check rejects every tensor carry.
  bufferizeOptions.allowReturnAllocsFromLoops = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));

  bufferization::BufferResultsToOutParamsPassOptions outParamsOptions;
  outParamsOptions.hoistStaticAllocs = true;
  // Kernel entry points are public; their results must still become
  // out-arguments (the launcher ABI is destination-passing style).
  outParamsOptions.modifyPublicFunctions = true;
  pm.addPass(
      bufferization::createBufferResultsToOutParamsPass(outParamsOptions));

  // Share buffers whose lifetimes do not overlap, before deallocation turns
  // the allocations into a fixed set.
  // Expand the copies bufferization leaves behind before anything else
  // reasons about the body: they are calls into a runtime the emitted
  // kernel does not link against.
  pm.addNestedPass<func::FuncOp>(sar::createSARLowerCopy());

  // allow-retype is safe on the CPU path: memref.view lowers through
  // expand-strided-metadata and finalize-memref-to-llvm without issue.
  // It is left off on the HLS path below: the C++ emitter only understands
  // typed arrays and has no memref.view emission.
  pm.addNestedPass<func::FuncOp>(sar::createSARReuseBuffers(
      {options.reuseBufferMinElements, /*allowRetype=*/true}));

  bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);

  // linalg -> parallel loops -> OpenMP -> LLVM dialect. Parallel dimensions
  // become scf.parallel and are distributed over hardware threads through
  // the OpenMP dialect (linked against LLVM's libomp).
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createConvertLinalgToParallelLoopsPass());
  pm.addPass(createConvertComplexToStandardPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertSCFToOpenMPPass());
  // Inline the alloca-free memref.alloca_scope regions the OpenMP
  // conversion introduces: reduction bodies keep inner scf.for loops whose
  // control-flow lowering would otherwise split the single-block region.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSCFToControlFlowPass());
  // C wrappers for public entry points only; private runtime-call
  // declarations keep their plain symbols.
  pm.addPass(sar::createSAREmitCInterface());
  pm.addPass(createConvertComplexToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  // Allocate through the runtime's plane pool rather than libc. A kernel's
  // intermediate planes are gigabytes each and are freed at the end of the
  // call; left to libc they are unmapped, and the next call faults and
  // zeroes every page again before it can store anything. Pooling them
  // makes that first touch a once-per-process cost instead of a
  // once-per-call one.
  FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
  memrefOptions.useGenericFunctions = true;
  pm.addPass(createFinalizeMemRefToLLVMConversionPass(memrefOptions));
  pm.addPass(createConvertOpenMPToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void mlir::sar::buildSARToAffinePipeline(
    OpPassManager &pm, const SARBufferPipelineOptions &options) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(sar::createSARDecomplexify());
  pm.addPass(sar::createConvertSARFFTToAffine(
      {options.fftStageGroup, options.fftParallelRows, options.fftIoUnroll}));
  ConvertSARInterpToAffineOptions interpOptions;
  interpOptions.enableBandedGather = options.interpEnableBandedGather;
  pm.addPass(sar::createConvertSARInterpToAffine(interpOptions));
  pm.addPass(sar::createConvertSARToLinalg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Generalize before fusing: fusion rewrites `linalg.generic`, so a named
  // op between two of them -- a broadcast of a frequency axis against the
  // raster, which every one of these chains has -- would otherwise block
  // the chain and materialize a full plane holding one row of values.
  pm.addPass(createLinalgGeneralizeNamedOpsPass());

  // Fuse element-wise chains here, while they are still on tensors: after
  // bufferization the reused buffers alias, and the write-after-write edges
  // hide the producer-consumer pairs fusion looks for. Each chain fused is
  // one fewer full-raster intermediate, and on the HLS path one fewer
  // buffer needing an interface -- which is why a full-size producer is
  // fused however many consumers it has.
  pm.addNestedPass<func::FuncOp>(
      sar::createSARFuseElementwise({options.recomputeMinElements}));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.bufferizeFunctionBoundaries = true;
  bufferizeOptions.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  // A compiled loop's body yields freshly computed planes; without this
  // the strict yield-equivalence check rejects every tensor carry.
  bufferizeOptions.allowReturnAllocsFromLoops = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));

  bufferization::BufferResultsToOutParamsPassOptions outParamsOptions;
  outParamsOptions.hoistStaticAllocs = true;
  outParamsOptions.modifyPublicFunctions = true;
  pm.addPass(
      bufferization::createBufferResultsToOutParamsPass(outParamsOptions));

  // When the working precision equals the result precision, bufferization
  // computes through the out-parameter; a multi-writer result port would
  // forfeit top-level dataflow, so give the computation a local buffer and
  // write the port once.
  pm.addNestedPass<func::FuncOp>(sar::createSARPrivatizeOutParams());

  // The HLS dataflow model forbids tasks with results, so compiled loops
  // stop carrying values here: the body iterates in the init buffer and a
  // per-iteration copy replaces the yield.
  pm.addNestedPass<func::FuncOp>(sar::createSARDemoteLoopCarries());

  // Expand the copies bufferization leaves behind before anything else
  // reasons about the body: they are calls into a runtime the emitted
  // kernel does not link against.
  pm.addNestedPass<func::FuncOp>(sar::createSARLowerCopy());

  // Same-type sharing only. Retyping a buffer through memref.view needs a
  // target that addresses bytes; here a buffer becomes a typed on-chip
  // array whose element width drives banking and partitioning, and the
  // HLS C++ emitter rejects memref.view outright.
  pm.addNestedPass<func::FuncOp>(sar::createSARReuseBuffers(
      {options.reuseBufferMinElements, /*allowRetype=*/false}));

  pm.addPass(createConvertLinalgToAffineLoopsPass());

  // Fold views into the accesses that reach through them. A view that
  // survives is a value the dataflow backend cannot place in a task, and on
  // any path it is a layout later passes would have to carry along; the
  // expansion turns the ones folding alone cannot reach into plain index
  // arithmetic first.
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createCanonicalizerPass());

  // Route transposes through an on-chip block, once they are loop nests and
  // before anything tiles them. Both sides of a corner turn then stream
  // contiguously instead of one of them taking a memory transaction per
  // element.
  if (options.transposeBlockBytes)
    pm.addNestedPass<func::FuncOp>(
        sar::createSARStageTransposes({options.transposeBlockBytes}));

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void mlir::sar::buildSARAffineToLLVMPipeline(
    OpPassManager &pm, const SARBufferPipelineOptions &options) {
  buildSARToAffinePipeline(pm, options);

  bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);

  pm.addPass(createLowerAffinePass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(sar::createSAREmitCInterface());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void mlir::sar::registerSARPipelines() {
  PassPipelineRegistration<>(
      "sar-to-linalg-pipeline",
      "Lower SAR kernels to linalg-on-tensors (external backend hand-off)",
      [](OpPassManager &pm) { buildSARToLinalgPipeline(pm); });
  PassPipelineRegistration<SARBufferPipelineOptions>(
      "sar-to-llvm-pipeline",
      "Lower SAR kernels to the LLVM dialect for CPU execution",
      buildSARToLLVMPipeline);
  PassPipelineRegistration<SARBufferPipelineOptions>(
      "sar-to-affine-pipeline",
      "Lower SAR kernels to split-complex affine IR (HLS hand-off level)",
      buildSARToAffinePipeline);
  PassPipelineRegistration<SARBufferPipelineOptions>(
      "sar-affine-to-llvm-pipeline",
      "The affine path lowered to the LLVM dialect (validation)",
      buildSARAffineToLLVMPipeline);
}
