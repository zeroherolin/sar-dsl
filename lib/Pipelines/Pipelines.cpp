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
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Tensors -> buffers, destination-passing style at function boundaries.
  bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.bufferizeFunctionBoundaries = true;
  bufferizeOptions.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
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

  pm.addNestedPass<func::FuncOp>(
      sar::createSARReuseBuffers({options.reuseBufferMinElements}));

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
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertOpenMPToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void mlir::sar::buildSARToAffinePipeline(
    OpPassManager &pm, const SARBufferPipelineOptions &options) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(sar::createSARDecomplexify());
  pm.addPass(sar::createConvertSARFFTToAffine());
  pm.addPass(sar::createConvertSARInterpToAffine());
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
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));

  bufferization::BufferResultsToOutParamsPassOptions outParamsOptions;
  outParamsOptions.hoistStaticAllocs = true;
  outParamsOptions.modifyPublicFunctions = true;
  pm.addPass(
      bufferization::createBufferResultsToOutParamsPass(outParamsOptions));

  // Expand the copies bufferization leaves behind before anything else
  // reasons about the body: they are calls into a runtime the emitted
  // kernel does not link against.
  pm.addNestedPass<func::FuncOp>(sar::createSARLowerCopy());

  pm.addNestedPass<func::FuncOp>(
      sar::createSARReuseBuffers({options.reuseBufferMinElements}));

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
      "Lower SAR kernels to linalg-on-tensors (HLS hand-off level)",
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
