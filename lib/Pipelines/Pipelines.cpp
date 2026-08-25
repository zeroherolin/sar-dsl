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
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/SAR/Transforms/Passes.h"

using namespace mlir;

void mlir::sar::buildSARToLinalgPipeline(OpPassManager &pm) {
  // Normalize first: folds and rewrites such as interp1d dim = 0 must run
  // before the signal ops are matched.
  pm.addPass(createCanonicalizerPass());
  // Transforms and interpolation have no linalg form -- they are not
  // structured loop nests over a fixed index space -- so they become calls
  // against the runtime ABI, exactly as on the CPU path. A backend consuming
  // this level implements those calls or lowers them itself; everything else
  // arrives as linalg on tensors.
  pm.addPass(sar::createConvertSARSignalToRuntime());
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

  // Expose named operations to element-wise fusion.
  pm.addPass(createLinalgGeneralizeNamedOpsPass());

  // Fuse multi-consumer element-wise chains by recomputing cheap producers.
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

  // CSE can make two result positions return the same allocation. The
  // upstream out-param conversion erases a hoisted allocation at its first
  // result and otherwise dereferences the duplicate. Split those results
  // before converting the ABI.
  pm.addNestedPass<func::FuncOp>(sar::createSARDistinctReturnBuffers());

  bufferization::BufferResultsToOutParamsPassOptions outParamsOptions;
  outParamsOptions.hoistStaticAllocs = true;
  // Kernel entry points are public; their results must still become
  // out-arguments (the launcher ABI is destination-passing style).
  outParamsOptions.modifyPublicFunctions = true;
  pm.addPass(
      bufferization::createBufferResultsToOutParamsPass(outParamsOptions));

  // Expand copies before lifetime-based buffer sharing.
  pm.addNestedPass<func::FuncOp>(sar::createSARLowerCopy());

  // The CPU path can lower byte-backed retyped buffers through memref.view.
  pm.addNestedPass<func::FuncOp>(sar::createSARReuseBuffers(
      {options.reuseBufferMinElements, /*allowRetype=*/true}));

  bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);

  // linalg -> parallel loops -> OpenMP -> LLVM dialect.
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createConvertLinalgToParallelLoopsPass());
  pm.addPass(createConvertComplexToStandardPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertSCFToOpenMPPass());
  // Canonicalize alloca_scope regions before control-flow lowering.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSCFToControlFlowPass());
  // C wrappers for public entry points only; private runtime-call
  // declarations keep their plain symbols.
  pm.addPass(sar::createSAREmitCInterface());
  pm.addPass(createConvertComplexToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  // Route allocations through the runtime plane pool.
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
  interpOptions.fullRowMaxBytes = options.interpFullRowMaxBytes;
  interpOptions.cacheCopies = options.interpCacheCopies;
  interpOptions.completeBankMaxElements = options.interpCompleteBankMaxElements;
  pm.addPass(sar::createConvertSARInterpToAffine(interpOptions));
  pm.addPass(sar::createConvertSARToLinalg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Expose named operations to element-wise fusion.
  pm.addPass(createLinalgGeneralizeNamedOpsPass());

  // Fuse before bufferization introduces aliasing edges.
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

  pm.addNestedPass<func::FuncOp>(sar::createSARDistinctReturnBuffers());

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

  // HLS packages cannot call the memref copy runtime.
  pm.addNestedPass<func::FuncOp>(sar::createSARLowerCopy());

  pm.addPass(createConvertLinalgToAffineLoopsPass());

  // Fold views into access indices before whole-sweep fusion. Buffer reuse is
  // deliberately later: aliasing a later output onto an earlier input makes
  // independent real/imaginary sweeps appear dependent and prevents the
  // compiler from fusing their phase arithmetic or corner turns.
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(sar::createAffineStoreForwardPass());
  pm.addPass(createCanonicalizerPass());
  if (options.fuseSiblingSweeps) {
    pm.addNestedPass<func::FuncOp>(sar::createFuseSiblingLoopsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // Stage corner turns through an on-chip block before tiling.
  if (options.transposeBlockBytes)
    pm.addNestedPass<func::FuncOp>(
        sar::createSARStageTransposes({options.transposeBlockBytes}));

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // HLS sharing preserves element types because storage width drives banking.
  // It runs after fusion and transpose staging so lifetime reuse cannot hide
  // legal fusion opportunities, and can also share non-overlapping staging
  // blocks introduced above.
  pm.addNestedPass<func::FuncOp>(sar::createSARReuseBuffers(
      {options.reuseBufferMinElements, /*allowRetype=*/false}));
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
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
