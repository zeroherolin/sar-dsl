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
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Linalg/Passes.h"
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

void mlir::sar::buildSARToLLVMPipeline(OpPassManager &pm) {
  // SAR dialect -> linalg + runtime calls (tensor level).
  pm.addPass(sar::createConvertSARSignalToRuntime());
  pm.addPass(sar::createConvertSARToLinalg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

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

  bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);

  // linalg -> parallel loops -> OpenMP -> LLVM dialect. Parallel dimensions
  // become scf.parallel and are distributed over hardware threads through
  // the OpenMP dialect (linked against LLVM's libomp).
  pm.addPass(createConvertLinalgToParallelLoopsPass());
  pm.addPass(createConvertComplexToStandardPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertSCFToOpenMPPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addNestedPass<func::FuncOp>(LLVM::createLLVMRequestCWrappersPass());
  pm.addPass(createConvertComplexToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertOpenMPToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void mlir::sar::buildSARToAffinePipeline(OpPassManager &pm) {
  pm.addPass(sar::createSARDecomplexify());
  pm.addPass(sar::createConvertSARFFTToAffine());
  pm.addPass(sar::createConvertSARInterpToAffine());
  pm.addPass(sar::createConvertSARToLinalg());
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

  pm.addPass(createConvertLinalgToAffineLoopsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void mlir::sar::buildSARAffineToLLVMPipeline(OpPassManager &pm) {
  buildSARToAffinePipeline(pm);

  bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);

  pm.addPass(createLowerAffinePass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addNestedPass<func::FuncOp>(LLVM::createLLVMRequestCWrappersPass());
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
  PassPipelineRegistration<>(
      "sar-to-llvm-pipeline",
      "Lower SAR kernels to the LLVM dialect for CPU execution",
      [](OpPassManager &pm) { buildSARToLLVMPipeline(pm); });
  PassPipelineRegistration<>(
      "sar-to-affine-pipeline",
      "Lower SAR kernels to split-complex affine IR (HLS hand-off level)",
      [](OpPassManager &pm) { buildSARToAffinePipeline(pm); });
  PassPipelineRegistration<>(
      "sar-affine-to-llvm-pipeline",
      "The affine path lowered to the LLVM dialect (validation)",
      [](OpPassManager &pm) { buildSARAffineToLLVMPipeline(pm); });
}
