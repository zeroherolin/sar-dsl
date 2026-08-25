add_mlir_library(SARPipelines
  Pipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sar/Pipelines

  LINK_LIBS PUBLIC
  SARConversion
  SARTransforms
  HLSTransforms
  MLIRArithToLLVM
  MLIRBufferizationPipelines
  MLIRBufferizationTransforms
  MLIRComplexToLLVM
  MLIRComplexToStandard
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLLVMIRTransforms
  MLIRLinalgTransforms
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIROpenMPToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRSCFToOpenMP
  MLIRTransforms
)
