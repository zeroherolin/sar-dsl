add_mlir_translation_library(HLSTarget
  EmitHLSCpp.cpp
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sar/Target/HLS

  LINK_LIBS PUBLIC
  HLSDialect
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTranslateLib
)

add_mlir_translation_library(SARKernelFactsTarget
  KernelFacts.cpp
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sar/Target/HLS

  LINK_LIBS PUBLIC
  SARDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRTranslateLib
)
