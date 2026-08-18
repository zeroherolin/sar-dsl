add_mlir_translation_library(SARKernelFactsTarget
  KernelFacts.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sar/Target

  LINK_LIBS PUBLIC
  SARDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRTranslateLib
)

add_subdirectory(HLS)
