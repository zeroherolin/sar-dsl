add_mlir_library(SARAnalysis
  DisplacementRange.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sar/Analysis

  LINK_LIBS PUBLIC
  SARDialect
  MLIRIR
  MLIRSupport
)
