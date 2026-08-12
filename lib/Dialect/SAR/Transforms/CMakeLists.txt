add_mlir_library(SARTransforms
  Decomplexify.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sar/Dialect/SAR/Transforms

  DEPENDS
  SARTransformsIncGen

  LINK_LIBS PUBLIC
  SARDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
