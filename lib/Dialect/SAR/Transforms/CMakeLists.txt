add_mlir_library(SARTransforms
  Decomplexify.cpp
  FuseElementwise.cpp
  LowerCopy.cpp
  EmitCInterface.cpp
  ReuseBuffers.cpp
  StageTransposes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sar/Dialect/SAR/Transforms

  DEPENDS
  SARTransformsIncGen

  LINK_LIBS PUBLIC
  SARDialect
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRAffineUtils
  MLIRMemRefDialect
  MLIRIR
  MLIRPass
)
