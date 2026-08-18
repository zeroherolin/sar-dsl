add_mlir_library(SARTransforms
  Decomplexify.cpp
  DemoteLoopCarries.cpp
  EmitCInterface.cpp
  FuseElementwise.cpp
  LowerCopy.cpp
  PrivatizeOutParams.cpp
  ReuseBuffers.cpp
  StageTransposes.cpp
  VerifyPrecision.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sar/Dialect/SAR/Transforms

  DEPENDS
  SARTransformsIncGen

  LINK_LIBS PUBLIC
  SARDialect
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRAffineDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRIR
  MLIRPass
)
