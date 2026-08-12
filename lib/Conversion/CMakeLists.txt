add_mlir_library(SARConversion
  SARToLinalg/SARToLinalg.cpp
  SARSignalToRuntime/SARSignalToRuntime.cpp
  SARFFTToAffine/SARFFTToAffine.cpp
  SARInterpToAffine/SARInterpToAffine.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sar/Conversion

  DEPENDS
  SARConversionIncGen

  LINK_LIBS PUBLIC
  SARDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
)
