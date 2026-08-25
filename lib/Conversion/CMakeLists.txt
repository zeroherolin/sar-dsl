add_mlir_library(SARConversion
  SARFFTToAffine/SARFFTToAffine.cpp
  SARInterpToAffine/SARInterpToAffine.cpp
  SARSignalToRuntime/SARSignalToRuntime.cpp
  SARToLinalg/SARToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sar/Conversion

  DEPENDS
  SARConversionIncGen

  LINK_LIBS PUBLIC
  SARAnalysis
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
