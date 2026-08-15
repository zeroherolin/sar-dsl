add_mlir_translation_library(HLSTarget
  EmitHLSCpp.cpp

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
