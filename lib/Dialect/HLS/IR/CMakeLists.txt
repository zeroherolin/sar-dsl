add_mlir_dialect_library(HLSDialect
  HLS.cpp
  Utils.cpp
  Analysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sar/Dialect/HLS/IR

  DEPENDS
  HLSDialectIncGen
  HLSAttributesIncGen
  HLSInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
)
