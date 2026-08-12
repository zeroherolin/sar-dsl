add_mlir_dialect_library(SARDialect
  SARDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sar/Dialect/SAR/IR

  DEPENDS
  SARDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
