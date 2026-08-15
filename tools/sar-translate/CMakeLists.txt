get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(sar-translate
  sar-translate.cpp

  DEPENDS
  HLSDialectIncGen
)
target_link_libraries(sar-translate PRIVATE
  HLSTarget
  HLSDialect
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTranslateLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
)
llvm_update_compile_flags(sar-translate)
mlir_check_all_link_libraries(sar-translate)
