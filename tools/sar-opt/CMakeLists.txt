get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(sar-opt
  sar-opt.cpp

  DEPENDS
  SARDialectIncGen
  SARConversionIncGen
)
target_link_libraries(sar-opt PRIVATE
  SARDialect
  SARTransforms
  SARConversion
  SARPipelines
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
)
llvm_update_compile_flags(sar-opt)
mlir_check_all_link_libraries(sar-opt)
