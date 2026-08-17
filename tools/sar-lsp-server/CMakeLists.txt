get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(sar-lsp-server
  sar-lsp-server.cpp

  DEPENDS
  SARDialectIncGen
  HLSDialectIncGen
)
target_link_libraries(sar-lsp-server PRIVATE
  SARDialect
  HLSDialect
  ${dialect_libs}
  ${extension_libs}
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRLspServerLib
)
llvm_update_compile_flags(sar-lsp-server)
mlir_check_all_link_libraries(sar-lsp-server)
