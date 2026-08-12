# TableGen for the SAR dialect and passes.

set(LLVM_TARGET_DEFINITIONS Dialect/SAR/IR/SAROps.td)
mlir_tablegen(Dialect/SAR/IR/SAROps.h.inc -gen-op-decls)
mlir_tablegen(Dialect/SAR/IR/SAROps.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect/SAR/IR/SARDialect.h.inc -gen-dialect-decls -dialect=sar)
mlir_tablegen(Dialect/SAR/IR/SARDialect.cpp.inc -gen-dialect-defs -dialect=sar)
add_public_tablegen_target(SARDialectIncGen)

set(LLVM_TARGET_DEFINITIONS Conversion/Passes.td)
mlir_tablegen(Conversion/Passes.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(SARConversionIncGen)

set(LLVM_TARGET_DEFINITIONS Dialect/SAR/Transforms/Passes.td)
mlir_tablegen(Dialect/SAR/Transforms/Passes.h.inc -gen-pass-decls
              -name SARTransforms)
add_public_tablegen_target(SARTransformsIncGen)
