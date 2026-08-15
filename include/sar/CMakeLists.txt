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

# TableGen for the HLS dialect and passes. The dialect models what a
# high-level-synthesis backend needs to say about a design -- dataflow
# structure, memory kinds, interfaces -- and is lowered to by the SAR
# pipelines rather than written by hand.

set(LLVM_TARGET_DEFINITIONS Dialect/HLS/IR/HLSOps.td)
mlir_tablegen(Dialect/HLS/IR/HLSOps.h.inc -gen-op-decls)
mlir_tablegen(Dialect/HLS/IR/HLSOps.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect/HLS/IR/HLSOpsDialect.h.inc -gen-dialect-decls -dialect=hls)
mlir_tablegen(Dialect/HLS/IR/HLSOpsDialect.cpp.inc -gen-dialect-defs -dialect=hls)
mlir_tablegen(Dialect/HLS/IR/HLSOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=hls)
mlir_tablegen(Dialect/HLS/IR/HLSOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=hls)
add_public_tablegen_target(HLSDialectIncGen)

set(LLVM_TARGET_DEFINITIONS Dialect/HLS/IR/HLSAttributes.td)
mlir_tablegen(Dialect/HLS/IR/HLSOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(Dialect/HLS/IR/HLSOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(Dialect/HLS/IR/HLSOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(Dialect/HLS/IR/HLSOpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(HLSAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS Dialect/HLS/IR/HLSInterfaces.td)
mlir_tablegen(Dialect/HLS/IR/HLSOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(Dialect/HLS/IR/HLSOpsInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(HLSInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS Dialect/HLS/Transforms/Passes.td)
mlir_tablegen(Dialect/HLS/Transforms/Passes.h.inc -gen-pass-decls
              -name HLSTransforms)
add_public_tablegen_target(HLSTransformsIncGen)
