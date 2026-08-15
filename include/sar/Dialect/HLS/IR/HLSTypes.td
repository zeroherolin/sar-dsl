//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_IR_HLSTYPES_TD
#define SAR_DIALECT_HLS_IR_HLSTYPES_TD

include "sar/Dialect/HLS/IR/HLSAttributes.td"

class HLSType<string name, list<Trait> traits = []> :
    TypeDef<HLSDialect, name, traits>;

def StreamType : HLSType<"Stream"> {
  let summary = "An HLS stream type";
  let description = [{
    Represents a stream of any type that can be transfered between HLS modules.
    This type is equal to the hls::stream<> type in Xilinx Vivado HLS.
  }];
  let mnemonic = "stream";

  let parameters = (ins "mlir::Type":$elementType, "unsigned":$depth);
  let assemblyFormat = "`<` qualified($elementType) `,` $depth `>`";
}

def AxiType : HLSType<"Axi"> {
  let summary = "An HLS AXI interface type";
  let mnemonic = "axi";

  let parameters = (ins "mlir::Type":$elementType);
  let assemblyFormat = "`<` qualified($elementType) `>`";

  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    Type getDataType();
  }];
}

def BundleType : HLSType<"Bundle"> {
  let summary = "An HLS AXI bundle type";
  let mnemonic = "bundle";

  let parameters = (ins "mlir::Type":$dataType, AxiKindParam:$kind);
  let assemblyFormat = "`<` qualified($dataType) `,` $kind `>`";
}

#endif // SAR_DIALECT_HLS_IR_HLSTYPES_TD
