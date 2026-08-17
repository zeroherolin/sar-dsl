//===- sar-lsp-server.cpp - MLIR LSP server with the SAR dialect ----------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "sar/Dialect/HLS/IR/HLS.h"
#include "sar/Dialect/SAR/IR/SARDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  // Both in-tree dialects, so IR from any pipeline stage opens cleanly.
  registry.insert<mlir::sar::SARDialect, mlir::sar::hls::HLSDialect>();

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
