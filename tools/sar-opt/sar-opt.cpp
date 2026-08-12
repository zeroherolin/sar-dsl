//===- sar-opt.cpp - SAR dialect optimizer driver -------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "sar/Conversion/Passes.h"
#include "sar/Dialect/SAR/IR/SARDialect.h"
#include "sar/Dialect/SAR/Transforms/Passes.h"
#include "sar/Pipelines/Pipelines.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::sar::registerConversionPasses();
  mlir::sar::registerSARTransformsPasses();
  mlir::sar::registerSARPipelines();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<mlir::sar::SARDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SAR-DSL optimizer driver\n", registry));
}
