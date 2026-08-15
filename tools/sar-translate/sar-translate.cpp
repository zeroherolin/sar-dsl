//===- sar-translate.cpp - SAR translation driver -------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Drives translations out of MLIR into a target's own text. Each target
// registers itself under `sar/Target/<name>`, so adding one is a matter of
// registering it here rather than of adding a tool.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "sar/Target/InitAllTranslations.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  mlir::sar::registerAllTranslations();

  return failed(mlir::mlirTranslateMain(argc, argv, "SAR translation driver"));
}
