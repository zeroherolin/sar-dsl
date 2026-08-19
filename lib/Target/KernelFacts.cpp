//===- KernelFacts.cpp - Emit structured HLS strategy inputs ------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "sar/Dialect/SAR/IR/SARDialect.h"
#include "sar/Dialect/SAR/IR/SAROps.h"
#include "sar/Target/KernelFacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::sar;

namespace {

static std::optional<unsigned> planeBytes(Type elementType) {
  if (auto floatType = dyn_cast<FloatType>(elementType))
    return floatType.getWidth() / 8;
  if (auto complexType = dyn_cast<ComplexType>(elementType))
    if (auto floatType = dyn_cast<FloatType>(complexType.getElementType()))
      return floatType.getWidth() / 8;
  return std::nullopt;
}

static void updateShapeFacts(Type type, uint64_t &largest,
                             unsigned &narrowest) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType || !shapedType.hasStaticShape())
    return;
  auto bytes = planeBytes(shapedType.getElementType());
  if (!bytes)
    return;
  largest = std::max<uint64_t>(largest, shapedType.getNumElements());
  narrowest = std::min(narrowest, *bytes);
}

static LogicalResult emitKernelFacts(Operation *root, llvm::raw_ostream &os) {
  auto module = dyn_cast<ModuleOp>(root);
  if (!module) {
    root->emitError("kernel facts translation expects a module");
    return failure();
  }

  uint64_t largest = 0;
  unsigned narrowest = std::numeric_limits<unsigned>::max();
  unsigned transposes = 0;
  SmallVector<std::pair<int64_t, unsigned>> transforms;
  SmallVector<std::pair<uint64_t, uint64_t>> buffers;

  module.walk([&](Operation *op) {
    for (Type type : op->getOperandTypes())
      updateShapeFacts(type, largest, narrowest);
    for (Type type : op->getResultTypes())
      updateShapeFacts(type, largest, narrowest);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          updateShapeFacts(argument.getType(), largest, narrowest);

    auto recordTransform = [&](ShapedType type, int64_t dim) {
      auto bytes = planeBytes(type.getElementType());
      if (bytes && dim >= 0 && dim < type.getRank())
        transforms.emplace_back(type.getDimSize(dim), *bytes);
    };
    if (auto fft = dyn_cast<FFTOp>(op))
      recordTransform(cast<ShapedType>(fft.getInput().getType()), fft.getDim());
    else if (auto fft = dyn_cast<IFFTOp>(op))
      recordTransform(cast<ShapedType>(fft.getInput().getType()), fft.getDim());
    else if (auto fft = dyn_cast<FFTSplitOp>(op))
      recordTransform(cast<ShapedType>(fft.getRe().getType()), fft.getDim());

    if (isa<TransposeOp>(op))
      ++transposes;
    if (auto interp = dyn_cast<Interp1DOp>(op); interp && interp.getDim() == 0)
      transposes += 3;

    if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
      MemRefType type = alloc.getType();
      auto bytes = planeBytes(type.getElementType());
      if (bytes && type.hasStaticShape()) {
        uint64_t elements = type.getNumElements();
        buffers.emplace_back(elements, elements * *bytes);
      }
    }
  });

  if (narrowest == std::numeric_limits<unsigned>::max())
    narrowest = 8;

  os << "{\"plane_elements\":" << largest << ",\"element_bytes\":" << narrowest
     << ",\"transposes\":" << transposes << ",\"transforms\":[";
  for (auto [index, transform] : llvm::enumerate(transforms)) {
    if (index)
      os << ',';
    os << '[' << transform.first << ',' << transform.second << ']';
  }
  os << "],\"buffers\":[";
  for (auto [index, buffer] : llvm::enumerate(buffers)) {
    if (index)
      os << ',';
    os << '[' << buffer.first << ',' << buffer.second << ']';
  }
  os << "]}\n";
  return success();
}

} // namespace

void sar::registerKernelFactsTranslation() {
  static TranslateFromMLIRRegistration registration(
      "sar-emit-kernel-facts",
      "Emit structured kernel and buffer facts as JSON", emitKernelFacts,
      [](DialectRegistry &registry) {
        registerAllDialects(registry);
        registry.insert<SARDialect>();
      });
}
