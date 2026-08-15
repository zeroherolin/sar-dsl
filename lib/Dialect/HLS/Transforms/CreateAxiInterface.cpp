//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_CREATEAXIINTERFACE
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace sar;
using namespace hls;

namespace {
struct CreateAxiInterface
    : public sar::impl::CreateAxiInterfaceBase<CreateAxiInterface> {
  CreateAxiInterface() = default;
  CreateAxiInterface(std::string hlsTopFunc) { topFunc = hlsTopFunc; }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module);
    auto context = builder.getContext();
    auto loc = builder.getUnknownLoc();

    // Get the top function of the module.
    auto func = getTopFunc(module, topFunc);
    if (!func) {
      emitError(module.getLoc(), "fail to find the top function");
      return signalPassFailure();
    }
    setTopFuncAttr(func);

    // Create the main function of runtime.
    // FIXME: Make sure there's no function called "main" already.
    builder.setInsertionPointAfter(func);
    auto mainFunc =
        builder.create<func::FuncOp>(loc, "main", func.getFunctionType());
    setRuntimeAttr(mainFunc);
    auto mainBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToEnd(mainBlock);

    // Move all the arguments of the top function to the main function.
    for (auto [funcArg, mainArg] :
         llvm::zip(func.getArguments(), mainBlock->getArguments()))
      funcArg.replaceAllUsesWith(mainArg);
    func.front().eraseArguments([](BlockArgument arg) { return true; });

    // A helper to handle vectorized buffers.
    auto getSelfOrVectorizedBuffer = [&](Value buffer) {
      if (llvm::any_of(buffer.getUses(), [](OpOperand &use) {
            return isa<BufferVectorizeOp>(use.getOwner());
          })) {
        if (!buffer.hasOneUse()) {
          emitError(buffer.getLoc(), "buffer can only be vectorized once");
          return signalPassFailure(), Value();
        }
        auto vectorize = cast<BufferVectorizeOp>(*buffer.user_begin());
        vectorize->remove();
        builder.insert(vectorize);
        return Value(vectorize.getResult());
      }
      return Value(buffer);
    };

    // Move buffer arguments of the top function to the main function. Collect
    // all buffers to be converted to AXI interfaces into "buffers". At the same
    // time, we also directly collect all scalar arguments into "funcPorts".
    SmallVector<Value, 32> buffers;
    SmallVector<Value, 32> funcPorts;
    for (auto arg : mainBlock->getArguments())
      if (isa<MemRefType, StreamType>(arg.getType())) {
        buffers.push_back(getSelfOrVectorizedBuffer(arg));
      } else if (isa<ShapedType>(arg.getType())) {
        emitError(arg.getLoc(), "unsupported argument type");
        return signalPassFailure();
      } else {
        funcPorts.push_back(arg);
        arg.replaceAllUsesWith(
            func.front().addArgument(arg.getType(), arg.getLoc()));
      }

    // Move buffers allocated in the top function to the main function. Collect
    // all buffers to be converted to AXI interfaces into "buffers".
    for (auto buffer :
         llvm::make_early_inc_range(func.getOps<hls::BufferLikeInterface>())) {
      if (!isExtBuffer(buffer.getMemref()))
        continue;
      buffer->remove();
      builder.insert(buffer);
      buffers.push_back(getSelfOrVectorizedBuffer(buffer.getMemref()));
    }

    // A helper to get AXI bundle type from a buffer.
    auto getBundleType = [&](Value buffer) {
      if (auto memrefType = dyn_cast<MemRefType>(buffer.getType()))
        return BundleType::get(context, memrefType.getElementType(),
                               AxiKind::MM);
      if (auto streamType = dyn_cast<StreamType>(buffer.getType()))
        return BundleType::get(context, streamType.getElementType(),
                               AxiKind::STREAM);
      llvm_unreachable("invalid buffer type");
    };

    // Convert collected buffers to AXI ports and collect them in "funcPorts".
    // One port per buffer, and ports of the same element type share one AXI
    // master: a design then presents a handful of interfaces and one
    // argument per buffer, which is what a device can wire up and a host can
    // bind. Accesses through a shared bundle are serialized by the interface
    // whether or not they have separate ports, so splitting per use would
    // only add arguments.
    unsigned bundleIndex = 0;
    llvm::DenseMap<BundleType, AxiBundleOp> bundles;
    for (auto buffer : buffers) {
      auto bundleType = getBundleType(buffer);
      auto &bundle = bundles[bundleType];
      if (!bundle) {
        builder.setInsertionPointToStart(&func.front());
        bundle = builder.create<AxiBundleOp>(
            loc, bundleType, "axi_" + std::to_string(bundleIndex++));
      }
      // Ports go after every bundle, so a reused bundle still dominates the
      // port that refers to it.
      builder.setInsertionPointAfter(bundle);

      auto axiType = AxiType::get(context, buffer.getType());
      auto axiPort = builder.create<AxiPortOp>(
          loc, buffer.getType(), bundle,
          func.front().addArgument(axiType, buffer.getLoc()));
      buffer.replaceUsesWithIf(axiPort, [&](OpOperand &use) {
        return use.getOwner() != axiPort;
      });

      builder.setInsertionPointToEnd(mainBlock);
      funcPorts.push_back(builder.create<AxiPackOp>(loc, axiType, buffer));
    }

    // Update the top function and call.
    builder.setInsertionPointToEnd(mainBlock);
    auto call = builder.create<func::CallOp>(func.getLoc(), func.getName(),
                                             func.getResultTypes(), funcPorts);
    func.setType(call.getCalleeType());
    builder.create<func::ReturnOp>(loc, call.getResults());
  }
};
} // namespace

std::unique_ptr<Pass>
sar::createCreateAxiInterfacePass(std::string hlsTopFunc) {
  return std::make_unique<CreateAxiInterface>(hlsTopFunc);
}
