//===- SARSignalToRuntime.cpp - Lower signal ops to runtime calls ---------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
// Lowers sar.fft / sar.ifft / sar.interp1d into calls to libsar_runtime.
// The rewrite happens at tensor level, before one-shot bufferization, using
// the sanctioned bufferization escape hatch:
//
//   %in  = bufferization.to_buffer %tensor
//   %out = memref.alloc()
//   call @sar_rt_xxx(%in, %out, ...)
//   %res = bufferization.to_tensor %out restrict writable
//
// Runtime functions are declared with `llvm.emit_c_interface` so that calls
// lower to `_mlir_ciface_*` symbols taking memref descriptors by pointer.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringSwitch.h"

#include "sar/Conversion/Passes.h"
#include "sar/Dialect/SAR/IR/SARDialect.h"
#include "sar/Dialect/SAR/IR/SAROps.h"
#include "sar/Runtime/RuntimeEnums.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_CONVERTSARSIGNALTORUNTIME
#include "sar/Conversion/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace mlir::sar;

namespace {

/// Returns the runtime symbol suffix for a complex element type: "c64" for
/// complex<f32>, "c128" for complex<f64>, "f64" for f64.
static StringRef getElementSuffix(Type elementType) {
  if (auto complexTy = dyn_cast<ComplexType>(elementType))
    return complexTy.getElementType().isF32() ? "c64" : "c128";
  assert(elementType.isF64() && "unexpected runtime element type");
  return "f64";
}

/// Returns a memref type with the same shape/element type as `tensorType`
/// and an identity (static) layout.
static MemRefType getStaticMemRefType(RankedTensorType tensorType) {
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
}

/// Returns the fully-dynamic strided memref type used in runtime function
/// signatures, so that one declaration serves any layout.
static MemRefType getDynamicMemRefType(RankedTensorType tensorType) {
  auto layout = StridedLayoutAttr::get(
      tensorType.getContext(), ShapedType::kDynamic,
      SmallVector<int64_t>(tensorType.getRank(), ShapedType::kDynamic));
  return MemRefType::get(
      SmallVector<int64_t>(tensorType.getRank(), ShapedType::kDynamic),
      tensorType.getElementType(), layout);
}

/// Ensures a private declaration of the runtime function exists and returns
/// it. The declaration carries `llvm.emit_c_interface`.
static func::FuncOp ensureRuntimeDecl(PatternRewriter &rewriter,
                                      ModuleOp module, StringRef name,
                                      TypeRange argTypes) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto funcType = rewriter.getFunctionType(argTypes, {});
  auto decl = func::FuncOp::create(rewriter, module.getLoc(), name, funcType);
  decl.setPrivate();
  decl->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
  return decl;
}

/// Materializes a tensor value as a dynamic-strided memref suitable for a
/// runtime call argument.
static Value tensorToRuntimeArg(PatternRewriter &rewriter, Location loc,
                                Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  Value buffer = bufferization::ToBufferOp::create(
      rewriter, loc, getStaticMemRefType(tensorType), tensor);
  return memref::CastOp::create(rewriter, loc, getDynamicMemRefType(tensorType),
                                buffer);
}

/// Allocates the result buffer, returning both the alloc (static type) and
/// the casted runtime argument.
static std::pair<Value, Value> allocResultBuffer(PatternRewriter &rewriter,
                                                 Location loc,
                                                 RankedTensorType tensorType) {
  Value alloc =
      memref::AllocOp::create(rewriter, loc, getStaticMemRefType(tensorType));
  Value casted = memref::CastOp::create(
      rewriter, loc, getDynamicMemRefType(tensorType), alloc);
  return {alloc, casted};
}

/// Wraps the filled result buffer back into tensor land.
static Value bufferToResultTensor(PatternRewriter &rewriter, Location loc,
                                  RankedTensorType tensorType, Value alloc) {
  return bufferization::ToTensorOp::create(
      rewriter, loc, tensorType, alloc, /*restrict=*/true, /*writable=*/true);
}

template <typename FFTOpTy, bool inverse>
struct FFTLikeLowering : OpRewritePattern<FFTOpTy> {
  using OpRewritePattern<FFTOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(FFTOpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->template getParentOfType<ModuleOp>();
    auto tensorType = cast<RankedTensorType>(op.getType());
    int64_t rank = tensorType.getRank();

    std::string symbol = ("sar_rt_fft_" + Twine(rank) + "d_" +
                          getElementSuffix(tensorType.getElementType()))
                             .str();

    Value input = tensorToRuntimeArg(rewriter, loc, op.getInput());
    auto [alloc, output] = allocResultBuffer(rewriter, loc, tensorType);
    Value dim = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(op.getDim()));
    Value inv =
        arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(inverse));

    auto decl = ensureRuntimeDecl(
        rewriter, module, symbol,
        {input.getType(), output.getType(), dim.getType(), inv.getType()});
    func::CallOp::create(rewriter, loc, decl,
                         ValueRange{input, output, dim, inv});

    rewriter.replaceOp(op,
                       bufferToResultTensor(rewriter, loc, tensorType, alloc));
    return success();
  }
};

struct Interp1DLowering : OpRewritePattern<Interp1DOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Interp1DOp op,
                                PatternRewriter &rewriter) const override {
    // dim = 0 is normalized into transposes by canonicalization, which
    // runs before this pass in every pipeline.
    if (op.getDim() != 1)
      return op.emitOpError(
          "interpolation with dim = 0 must be canonicalized to dim = 1 with "
          "transposes; run --canonicalize before this pass");
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto dataType = cast<RankedTensorType>(op.getData().getType());

    std::string symbol = ("sar_rt_interp1d_2d_" +
                          Twine(getElementSuffix(dataType.getElementType())))
                             .str();

    Value data = tensorToRuntimeArg(rewriter, loc, op.getData());
    Value positions = tensorToRuntimeArg(rewriter, loc, op.getPositions());
    auto [alloc, output] = allocResultBuffer(rewriter, loc, dataType);

    // Kernel/window encodings are shared with the runtime through
    // sar/Runtime/RuntimeEnums.h.
    int64_t kernelId = llvm::StringSwitch<int64_t>(op.getKernel())
                           .Case("nearest", sar_rt::kNearest)
                           .Case("linear", sar_rt::kLinear)
                           .Case("cubic", sar_rt::kCubic)
                           .Default(sar_rt::kSinc);
    int64_t windowId = llvm::StringSwitch<int64_t>(op.getWindow())
                           .Case("rect", sar_rt::kRect)
                           .Case("hann", sar_rt::kHann)
                           .Case("hamming", sar_rt::kHamming)
                           .Default(sar_rt::kKaiser);
    Value kernel = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(kernelId));
    Value taps = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(op.getTaps()));
    Value window = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(windowId));
    Value beta = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getF64FloatAttr(op.getBeta().convertToDouble()));

    auto decl = ensureRuntimeDecl(
        rewriter, module, symbol,
        {data.getType(), positions.getType(), output.getType(),
         kernel.getType(), taps.getType(), window.getType(), beta.getType()});
    func::CallOp::create(
        rewriter, loc, decl,
        ValueRange{data, positions, output, kernel, taps, window, beta});

    rewriter.replaceOp(op,
                       bufferToResultTensor(rewriter, loc, dataType, alloc));
    return success();
  }
};

struct ConvertSARSignalToRuntimePass
    : sar::impl::ConvertSARSignalToRuntimeBase<ConvertSARSignalToRuntimePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addIllegalOp<FFTOp, IFFTOp, Interp1DOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    patterns.add<FFTLikeLowering<FFTOp, /*inverse=*/false>,
                 FFTLikeLowering<IFFTOp, /*inverse=*/true>, Interp1DLowering>(
        context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
