//===- Pipelines.h - SAR compilation pipelines -------------------*- C++-*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_PIPELINES_PIPELINES_H
#define SAR_PIPELINES_PIPELINES_H

namespace mlir {
class OpPassManager;

namespace sar {

/// Lowers SAR kernels to linalg-on-tensors. This is the hand-off level for
/// HLS-oriented backends (e.g. ScaleHLS-HIDA), which run their own
/// bufferization and loop transformations.
void buildSARToLinalgPipeline(OpPassManager &pm);

/// Lowers SAR kernels all the way to the LLVM dialect for CPU execution.
/// Kernel functions are converted to destination-passing style (results
/// become trailing out-parameters) and exported through the MLIR C
/// interface (`_mlir_ciface_<name>`).
void buildSARToLLVMPipeline(OpPassManager &pm);

/// Lowers SAR kernels to split-complex affine/memref form: decomplexify,
/// Stockham FFT loop nests, windowed-sinc interpolation loops, linalg ->
/// affine loops, destination-passing style. This is the hand-off level for
/// HLS flows consuming affine IR (ScaleHLS `hida-cpp-pipeline`).
void buildSARToAffinePipeline(OpPassManager &pm);

/// The affine path continued down to the LLVM dialect; used to validate
/// the Stockham FFT lowering numerically on the CPU.
void buildSARAffineToLLVMPipeline(OpPassManager &pm);

/// Registers the pipelines above with the global pass pipeline registry.
void registerSARPipelines();

} // namespace sar
} // namespace mlir

#endif // SAR_PIPELINES_PIPELINES_H
