//===- Pipelines.h - SAR compilation pipelines -------------------*- C++-*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_PIPELINES_PIPELINES_H
#define SAR_PIPELINES_PIPELINES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir {
class OpPassManager;

namespace sar {

/// Options shared by the pipelines that bufferize a kernel.
struct SARBufferPipelineOptions
    : public PassPipelineOptions<SARBufferPipelineOptions> {
  /// Buffers holding at least this many elements are shared between
  /// non-overlapping lifetimes rather than allocated one per intermediate.
  /// Give it the threshold the backend uses to place buffers off chip: below
  /// it a buffer acts as a dataflow channel and distinct producers are what
  /// let the backend pipeline the stages, at and above it the buffer is
  /// memory and sharing is free. Zero shares every buffer it can.
  Option<uint64_t> reuseBufferMinElements{
      *this, "reuse-buffer-min-elements",
      llvm::cl::desc("Only share buffers holding at least this many elements"),
      llvm::cl::init(0)};

  /// Element count above which a producer is fused into every consumer
  /// rather than stored. Recomputing costs an arithmetic unit; storing
  /// costs a buffer and the DRAM traffic through it. Zero recomputes
  /// everything it can.
  Option<uint64_t> recomputeMinElements{
      *this, "recompute-min-elements",
      llvm::cl::desc("Recompute rather than store results of at least this "
                     "many elements"),
      llvm::cl::init(0)};

  /// On-chip bytes a transposing loop nest may stage per block. Zero leaves
  /// transposes as plain copies, where one side of every access strides by a
  /// whole row.
  Option<unsigned> transposeBlockBytes{
      *this, "transpose-block-bytes",
      llvm::cl::desc("On-chip bytes a staged transpose block may occupy"),
      llvm::cl::init(0)};
};

/// Lowers SAR kernels to linalg-on-tensors. This is the hand-off level for
/// HLS-oriented backends (e.g. ScaleHLS-HIDA), which run their own
/// bufferization and loop transformations.
void buildSARToLinalgPipeline(OpPassManager &pm);

/// Lowers SAR kernels all the way to the LLVM dialect for CPU execution.
/// Kernel functions are converted to destination-passing style (results
/// become trailing out-parameters) and exported through the MLIR C
/// interface (`_mlir_ciface_<name>`).
void buildSARToLLVMPipeline(OpPassManager &pm,
                            const SARBufferPipelineOptions &options = {});

/// Lowers SAR kernels to split-complex affine/memref form: decomplexify,
/// Stockham FFT loop nests, windowed-sinc interpolation loops, linalg ->
/// affine loops, destination-passing style. This is the hand-off level for
/// HLS flows consuming affine IR (ScaleHLS `hida-cpp-pipeline`).
void buildSARToAffinePipeline(OpPassManager &pm,
                              const SARBufferPipelineOptions &options = {});

/// The affine path continued down to the LLVM dialect; used to validate
/// the Stockham FFT lowering numerically on the CPU.
void buildSARAffineToLLVMPipeline(OpPassManager &pm,
                                  const SARBufferPipelineOptions &options = {});

/// Registers the pipelines above with the global pass pipeline registry.
void registerSARPipelines();

} // namespace sar
} // namespace mlir

#endif // SAR_PIPELINES_PIPELINES_H
