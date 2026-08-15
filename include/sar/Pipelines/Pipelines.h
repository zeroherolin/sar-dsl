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
  /// Below this size a buffer acts as a dataflow channel, where distinct
  /// producers are what let a backend pipeline the stages; at and above it
  /// the buffer is memory, where sharing costs nothing. Zero shares
  /// everything it can.
  Option<uint64_t> reuseBufferMinElements{
      *this, "reuse-buffer-min-elements",
      llvm::cl::desc("Only share buffers holding at least this many elements"),
      llvm::cl::init(0)};

  /// Above this size a producer is fused into every consumer rather than
  /// stored: recomputing costs an arithmetic unit, storing costs a buffer
  /// and the traffic through it. Zero recomputes everything it can.
  Option<uint64_t> recomputeMinElements{
      *this, "recompute-min-elements",
      llvm::cl::desc("Recompute rather than store results of at least this "
                     "many elements"),
      llvm::cl::init(0)};

  /// On-chip bytes a transposing nest may stage per block. Zero leaves
  /// transposes as plain copies, where one side strides by a whole row.
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
