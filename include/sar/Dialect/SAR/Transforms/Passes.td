//===- Passes.td - SAR dialect transformation passes -------*- tablegen -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_SAR_TRANSFORMS_PASSES_TD
#define SAR_DIALECT_SAR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SARDecomplexify : Pass<"sar-decomplexify", "ModuleOp"> {
  let summary = "Split complex tensors into real/imaginary float planes";
  let description = [{
    Rewrites kernels so that no complex element types remain: every complex
    tensor value becomes a pair of float tensors and complex arithmetic is
    expanded into real arithmetic. Function signatures change accordingly
    (each complex argument/result is replaced by adjacent `re, im` float
    tensors, in order).

    This enables targets without complex-number support -- notably HLS
    C++ emission through ScaleHLS-HIDA. Signal and interpolation ops map
    to their split-complex forms: `sar.fft`/`sar.ifft` become
    `sar.fft_split` and `sar.interp1d` becomes `sar.interp1d_split`.
  }];
  let dependentDialects = ["::mlir::func::FuncDialect"];
}

def SARReuseBuffers : Pass<"sar-reuse-buffers", "::mlir::func::FuncOp"> {
  let summary = "Share buffers whose lifetimes do not overlap";
  let description = [{
    Bufferization gives every intermediate tensor its own allocation, so an
    imaging chain ends up holding dozens of full rasters even though it is a
    sequence of whole-raster passes with only a handful live at a time. This
    pass runs a linear scan over the function body and lets a buffer whose
    last use has passed carry a later one of the same type.

    `min-elements` selects which buffers take part. Below it a buffer acts as
    a dataflow channel, where distinct producers are what let a backend
    pipeline the stages; at and above it the buffer is memory, where sharing
    is free. Give it the same threshold the backend uses to place buffers off
    chip and the two decisions stay consistent.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect", "::mlir::memref::MemRefDialect"
  ];
  let options = [
    Option<"minElements", "min-elements", "uint64_t", /*default=*/"0",
           "Only share buffers holding at least this many elements">
  ];
}

def SARLowerCopy : Pass<"sar-lower-copy", "::mlir::func::FuncOp"> {
  let summary = "Expand memref.copy into an affine sweep";
  let description = [{
    `memref.copy` lowers to a call into the MLIR runtime support library,
    which neither the standalone kernel nor an HLS target links against.
    Bufferization introduces it wherever a value has to change layout, so
    expanding it into a loop nest is what keeps those cases working -- and
    on the HLS path leaves a sweep the backend can pipeline rather than an
    opaque call.
  }];
  let dependentDialects = [
    "::mlir::affine::AffineDialect", "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def SARFuseElementwise : Pass<"sar-fuse-elementwise", "::mlir::func::FuncOp"> {
  let summary = "Fuse element-wise chains, recomputing rather than storing";
  let description = [{
    Upstream's elementwise fusion declines to fuse a producer with several
    consumers, since each would recompute it. At scene scale the alternative
    is worse: declining materializes a whole raster, so the design pays
    gigabytes of DRAM traffic to save one arithmetic unit.

    Above `min-elements` a producer is fused however many consumers it has;
    below it upstream's judgement stands. The pattern this rescues is the one
    every imaging chain ends with -- a phase ramp whose sine and cosine each
    feed both halves of a complex multiply.
  }];
  let dependentDialects = ["::mlir::linalg::LinalgDialect"];
  let options = [
    Option<"minElements", "min-elements", "uint64_t", /*default=*/"0",
           "Recompute rather than store results of at least this many "
           "elements">
  ];
}

def SARStageTransposes : Pass<"sar-stage-transposes", "::mlir::func::FuncOp"> {
  let summary = "Stage transposing loop nests through an on-chip block";
  let description = [{
    A transposing loop nest walks one buffer along its rows and the other
    along its columns, so whichever loop is innermost, one side strides by a
    whole row and gets a memory transaction per element. Staging a square
    block in between makes both sides contiguous: the block is read
    row-wise, then written out column-wise, and it is the only thing left
    striding.

    `block-bytes` bounds the block. The edge is the largest power of two
    that divides both loop extents and whose square fits, so a nest whose
    extents share no such factor is left alone.
  }];
  let dependentDialects = [
    "::mlir::affine::AffineDialect", "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect"
  ];
  let options = [
    Option<"blockBytes", "block-bytes", "unsigned", /*default=*/"524288",
           "On-chip bytes a staged block may occupy">
  ];
}

def SAREmitCInterface : Pass<"sar-emit-c-interface", "ModuleOp"> {
  let summary = "Request C interface wrappers for public functions";
  let description = [{
    Attaches `llvm.emit_c_interface` to every public function so kernel
    entry points get `_mlir_ciface_*` wrappers for the ctypes launcher.
    Private runtime-call declarations are left untouched: they resolve
    against plain C symbols.
  }];
}

#endif // SAR_DIALECT_SAR_TRANSFORMS_PASSES_TD
