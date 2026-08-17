// RUN: sar-opt %s --hls-place-dataflow-buffer -verify-diagnostics

// Placement retypes values in place; the only view it can re-infer is
// memref.subview. Any other view-like op is rejected by name instead of
// surfacing as a verifier type mismatch two passes later.

func.func @view_op_rejected(%v: f32) {
  %buf = hls.dataflow.buffer {depth = 1 : i32} : memref<4x16xf32>
  // expected-error @below {{cannot be retyped by buffer placement}}
  %flat = memref.collapse_shape %buf [[0, 1]]
      : memref<4x16xf32> into memref<64xf32>
  %c0 = arith.constant 0 : index
  memref.store %v, %flat[%c0] : memref<64xf32>
  return
}
