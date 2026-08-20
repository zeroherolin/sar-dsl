// RUN: sar-opt %s --hls-pipeline="top-func=stage" | FileCheck %s

// Two-stage affine chain: a multiply loop feeding an add loop through a
// scratch buffer. The pipeline must extract two dataflow node functions,
// wire them with a token stream, and attach loop directives. (The two
// loops are deliberately not fused: each becomes its own dataflow node.)

// Two node functions must appear in the output.
// CHECK: func.func @stage_node0
// CHECK: func.func @stage_node1

// The dataflow wrapper must connect them with a stream.
// CHECK-LABEL: func.func @stage(
// CHECK-SAME:    top_func
// CHECK:         hls.dataflow.stream
// CHECK-COUNT-2: call @stage_node
// CHECK:         return

func.func @stage(%in: memref<32x32xf32>, %out: memref<32x32xf32>)
    attributes {top_func} {
  %scratch = memref.alloc() : memref<32x32xf32>
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      %v = affine.load %in[%i, %j] : memref<32x32xf32>
      %w = arith.mulf %v, %v : f32
      affine.store %w, %scratch[%i, %j] : memref<32x32xf32>
    }
  }
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      %v = affine.load %scratch[%i, %j] : memref<32x32xf32>
      %c = arith.constant 1.000000e+00 : f32
      %w = arith.addf %v, %c : f32
      affine.store %w, %out[%i, %j] : memref<32x32xf32>
    }
  }
  return
}
