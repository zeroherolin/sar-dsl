// RUN: sar-opt %s --hls-create-dataflow-from-affine | FileCheck %s

// The dataflow hierarchy is seeded from loop structure: each top-level loop
// becomes a task (the future dataflow stage), and allocations are hoisted to
// the front of the dispatch so the buffers they define are visible to every
// task that will communicate through them.

// CHECK-LABEL: func.func @two_stage
func.func @two_stage(%in: memref<8xf32>, %out: memref<8xf32>) attributes {top_func} {
  // CHECK: hls.dataflow.dispatch
  // CHECK-NEXT: %[[TMP:.*]] = memref.alloc
  // CHECK: hls.dataflow.task
  // CHECK: arith.mulf
  %tmp = memref.alloc() : memref<8xf32>
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    %w = arith.mulf %v, %v : f32
    affine.store %w, %tmp[%i] : memref<8xf32>
  }
  // CHECK: hls.dataflow.task
  // CHECK: arith.addf
  affine.for %i = 0 to 8 {
    %v = affine.load %tmp[%i] : memref<8xf32>
    %w = arith.addf %v, %v : f32
    affine.store %w, %out[%i] : memref<8xf32>
  }
  return
}

// A loop-free function produces no dataflow task.
// CHECK-LABEL: func.func @no_loop
// CHECK-NOT: hls.dataflow.task
func.func @no_loop() attributes {top_func} {
  return
}
