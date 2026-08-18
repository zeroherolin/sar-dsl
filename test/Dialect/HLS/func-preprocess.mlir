// RUN: sar-opt %s --hls-func-preprocess="top-func=top" | FileCheck %s

// Everything downstream of preprocessing reasons in affine terms: raising the
// memref/arith index forms the frontend leaves behind is what makes the loop
// and memory analyses see the accesses at all. Stack allocas become dataflow
// buffers -- an HLS kernel has no stack -- and only the function named by
// top-func is stamped top_func, since the pipeline keys the interface on it.

// CHECK-LABEL: func.func @top
// CHECK-SAME: attributes {top_func}
func.func @top(%in: memref<8xf32>) {
  %tmp = memref.alloca() : memref<8xf32>
  %c1 = arith.constant 1 : index
  // CHECK: hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
  // CHECK: affine.for %[[I:.*]] = 0 to 7
  affine.for %i = 0 to 7 {
    // CHECK: %[[J:.*]] = affine.apply #map(%[[I]], %{{.*}})
    %j = arith.addi %i, %c1 : index
    // CHECK: %[[V:.*]] = affine.load %{{.*}}[%[[I]]]
    %v = memref.load %in[%i] : memref<8xf32>
    // CHECK: affine.store %[[V]], %{{.*}}[%[[J]]]
    memref.store %v, %tmp[%j] : memref<8xf32>
    // CHECK-NOT: memref.load
    // CHECK-NOT: memref.store
    // CHECK-NOT: memref.alloca
  }
  return
}

// A function not named by top-func is still preprocessed (parallel loops get
// marked) but must not be stamped: two top functions would be two interface
// roots.

// CHECK-LABEL: func.func @other
// CHECK-NOT: top_func
// CHECK: {parallel}
func.func @other(%m: memref<8xf32>) {
  %c = arith.constant 0.0 : f32
  affine.for %i = 0 to 8 {
    affine.store %c, %m[%i] : memref<8xf32>
  }
  return
}

// Dynamic sizes and alignment must survive stack-to-heap demotion; they
// cannot be represented by hls.dataflow.buffer.
// CHECK-LABEL: func.func @dynamic_alloca
// CHECK: memref.alloc(%{{.*}}) {alignment = 64 : i64} : memref<?xf32>
// CHECK-NOT: hls.dataflow.buffer
func.func @dynamic_alloca(%n: index) {
  %0 = memref.alloca(%n) {alignment = 64 : i64} : memref<?xf32>
  %c0 = arith.constant 0 : index
  %value = arith.constant 1.0 : f32
  memref.store %value, %0[%c0] : memref<?xf32>
  return
}
