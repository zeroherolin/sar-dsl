// RUN: sar-opt %s --hls-collapse-memref-unit-dims | FileCheck %s

// Function arguments are ABI values. Even a one-element result buffer must
// stay an array so writes are visible to the caller.

// CHECK-LABEL: func.func @collapse
// CHECK-SAME: (%{{.*}}: memref<1x8xf32>, %{{.*}}: memref<8x1xf32>)
func.func @collapse(%in: memref<1x8xf32>, %out: memref<8x1xf32>) {
  affine.for %i = 0 to 8 {
    // CHECK: affine.load %{{.*}}[0, %[[I:.*]]] : memref<1x8xf32>
    // CHECK: affine.store %{{.*}}, %{{.*}}[%[[I]], 0] : memref<8x1xf32>
    %v = affine.load %in[0, %i] : memref<1x8xf32>
    affine.store %v, %out[%i, 0] : memref<8x1xf32>
  }
  return
}

// Local dataflow buffers have no external ABI and can drop unit dimensions.

// CHECK-LABEL: func.func @collapse_local
// CHECK: %[[BUF:.*]] = hls.dataflow.buffer {{.*}} : memref<8xf32>
// CHECK: affine.store %{{.*}}, %[[BUF]][%[[I:.*]]] : memref<8xf32>
func.func @collapse_local(%value: f32) {
  %buf = hls.dataflow.buffer {depth = 1 : i32} : memref<1x8xf32>
  affine.for %i = 0 to 8 {
    affine.store %value, %buf[0, %i] : memref<1x8xf32>
  }
  return
}

// Collapsing the last dimension would turn a dataflow channel into a scalar
// passed by value after task outlining, so rank-1 unit buffers stay arrays.

// CHECK-LABEL: func.func @keep_scalar_channel
// CHECK: hls.dataflow.buffer {{.*}} : memref<1xf32>
func.func @keep_scalar_channel(%value: f32) {
  %buf = hls.dataflow.buffer {depth = 1 : i32} : memref<1xf32>
  affine.store %value, %buf[0] : memref<1xf32>
  return
}

// A non-affine user has no access map to rewrite, so collapsing under it
// would silently change what the op reads. The type must stay.

// CHECK-LABEL: func.func @blocked
// CHECK-SAME: memref<1x8xf32>
// CHECK: memref.copy %{{.*}} : memref<1x8xf32> to memref<1x8xf32>
func.func @blocked(%in: memref<1x8xf32>, %out: memref<1x8xf32>) {
  memref.copy %in, %out : memref<1x8xf32> to memref<1x8xf32>
  return
}
