// RUN: sar-opt %s --hls-lower-copy-to-affine | FileCheck %s
// RUN: sar-opt %s --hls-lower-copy-to-affine="internal-copy-only=true" \
// RUN:   | FileCheck %s --check-prefix=INTERNAL

// Lowering a copy to an affine nest is what exposes it to the loop
// optimizations -- but only an internal copy is marked as a point band for
// them: an external copy's loops are the AXI burst itself and must keep
// their shape.

// CHECK-LABEL: func.func @copies
func.func @copies(%src: memref<4x8xf32>, %ext: memref<4x8xf32, #hls.mem<dram>>) {
  // Internal copy: nest marked parallel *and* point.
  // CHECK: affine.for %{{.*}} = 0 to 4
  // CHECK: affine.for %{{.*}} = 0 to 8
  // CHECK: affine.load
  // CHECK-NEXT: affine.store
  // CHECK: } {parallel, point}
  // CHECK: } {parallel, point}
  %buf = hls.dataflow.buffer {depth = 1 : i32} : memref<4x8xf32>
  memref.copy %src, %buf : memref<4x8xf32> to memref<4x8xf32>

  // External copy: lowered too by default, but never marked point.
  // CHECK: affine.for %{{.*}} = 0 to 4
  // CHECK: } {parallel}
  // CHECK: } {parallel}
  // CHECK-NOT: memref.copy
  %ebuf = hls.dataflow.buffer {depth = 1 : i32} : memref<4x8xf32, #hls.mem<dram>>
  memref.copy %ext, %ebuf : memref<4x8xf32, #hls.mem<dram>> to memref<4x8xf32, #hls.mem<dram>>
  return
}

// With internal-copy-only the external copy is left intact -- it stays one
// raisable burst unit -- while the internal one still lowers.

// INTERNAL-LABEL: func.func @copies
// INTERNAL: } {parallel, point}
// INTERNAL: memref.copy %{{.*}} : memref<4x8xf32, #hls.mem<dram>>
