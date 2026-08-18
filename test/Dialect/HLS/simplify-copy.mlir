// RUN: sar-opt %s --hls-simplify-copy | FileCheck %s

// A buffer that exists only to be copied into a port is a wasted plane and a
// wasted pass over the data: the writer can target the port directly. The
// buffer and the copy must both disappear.

// CHECK-LABEL: func.func @staging_buffer
func.func @staging_buffer(%in: memref<8xf32>, %out: memref<8xf32>) {
  // CHECK-NOT: hls.dataflow.buffer
  %buf = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    %w = arith.mulf %v, %v : f32
    // CHECK: affine.store %{{.*}}, %arg1[%{{.*}}]
    affine.store %w, %buf[%i] : memref<8xf32>
  }
  // CHECK-NOT: memref.copy
  memref.copy %buf, %out : memref<8xf32> to memref<8xf32>
  return
}

// Two block arguments are two fixed interfaces: neither can be substituted
// for the other, so a port-to-port copy is real data movement and must stay.

// CHECK-LABEL: func.func @port_to_port
// CHECK: memref.copy
func.func @port_to_port(%in: memref<8xf32>, %out: memref<8xf32>) {
  memref.copy %in, %out : memref<8xf32> to memref<8xf32>
  return
}

// Buffers observed before the copy are interchangeable only when their
// initial values are identical.

// CHECK-LABEL: func.func @different_initial_state
// CHECK: memref.copy
func.func @different_initial_state(%out: memref<2xf32>) {
  %source = hls.dataflow.buffer {depth = 1 : i32, init_value = 0.0 : f32}
      : memref<2xf32>
  %target = hls.dataflow.buffer {depth = 1 : i32} : memref<2xf32>
  %a = affine.load %source[0] : memref<2xf32>
  %b = affine.load %target[0] : memref<2xf32>
  affine.store %a, %out[0] : memref<2xf32>
  affine.store %b, %out[1] : memref<2xf32>
  memref.copy %source, %target : memref<2xf32> to memref<2xf32>
  return
}
