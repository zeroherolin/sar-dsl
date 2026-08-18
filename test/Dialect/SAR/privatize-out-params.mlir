// RUN: sar-opt %s --sar-privatize-out-params | FileCheck %s

// An out-parameter that several stages write (or that any stage reads
// back) becomes a local buffer with one final copy to the port: a
// multi-writer top-level port would forfeit dataflow for the design.

// CHECK-LABEL: func.func @multi_writer
func.func @multi_writer(%in: memref<8xf32>, %out: memref<8xf32>)
    attributes {sar.arg_names = ["x"]} {
  // CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<8xf32>
  // CHECK: affine.store %{{.*}}, %[[LOCAL]]
  // CHECK: affine.store %{{.*}}, %[[LOCAL]]
  // CHECK: memref.copy %[[LOCAL]], %arg1
  // CHECK-NEXT: return
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
  }
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    %w = arith.mulf %v, %v : f32
    affine.store %w, %out[%i] : memref<8xf32>
  }
  return
}

// A port written by exactly one stage and never read keeps the direct
// write; no local buffer, no copy.

// CHECK-LABEL: func.func @single_writer
func.func @single_writer(%in: memref<8xf32>, %out: memref<8xf32>)
    attributes {sar.arg_names = ["x"]} {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
  }
  return
}

// A read-back through the out-param (compute in place) is privatized even
// with a single writing loop.

// CHECK-LABEL: func.func @read_back
func.func @read_back(%in: memref<8xf32>, %out: memref<8xf32>)
    attributes {sar.arg_names = ["x"]} {
  // CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<8xf32>
  // CHECK: memref.copy %[[LOCAL]], %arg1
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
    %w = affine.load %out[%i] : memref<8xf32>
    %s = arith.addf %w, %w : f32
    affine.store %s, %out[%i] : memref<8xf32>
  }
  return
}

// Inputs (named in sar.arg_names) are never touched, and a function
// without the attribute is left alone.

// CHECK-LABEL: func.func @no_arg_names
func.func @no_arg_names(%in: memref<8xf32>, %out: memref<8xf32>) {
  // CHECK-NOT: memref.alloc
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
    %w = affine.load %out[%i] : memref<8xf32>
    affine.store %w, %out[%i] : memref<8xf32>
  }
  return
}
