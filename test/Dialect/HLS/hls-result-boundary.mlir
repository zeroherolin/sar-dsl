// RUN: sar-opt %s --hls-pipeline="top-func=boundary axi-interface=false loop-tile-size=1" | sar-translate --hls-emit-hlscpp | FileCheck %s

// A result that also feeds another result stays behind a local buffer. The
// emitted public ABI must contain two write-only outputs, never an inout
// working port.

// CHECK: SAR_DSL_INTERFACE: {"name":"in0","protocol":"bram","direction":"in"
// CHECK-COUNT-2: "direction":"out","kind":"public"
// CHECK-NOT: "direction":"inout"
// CHECK: void boundary(
func.func @boundary(%input: memref<16xf32>, %out: memref<16xf32>,
                    %extra: memref<16xf32>)
    attributes {sar.arg_names = ["input"]} {
  %buffer = memref.alloc() : memref<16xf32>
  affine.for %i = 0 to 16 {
    %value = affine.load %input[%i] : memref<16xf32>
    affine.store %value, %buffer[%i] : memref<16xf32>
  }
  affine.for %i = 0 to 16 {
    %value = affine.load %buffer[%i] : memref<16xf32>
    affine.store %value, %out[%i] : memref<16xf32>
  }
  affine.for %i = 0 to 16 {
    %value = affine.load %buffer[%i] : memref<16xf32>
    %two = arith.constant 2.0 : f32
    %scaled = arith.mulf %value, %two : f32
    affine.store %scaled, %extra[%i] : memref<16xf32>
  }
  return
}
