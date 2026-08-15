// RUN: sar-opt %s --convert-sar-interp-to-affine | FileCheck %s

// CHECK-LABEL: func.func @interp
func.func @interp(%re: tensor<8x16xf32>, %im: tensor<8x16xf32>,
                  %p: tensor<8x16xf64>)
    -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK: bufferization.to_buffer
  // CHECK: memref.alloc
  // CHECK: affine.for
  // CHECK: affine.for
  // CHECK: affine.load
  // 8 statically unrolled taps, each a masked (select) gather.
  // CHECK-COUNT-8: arith.select
  // CHECK: memref.load
  // CHECK: math.sin
  // CHECK: math.cos
  // CHECK-NOT: sar.interp1d_split
  // CHECK-NOT: scf.if
  %r, %i = sar.interp1d_split %re, %im, %p
      : (tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf64>)
      -> (tensor<8x16xf32>, tensor<8x16xf32>)
  return %r, %i : tensor<8x16xf32>, tensor<8x16xf32>
}
