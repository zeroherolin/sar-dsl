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

// CHECK-LABEL: func.func @stolt
func.func @stolt(%re: tensor<8x16xf64>, %im: tensor<8x16xf64>,
                 %fa: tensor<8xf64>, %fr: tensor<16xf64>)
    -> (tensor<8x16xf64>, tensor<8x16xf64>) {
  // Two loop nests: smoothing ramp, then mapping + gather + de-smoothing.
  // CHECK: affine.for
  // CHECK: math.cos
  // CHECK: math.sin
  // CHECK: affine.for
  // CHECK: math.sqrt
  // CHECK-NOT: sar.stolt_interp_split
  // CHECK-NOT: math.floor
  // CHECK-NOT: arith.maxnumf
  %r, %i = sar.stolt_interp_split %re, %im, %fa, %fr
      {c = 3.0e8, fc = 1.0e9, vr = 7000.0, t_shift = 1.0e-4}
      : (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8xf64>, tensor<16xf64>)
      -> (tensor<8x16xf64>, tensor<8x16xf64>)
  return %r, %i : tensor<8x16xf64>, tensor<8x16xf64>
}
