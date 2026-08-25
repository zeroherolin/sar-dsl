// RUN: sar-opt %s --convert-sar-interp-to-affine --canonicalize | FileCheck %s

// Position expressions composed from pure element-wise operations are sunk
// into the gather loop instead of materializing a full f64 plane. The
// transpose matches the normalized dim-0 interpolation used by polar
// regridding, and atan2 exercises its non-linear position expression.

// CHECK-LABEL: func.func @atan2_positions
// CHECK-NOT: bufferization.to_buffer
// CHECK-NOT: sar.transpose
// CHECK-NOT: sar.atan2
// CHECK: affine.for
// CHECK: affine.for
// CHECK: math.atan2
func.func @atan2_positions(%re: tensor<8x16xf32>, %im: tensor<8x16xf32>)
    -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  %y_axis = sar.constant dense<[
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
      0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    ]> : tensor<16xf64>
  %x_axis = sar.constant dense<[
      1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7
    ]> : tensor<8xf64>
  %y = sar.broadcast %y_axis {dim = 0 : i64}
      : tensor<16xf64> -> tensor<16x8xf64>
  %x = sar.broadcast %x_axis {dim = 1 : i64}
      : tensor<8xf64> -> tensor<16x8xf64>
  %angles = sar.atan2 %y, %x : tensor<16x8xf64>
  %positions = sar.transpose %angles
      : tensor<16x8xf64> -> tensor<8x16xf64>
  %r, %i = sar.interp1d_split %re, %im, %positions
      : (tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf64>)
      -> (tensor<8x16xf32>, tensor<8x16xf32>)
  return %r, %i : tensor<8x16xf32>, tensor<8x16xf32>
}
