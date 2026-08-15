// RUN: sar-opt %s -canonicalize | FileCheck %s
// Bit-exact folds only; see SARDialect.cpp.

// CHECK-LABEL: func.func @double_transpose
func.func @double_transpose(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NOT: sar.transpose
  // CHECK: return %arg0
  %0 = sar.transpose %x : tensor<4x8xf32> -> tensor<8x4xf32>
  %1 = sar.transpose %0 : tensor<8x4xf32> -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @shift_roundtrip
func.func @shift_roundtrip(%x: tensor<9xf64>) -> (tensor<9xf64>, tensor<9xf64>) {
  // CHECK-NOT: sar.fftshift
  // CHECK: return %arg0, %arg0
  %0 = sar.fftshift %x {dim = 0 : i64} : tensor<9xf64>
  %1 = sar.fftshift %0 {dim = 0 : i64, inverse} : tensor<9xf64>
  %2 = sar.fftshift %x {dim = 0 : i64, inverse} : tensor<9xf64>
  %3 = sar.fftshift %2 {dim = 0 : i64} : tensor<9xf64>
  return %1, %3 : tensor<9xf64>, tensor<9xf64>
}

// CHECK-LABEL: func.func @shift_different_axis_not_folded
func.func @shift_different_axis_not_folded(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: sar.fftshift
  // CHECK: sar.fftshift
  %0 = sar.fftshift %x {dim = 0 : i64} : tensor<4x8xf32>
  %1 = sar.fftshift %0 {dim = 1 : i64, inverse} : tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @same_direction_shift_not_folded
func.func @same_direction_shift_not_folded(%x: tensor<9xf32>) -> tensor<9xf32> {
  // CHECK: sar.fftshift
  // CHECK: sar.fftshift
  %0 = sar.fftshift %x {dim = 0 : i64} : tensor<9xf32>
  %1 = sar.fftshift %0 {dim = 0 : i64} : tensor<9xf32>
  return %1 : tensor<9xf32>
}

// CHECK-LABEL: func.func @mul_by_one
func.func @mul_by_one(%x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: sar.mul_scalar
  // CHECK: return %arg0
  %0 = sar.mul_scalar %x, 1.0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @mul_by_other_scalar_kept
func.func @mul_by_other_scalar_kept(%x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: sar.mul_scalar
  %0 = sar.mul_scalar %x, 2.0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @double_conj
func.func @double_conj(%z: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  // CHECK-NOT: sar.conj
  // CHECK: return %arg0
  %0 = sar.conj %z : tensor<4xcomplex<f32>>
  %1 = sar.conj %0 : tensor<4xcomplex<f32>>
  return %1 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: func.func @real_imag_of_complex
func.func @real_imag_of_complex(%re: tensor<4xf64>, %im: tensor<4xf64>)
    -> (tensor<4xf64>, tensor<4xf64>) {
  // CHECK-NOT: sar.complex
  // CHECK: return %arg0, %arg1
  %0 = sar.complex %re, %im : tensor<4xf64> -> tensor<4xcomplex<f64>>
  %1 = sar.real %0 : tensor<4xcomplex<f64>> -> tensor<4xf64>
  %2 = sar.imag %0 : tensor<4xcomplex<f64>> -> tensor<4xf64>
  return %1, %2 : tensor<4xf64>, tensor<4xf64>
}

// CHECK-LABEL: func.func @interp_dim0_normalizes
func.func @interp_dim0_normalizes(%z: tensor<8x4xcomplex<f64>>,
                                  %p: tensor<8x4xf64>)
    -> tensor<8x4xcomplex<f64>> {
  // CHECK: sar.transpose
  // CHECK: sar.transpose
  // CHECK: sar.interp1d %{{.*}}, %{{.*}} : (tensor<4x8xcomplex<f64>>, tensor<4x8xf64>)
  // CHECK: sar.transpose
  %0 = sar.interp1d %z, %p {dim = 0 : i64}
      : (tensor<8x4xcomplex<f64>>, tensor<8x4xf64>)
      -> (tensor<8x4xcomplex<f64>>)
  return %0 : tensor<8x4xcomplex<f64>>
}

// CHECK-LABEL: func.func @double_reverse
func.func @double_reverse(%x: tensor<8xf64>) -> tensor<8xf64> {
  // CHECK-NOT: sar.reverse
  // CHECK: return %arg0
  %0 = sar.reverse %x {dim = 0 : i64} : tensor<8xf64>
  %1 = sar.reverse %0 {dim = 0 : i64} : tensor<8xf64>
  return %1 : tensor<8xf64>
}
