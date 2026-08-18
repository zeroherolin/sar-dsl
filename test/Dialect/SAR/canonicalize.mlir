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

// The shift calculus: element-wise ops commute with the pure permutation
// fftshift is. Both operands shifted the same way hoists the shift over
// the multiply, where the surrounding unshift cancels it -- the classic
// "multiply spectra in shifted form" pattern costs no rotation at all.

// CHECK-LABEL: func.func @shift_calculus_cancels
func.func @shift_calculus_cancels(%x: tensor<4x8xcomplex<f64>>,
                                  %h: tensor<4x8xcomplex<f64>>)
    -> tensor<4x8xcomplex<f64>> {
  // CHECK-NOT: sar.fftshift
  // CHECK: sar.mul %arg0, %arg1
  // CHECK-NOT: sar.fftshift
  %sx = sar.fftshift %x {dim = 1 : i64} : tensor<4x8xcomplex<f64>>
  %sh = sar.fftshift %h {dim = 1 : i64} : tensor<4x8xcomplex<f64>>
  %p = sar.mul %sx, %sh : tensor<4x8xcomplex<f64>>
  %u = sar.fftshift %p {dim = 1 : i64, inverse} : tensor<4x8xcomplex<f64>>
  return %u : tensor<4x8xcomplex<f64>>
}

// A unary op lets the shift through; mismatched axes or directions on a
// binary op do not.

// CHECK-LABEL: func.func @shift_over_unary
func.func @shift_over_unary(%x: tensor<4x8xcomplex<f64>>)
    -> tensor<4x8xcomplex<f64>> {
  // CHECK: sar.conj %arg0
  // CHECK-NOT: sar.fftshift
  %s = sar.fftshift %x {dim = 1 : i64} : tensor<4x8xcomplex<f64>>
  %c = sar.conj %s : tensor<4x8xcomplex<f64>>
  %u = sar.fftshift %c {dim = 1 : i64, inverse} : tensor<4x8xcomplex<f64>>
  return %u : tensor<4x8xcomplex<f64>>
}

// CHECK-LABEL: func.func @shift_axis_mismatch_stays
func.func @shift_axis_mismatch_stays(%x: tensor<4x8xf64>, %h: tensor<4x8xf64>)
    -> tensor<4x8xf64> {
  // CHECK-COUNT-2: sar.fftshift
  // CHECK: sar.mul
  %sx = sar.fftshift %x {dim = 0 : i64} : tensor<4x8xf64>
  %sh = sar.fftshift %h {dim = 1 : i64} : tensor<4x8xf64>
  %p = sar.mul %sx, %sh : tensor<4x8xf64>
  return %p : tensor<4x8xf64>
}

// Small constants rotate at compile time, bit for bit (numpy fftshift /
// ifftshift, which differ for odd sizes); reversal folds the same way.

// CHECK-LABEL: func.func @shift_constant_folds
func.func @shift_constant_folds() -> (tensor<5xf64>, tensor<5xf64>, tensor<4xf64>) {
  // CHECK: dense<[4.000000e+00, 5.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]>
  // CHECK: dense<[3.000000e+00, 4.000000e+00, 5.000000e+00, 1.000000e+00, 2.000000e+00]>
  // CHECK: dense<[4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]>
  // CHECK-NOT: sar.fftshift
  // CHECK-NOT: sar.reverse
  %c5 = sar.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf64>
  %c4 = sar.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>
  %s = sar.fftshift %c5 {dim = 0 : i64} : tensor<5xf64>
  %i = sar.fftshift %c5 {dim = 0 : i64, inverse} : tensor<5xf64>
  %r = sar.reverse %c4 {dim = 0 : i64} : tensor<4xf64>
  return %s, %i, %r : tensor<5xf64>, tensor<5xf64>, tensor<4xf64>
}
