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
