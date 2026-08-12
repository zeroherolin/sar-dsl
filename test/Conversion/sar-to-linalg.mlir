// RUN: sar-opt %s --convert-sar-to-linalg | FileCheck %s

// CHECK-LABEL: func.func @binary
func.func @binary(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: tensor.empty()
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK: linalg.yield
  %0 = sar.add %a, %b : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @complex_mul
func.func @complex_mul(%a: tensor<4xcomplex<f32>>, %b: tensor<4xcomplex<f32>>)
    -> tensor<4xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.mul
  %0 = sar.mul %a, %b : tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: func.func @expj
func.func @expj(%p: tensor<4xf64>) -> tensor<4xcomplex<f64>> {
  // CHECK: linalg.generic
  // CHECK-DAG: math.cos
  // CHECK-DAG: math.sin
  // CHECK: complex.create
  %0 = sar.expj %p : tensor<4xf64> -> tensor<4xcomplex<f64>>
  return %0 : tensor<4xcomplex<f64>>
}

// CHECK-LABEL: func.func @transpose
func.func @transpose(%x: tensor<4x8xf32>) -> tensor<8x4xf32> {
  // CHECK: linalg.transpose
  // CHECK-SAME: permutation = [1, 0]
  %0 = sar.transpose %x : tensor<4x8xf32> -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @broadcast
func.func @broadcast(%v: tensor<8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.broadcast
  // CHECK-SAME: dimensions = [0]
  %0 = sar.broadcast %v {dim = 1 : i64} : tensor<8xf32> -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @fftshift
func.func @fftshift(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: linalg.index
  // CHECK: arith.remui
  // CHECK: tensor.extract
  %0 = sar.fftshift %x {dim = 1 : i64} : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @constant
func.func @constant() -> tensor<2xf32> {
  // CHECK: arith.constant dense<[1.000000e+00, 2.000000e+00]>
  %0 = sar.constant dense<[1.0, 2.0]> : tensor<2xf32>
  return %0 : tensor<2xf32>
}
