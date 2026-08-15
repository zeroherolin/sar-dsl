// RUN: sar-opt %s --convert-sar-to-linalg | FileCheck %s

// Reversal reaches linalg as an indexing map rather than a gather.
// CHECK-DAG: #[[REV:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 7)>

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

// CHECK-LABEL: func.func @complex_create
func.func @complex_create(%re: tensor<4xf64>, %im: tensor<4xf64>)
    -> tensor<4xcomplex<f64>> {
  // CHECK: linalg.generic
  // CHECK: complex.create
  %0 = sar.complex %re, %im : tensor<4xf64> -> tensor<4xcomplex<f64>>
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

// CHECK-LABEL: func.func @conj_lowering
func.func @conj_lowering(%z: tensor<4x8xcomplex<f64>>)
    -> tensor<4x8xcomplex<f64>> {
  // CHECK: linalg.generic
  // CHECK: complex.conj
  %0 = sar.conj %z : tensor<4x8xcomplex<f64>>
  return %0 : tensor<4x8xcomplex<f64>>
}

// CHECK-LABEL: func.func @complex_create_atan2
func.func @complex_create_atan2(%a: tensor<8xf32>, %b: tensor<8xf32>)
    -> tensor<8xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: math.atan2
  // CHECK: linalg.generic
  // CHECK: complex.create
  %0 = sar.atan2 %a, %b : tensor<8xf32>
  %1 = sar.complex %0, %b : tensor<8xf32> -> tensor<8xcomplex<f32>>
  return %1 : tensor<8xcomplex<f32>>
}

// CHECK-LABEL: func.func @sum_reduce
func.func @sum_reduce(%x: tensor<4x8xf64>) -> tensor<4xf64> {
  // CHECK: linalg.fill
  // CHECK: linalg.reduce
  // CHECK: arith.addf
  %0 = sar.reduce %x {kind = "sum", dim = 1 : i64} : tensor<4x8xf64> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func.func @max_reduce_uses_select
func.func @max_reduce_uses_select(%x: tensor<4x8xf64>) -> tensor<8xf64> {
  // CHECK: linalg.reduce
  // CHECK: arith.cmpf ogt
  // CHECK: arith.select
  %0 = sar.reduce %x {kind = "max", dim = 0 : i64} : tensor<4x8xf64> -> tensor<8xf64>
  return %0 : tensor<8xf64>
}

// CHECK-LABEL: func.func @argmax_two_accumulators
func.func @argmax_two_accumulators(%x: tensor<4x8xf32>) -> tensor<4xi64> {
  // CHECK: linalg.generic
  // CHECK: linalg.index 1
  // CHECK: arith.cmpf ogt
  // CHECK: arith.select
  %0 = sar.argmax %x {dim = 1 : i64} : tensor<4x8xf32> -> tensor<4xi64>
  return %0 : tensor<4xi64>
}

// CHECK-LABEL: func.func @slice_gather
func.func @slice_gather(%x: tensor<4x8xf64>) -> tensor<2x4xf64> {
  // CHECK: linalg.generic
  // CHECK: linalg.index 0
  // CHECK: tensor.extract
  %0 = sar.slice %x {offsets = array<i64: 1, 2>, sizes = array<i64: 2, 4>, strides = array<i64: 1, 1>} : tensor<4x8xf64> -> tensor<2x4xf64>
  return %0 : tensor<2x4xf64>
}

// CHECK-LABEL: func.func @concat_select
func.func @concat_select(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<8xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.cmpi ult
  // CHECK: arith.select
  %0 = sar.concat %a, %b {dim = 0 : i64} : (tensor<4xf32>, tensor<4xf32>) -> (tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @pad_select
func.func @pad_select(%x: tensor<4xf32>) -> tensor<8xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.select
  %0 = sar.pad %x {low = array<i64: 2>, high = array<i64: 2>, value = 1.0} : tensor<4xf32> -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @cmp_where_selects
func.func @cmp_where_selects(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.cmpf ogt
  // CHECK: arith.select
  %0 = sar.cmp %a, %b {predicate = "gt"} : tensor<4xf32>
  // CHECK: linalg.generic
  // CHECK: arith.cmpf une
  // CHECK: arith.select
  %1 = sar.where %0, %a, %b : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @reverse_gather
func.func @reverse_gather(%x: tensor<4x8xf64>) -> tensor<4x8xf64> {
  // CHECK: linalg.generic {indexing_maps = [#[[REV]]
  // CHECK-NOT: tensor.extract
  %0 = sar.reverse %x {dim = 1 : i64} : tensor<4x8xf64>
  return %0 : tensor<4x8xf64>
}
