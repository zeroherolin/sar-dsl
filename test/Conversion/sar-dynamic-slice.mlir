// RUN: sar-opt %s --convert-sar-to-linalg | FileCheck %s

func.func @dynamic_windows(%input: tensor<16x16xf32>,
                           %update: tensor<4x16xf32>,
                           %row: tensor<1xi64>,
                           %zero: tensor<1xi64>) -> tensor<16x16xf32> {
  // CHECK-LABEL: func.func @dynamic_windows
  // CHECK-NOT: sar.dynamic_slice
  // CHECK: linalg.generic
  // CHECK: tensor.extract
  %tile = "sar.dynamic_slice"(%input, %row, %zero) <{sizes = array<i64: 4, 16>, strides = array<i64: 1, 1>}> : (tensor<16x16xf32>, tensor<1xi64>, tensor<1xi64>) -> tensor<4x16xf32>
  // CHECK-NOT: sar.dynamic_update_slice
  // CHECK: linalg.generic
  // CHECK: arith.select
  %result = "sar.dynamic_update_slice"(%input, %update, %row, %zero) : (tensor<16x16xf32>, tensor<4x16xf32>, tensor<1xi64>, tensor<1xi64>) -> tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}
