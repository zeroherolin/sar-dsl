// RUN: sar-opt %s | FileCheck %s

// CHECK-LABEL: func.func @sort_dim1
func.func @sort_dim1(%x: tensor<8x16xf64>) -> tensor<8x16xf64> {
  // CHECK: sar.sort %{{.*}} {dim = 1 : i64} : tensor<8x16xf64>
  %0 = sar.sort %x {dim = 1 : i64} : tensor<8x16xf64>
  return %0 : tensor<8x16xf64>
}

// CHECK-LABEL: func.func @sort_dim0
func.func @sort_dim0(%x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK: sar.sort %{{.*}} {dim = 0 : i64} : tensor<8x16xf32>
  %0 = sar.sort %x {dim = 0 : i64} : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
