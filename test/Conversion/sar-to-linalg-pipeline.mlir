// RUN: sar-opt %s --sar-to-linalg-pipeline | FileCheck %s

// CHECK-LABEL: func.func @elementwise_chain
// CHECK-NOT: sar.
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-NOT: sar.
func.func @elementwise_chain(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
  %0 = sar.add %a, %b : tensor<4x8xf32>
  %1 = sar.mul_scalar %0, 2.0 : tensor<4x8xf32>
  %2 = sar.sqrt %1 : tensor<4x8xf32>
  return %2 : tensor<4x8xf32>
}
