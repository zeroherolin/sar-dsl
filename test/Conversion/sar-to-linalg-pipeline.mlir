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

// A transform and a gather are not structured loop nests over a fixed index
// space, so they have no linalg form and become runtime calls -- the same
// ABI the CPU path uses. Nothing `sar.` may survive, or a backend consuming
// this level would have to implement the dialect too.

// CHECK-LABEL: func.func @signal_chain
// CHECK-NOT: sar.
// CHECK: call @sar_rt_fft_2d_c128
// CHECK: call @sar_rt_interp1d_2d_c128
// CHECK-NOT: sar.
func.func @signal_chain(%z: tensor<8x8xcomplex<f64>>, %p: tensor<8x8xf64>)
    -> tensor<8x8xcomplex<f64>> {
  %0 = sar.fft %z {dim = 1 : i64} : tensor<8x8xcomplex<f64>>
  %1 = sar.interp1d %0, %p
      : (tensor<8x8xcomplex<f64>>, tensor<8x8xf64>) -> tensor<8x8xcomplex<f64>>
  return %1 : tensor<8x8xcomplex<f64>>
}
