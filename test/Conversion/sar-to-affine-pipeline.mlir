// RUN: sar-opt %s --sar-to-affine-pipeline | FileCheck %s

// The HLS hand-off: complex tensors are split into real/imaginary planes,
// FFT and interpolation become affine loop nests, and nothing of the `sar`
// or `linalg` level survives.
// CHECK-NOT: sar.
// CHECK-NOT: linalg.

// CHECK-LABEL: func.func @fft_and_interp
// CHECK-SAME: memref<8x16xf32>
func.func @fft_and_interp(%data: tensor<8x16xcomplex<f32>>,
                           %pos: tensor<8x16xf64>)
    -> tensor<8x16xcomplex<f32>> {
  // CHECK: memref.alloc
  // CHECK: affine.for
  // CHECK: affine.load
  // Interpolation taps are masked with selects rather than branches.
  // CHECK: arith.select
  // CHECK-NOT: scf.if
  %0 = sar.fft %data {dim = 1 : i64} : tensor<8x16xcomplex<f32>>
  %1 = sar.interp1d %0, %pos {kernel = "sinc", taps = 8 : i64}
      : (tensor<8x16xcomplex<f32>>, tensor<8x16xf64>)
      -> tensor<8x16xcomplex<f32>>
  return %1 : tensor<8x16xcomplex<f32>>
}
