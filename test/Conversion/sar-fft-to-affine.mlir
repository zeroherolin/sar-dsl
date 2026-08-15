// RUN: sar-opt %s --convert-sar-fft-to-affine | FileCheck %s

// Twiddle tables are private constant globals of length L-1.
// CHECK-DAG: memref.global "private" constant @__sar_fft_twiddle_cos_8_f64 : memref<7xf64>
// CHECK-DAG: memref.global "private" constant @__sar_fft_twiddle_sin_8_f64 : memref<7xf64>

// CHECK-LABEL: func.func @fft_split_2d
func.func @fft_split_2d(%re: tensor<4x8xf64>, %im: tensor<4x8xf64>)
    -> (tensor<4x8xf64>, tensor<4x8xf64>) {
  // CHECK: bufferization.to_buffer
  // CHECK: memref.get_global @__sar_fft_twiddle_cos_8_f64
  // 3 unrolled Stockham stages (log2 8) + input and output copies.
  // CHECK-COUNT-5: affine.for
  // CHECK: affine.load
  // CHECK: arith.mulf
  // CHECK-NOT: sar.fft_split
  %r, %i = sar.fft_split %re, %im {dim = 1 : i64} : tensor<4x8xf64>
  return %r, %i : tensor<4x8xf64>, tensor<4x8xf64>
}

// CHECK-LABEL: func.func @ifft_scales
func.func @ifft_scales(%re: tensor<16xf32>, %im: tensor<16xf32>)
    -> (tensor<16xf32>, tensor<16xf32>) {
  // The inverse transform multiplies by 1/16 in the output copy.
  // CHECK: arith.constant 6.250000e-02
  %r, %i = sar.fft_split %re, %im {dim = 0 : i64, inverse} : tensor<16xf32>
  return %r, %i : tensor<16xf32>, tensor<16xf32>
}
