// RUN: sar-opt %s --convert-sar-fft-to-affine | FileCheck %s

// Non-power-of-two sizes go through Bluestein's chirp-z reduction: the DFT
// becomes a convolution that a padded power-of-two Stockham transform can
// carry. For n = 12 the padded length is M = 32 (next power of two above
// 2n-2), so the twiddle tables are the M-sized ones.
//
// Both the chirp w[j] = exp(-i pi j^2 / n) and the kernel spectrum
// B = FFT_M(b) are folded on the host, so no runtime transform of the
// kernel appears in the emitted code.

// CHECK-DAG: memref.global "private" constant @__sar_bluestein_chirp_cos_12_f64 : memref<12xf64>
// CHECK-DAG: memref.global "private" constant @__sar_bluestein_chirp_sin_12_f64 : memref<12xf64>
// CHECK-DAG: memref.global "private" constant @__sar_bluestein_kernel_re_12_f64 : memref<32xf64>
// CHECK-DAG: memref.global "private" constant @__sar_bluestein_kernel_im_12_f64 : memref<32xf64>
// CHECK-DAG: memref.global "private" constant @__sar_fft_twiddle_cos_32_f64 : memref<33xf64>

// CHECK-LABEL: func.func @fft_split_bluestein
func.func @fft_split_bluestein(%re: tensor<4x12xf64>, %im: tensor<4x12xf64>)
    -> (tensor<4x12xf64>, tensor<4x12xf64>) {
  // Scratch is one padded line wide, not a full raster: the working set is
  // O(M) whatever the scene height.
  // CHECK: memref.alloc() : memref<32xf64>
  // CHECK: affine.for
  // CHECK: affine.load
  // Straight-line bodies only: nothing for HLS to serialize.
  // CHECK-NOT: scf.if
  // CHECK-NOT: sar.fft_split
  %r, %i = sar.fft_split %re, %im {dim = 1 : i64} : tensor<4x12xf64>
  return %r, %i : tensor<4x12xf64>, tensor<4x12xf64>
}
