// RUN: sar-opt %s --convert-sar-fft-to-affine | FileCheck %s

// Each Stockham stage owns only the twiddles it reads. For N=8 the leading
// radix-2 stage has four entries and the radix-4 stage has three.
// CHECK-DAG: memref.global "private" constant @__sar_fft_twiddle_cos_8_s0_f64 : memref<4xf64>
// CHECK-DAG: memref.global "private" constant @__sar_fft_twiddle_sin_8_s0_f64 : memref<4xf64>
// CHECK-DAG: memref.global "private" constant @__sar_fft_twiddle_cos_8_s1_f64 : memref<3xf64>
// CHECK-DAG: memref.global "private" constant @__sar_fft_twiddle_sin_8_s1_f64 : memref<3xf64>

// CHECK-LABEL: func.func @fft_split_2d
func.func @fft_split_2d(%re: tensor<4x8xf64>, %im: tensor<4x8xf64>)
    -> (tensor<4x8xf64>, tensor<4x8xf64>) {
  // CHECK: bufferization.to_buffer
  // A serial single-beat transform leaves banking to the automatic
  // search: hints are pinned, so only parallel structure earns them.
  // CHECK: memref.get_global @__sar_fft_twiddle_cos_8_s0_f64
  // CHECK-NOT: hls.partition_kinds
  // Line-block loop, then prefetch sweep, mixed-radix stages (8 = 2 * 4)
  // and write-back sweep inside it.
  // CHECK: affine.for %{{.*}} = 0 to 4 {
  // CHECK-COUNT-5: affine.for
  // CHECK: arith.mulf
  // CHECK: affine.for
  // CHECK-NOT: sar.fft_split
  %r, %i = sar.fft_split %re, %im {dim = 1 : i64} : tensor<4x8xf64>
  return %r, %i : tensor<4x8xf64>, tensor<4x8xf64>
}

// CHECK-LABEL: func.func @ifft_scales
func.func @ifft_scales(%re: tensor<16xf32>, %im: tensor<16xf32>)
    -> (tensor<16xf32>, tensor<16xf32>) {
  // The inverse transform multiplies by 1/16 on the final stage's stores.
  // CHECK: arith.constant 6.250000e-02
  %r, %i = sar.fft_split %re, %im {dim = 0 : i64, inverse} : tensor<16xf32>
  return %r, %i : tensor<16xf32>, tensor<16xf32>
}
