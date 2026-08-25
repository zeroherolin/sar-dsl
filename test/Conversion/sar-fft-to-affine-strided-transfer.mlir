// RUN: sar-opt %s --convert-sar-fft-to-affine='fft-parallel-rows=4 fft-io-unroll=8' \
// RUN:   | FileCheck %s
// RUN: sar-opt %s \
// RUN:   --convert-sar-fft-to-affine="fft-parallel-rows=3 fft-io-unroll=4" \
// RUN:   | FileCheck %s --check-prefix=DIVIDES

// A slow-axis transform stages a complete eight-element transfer word while
// retaining four butterfly lanes. Only the prefetch/write-back blocks grow to
// eight lines; the intermediate Stockham storage remains four lines wide.
// CHECK-LABEL: func.func @slow_axis
// CHECK-COUNT-4: memref.alloc() {hls.partition_factors = [8, 1], hls.partition_kinds = ["complete", "none"]} : memref<8x64xf64>
// CHECK-COUNT-4: memref.alloc() {hls.partition_factors = [4, 1], hls.partition_kinds = ["complete", "none"]} : memref<4x64xf64>
// CHECK: affine.for {{.*}} = 0 to 8 step 8 {
// CHECK: affine.for {{.*}} = 0 to 8 step 4 {

func.func @slow_axis(%re: tensor<64x8xf64>, %im: tensor<64x8xf64>)
    -> (tensor<64x8xf64>, tensor<64x8xf64>) {
  %r, %i = sar.fft_split %re, %im {dim = 0 : i64} : tensor<64x8xf64>
  return %r, %i : tensor<64x8xf64>, tensor<64x8xf64>
}

// A slow-axis transfer block spans max(lanes, io) lines while the compute
// engine walks it `lanes` at a time. A pinned lane count that does not
// divide that block would leave the final sub-block running past its end,
// so the lanes are reduced to a divisor. The autotuner only offers powers
// of two, where this already holds; the guard is for a pinned option.

// DIVIDES-LABEL: func.func @slow_axis_pinned_lanes
// DIVIDES: memref.alloc() {{.*}} : memref<4x64xf64>
// DIVIDES-NOT: step 3
func.func @slow_axis_pinned_lanes(%re: tensor<64x12xf64>, %im: tensor<64x12xf64>)
    -> (tensor<64x12xf64>, tensor<64x12xf64>) {
  %r, %i = sar.fft_split %re, %im {dim = 0 : i64} : tensor<64x12xf64>
  return %r, %i : tensor<64x12xf64>, tensor<64x12xf64>
}
