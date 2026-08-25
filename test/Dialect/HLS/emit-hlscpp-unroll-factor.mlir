// RUN: sar-translate --hls-emit-hlscpp %s | FileCheck %s

// `hls.unroll_factor` is how a loop that was left compact asks Vitis for
// partial unrolling: the pipelining pass attaches it when full unroll would
// exceed the operation budget, and the packing pass when a lane loop should
// stay one loop in source. Production designs carry a dozen of these, so
// the attribute reaching the emitter has to survive as a pragma.

// CHECK-LABEL: void unrolled(
// CHECK: #pragma HLS unroll factor=4
// CHECK: #pragma HLS pipeline II=1
func.func @unrolled(%in: memref<8x8xf32>, %out: memref<8x8xf32>)
    attributes {top_func} {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %v = affine.load %in[%i, %j] : memref<8x8xf32>
      %w = arith.mulf %v, %v : f32
      affine.store %w, %out[%i, %j] : memref<8x8xf32>
    } {hls.unroll_factor = 4 : i64,
       loop_directive = #hls.loop<pipeline = true, target_ii = 1>}
  }
  return
}
