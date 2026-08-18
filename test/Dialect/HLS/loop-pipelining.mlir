// RUN: sar-opt %s --hls-loop-pipelining="target-ii=2" | FileCheck %s
// RUN: sar-opt %s --hls-loop-pipelining="pipeline-level=1" \
// RUN:   | FileCheck %s --check-prefix=LEVEL1

// Pipelining is a whole-band decision: the chosen level gets the pipeline
// directive with the requested II, and every perfectly-nesting outer loop is
// marked too -- the directive's presence is what earns it `#pragma HLS
// dependence false` at emission (Vitis flattens perfect nests on its own).

// CHECK: affine.for %{{.*}} = 0 to 8
// CHECK: affine.for %{{.*}} = 0 to 8
// CHECK: } {loop_directive = #hls.loop<pipeline = true, target_ii = 2>}
// CHECK: } {loop_directive = #hls.loop<pipeline = false, target_ii = 1>}

// Pipelining one level up implies the loops below it must go: an inner loop
// inside a pipelined body cannot be pipelined itself, so it is fully unrolled
// (8 multiplies) and only the pipelined loop survives.

// LEVEL1: affine.for %{{.*}} = 0 to 8
// LEVEL1-COUNT-8: arith.mulf
// LEVEL1: } {loop_directive = #hls.loop<pipeline = true, target_ii = 1>}
// LEVEL1-NOT: affine.for
func.func @pipe(%in: memref<8x8xf32>, %out: memref<8x8xf32>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %v = affine.load %in[%i, %j] : memref<8x8xf32>
      %w = arith.mulf %v, %v : f32
      affine.store %w, %out[%i, %j] : memref<8x8xf32>
    }
  }
  return
}
