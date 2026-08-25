// RUN: sar-opt %s --hls-loop-pipelining="target-ii=2" | FileCheck %s
// RUN: sar-opt %s --hls-loop-pipelining="pipeline-level=2" \
// RUN:   | FileCheck %s --check-prefix=DEEP
// RUN: sar-opt %s --hls-loop-pipelining="pipeline-level=1" \
// RUN:   | FileCheck %s --check-prefix=LEVEL1
// RUN: sar-opt %s --hls-loop-pipelining="target-ii=1" \
// RUN:   | FileCheck %s --check-prefix=MINII
// RUN: sar-opt %s \
// RUN:   --hls-loop-pipelining="pipeline-level=1 max-unrolled-ops=12" \
// RUN:   | FileCheck %s --check-prefix=BUDGET
// RUN: sar-opt %s --hls-loop-pipelining="pipeline-level=1 \
// RUN:   max-unrolled-ops=12 max-unroll-factor=2" \
// RUN:   | FileCheck %s --check-prefix=FACTOR

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

// An operation budget below the cost of materializing the body keeps the
// loop compact and asks Vitis for a bounded factor instead: the 8-iteration
// body costs 3 operations a copy, so 12 affords 4 of them.

// BUDGET: affine.for %{{.*}} = 0 to 8
// BUDGET: affine.for %{{.*}} = 0 to 8
// BUDGET: } {hls.unroll_factor = 4 : i64}

// The factor cap bounds that pragma independently of the budget, and is
// rounded down to a power of two.

// FACTOR: affine.for %{{.*}} = 0 to 8
// FACTOR: affine.for %{{.*}} = 0 to 8
// FACTOR: } {hls.unroll_factor = 2 : i64}
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

// LEVEL1-LABEL: func.func @minimum_ii
// MINII-LABEL: func.func @minimum_ii
// MINII: loop_directive = #hls.loop<pipeline = true, target_ii = 2>
func.func @minimum_ii(%in: memref<8xf32>, %out: memref<8xf32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
  } {hls.min_ii = 2 : i64}
  return
}

// Pipelining above a two-deep nest has to flatten both levels below it.
// Unrolling clones a body -- inner loops included -- into the parent, and
// the walk cannot descend into the loop it just erased, so the sweep
// repeats until nothing changes. Stopping after one pass would leave inner
// loops standing inside a body marked pipelined, contradicting the pass.

// DEEP-LABEL: func.func @deep
// DEEP: affine.for %{{.*}} = 0 to 4
// DEEP-COUNT-16: arith.mulf
// DEEP: } {loop_directive = #hls.loop<pipeline = true, target_ii = 1>}
// DEEP-NOT: affine.for
func.func @deep(%in: memref<4x4x4xf32>, %out: memref<4x4x4xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      affine.for %k = 0 to 4 {
        %v = affine.load %in[%i, %j, %k] : memref<4x4x4xf32>
        %w = arith.mulf %v, %v : f32
        affine.store %w, %out[%i, %j, %k] : memref<4x4x4xf32>
      }
    }
  }
  return
}
