// RUN: sar-opt %s --hls-affine-loop-tile="tile-size=8 tile-buffer-bytes=2048" | FileCheck %s

// A pass that streams DRAM contiguously keeps its sweep whole: tiling the
// dimension every access agrees on would turn one long AXI burst per row
// into a burst per tile, and the dimension carries no reuse to make up for
// it. Both loops survive at their full extent.

// CHECK-LABEL: func.func @streaming
// CHECK: affine.for %{{.*}} = 0 to 64
// CHECK: affine.for %{{.*}} = 0 to 64
// CHECK-NOT: affine.for
func.func @streaming(%in: memref<64x64xf32, #hls.mem<dram>>,
                     %out: memref<64x64xf32, #hls.mem<dram>>) {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %v = affine.load %in[%i, %j] : memref<64x64xf32, #hls.mem<dram>>
      %w = arith.mulf %v, %v : f32
      affine.store %w, %out[%i, %j] : memref<64x64xf32, #hls.mem<dram>>
    }
  }
  return
}

// -----

// A transpose has no such dimension: one buffer is contiguous along the row
// and the other along the column. Tiling is the answer there, and the edge
// comes from the block the band has to stage -- 2048 bytes of f32 over two
// dimensions gives a 16 x 16 tile.

// CHECK-LABEL: func.func @transpose
// CHECK: affine.for %{{.*}} = 0 to 4
// CHECK: affine.for %{{.*}} = 0 to 4
// CHECK: affine.for %{{.*}} = 0 to 16
// CHECK: affine.for %{{.*}} = 0 to 16
func.func @transpose(%in: memref<64x64xf32, #hls.mem<dram>>,
                     %out: memref<64x64xf32, #hls.mem<dram>>) {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %v = affine.load %in[%j, %i] : memref<64x64xf32, #hls.mem<dram>>
      affine.store %v, %out[%i, %j] : memref<64x64xf32, #hls.mem<dram>>
    }
  }
  return
}

// -----

// The accesses agree on which dimension they stream, but that loop is the
// outer one -- so every step of the inner loop jumps a whole row. Putting
// the streaming loop innermost is what makes the sweep contiguous, and it
// costs nothing; burst length, by contrast, no later pass can recover.

// CHECK-LABEL: func.func @streaming_loop_is_outermost
// CHECK: affine.for %[[I:.*]] = 0 to 64
// CHECK: affine.apply #map(%{{.*}}, %[[I]])
// CHECK: affine.for %[[J:.*]] = 0 to 64
// CHECK: affine.apply #map(%{{.*}}, %[[J]])
func.func @streaming_loop_is_outermost(%in: memref<64x64xf32, #hls.mem<dram>>,
                                       %out: memref<64x64xf32, #hls.mem<dram>>) {
  affine.for %j = 0 to 64 {
    affine.for %i = 0 to 64 {
      %v = affine.load %in[%i, %j] : memref<64x64xf32, #hls.mem<dram>>
      affine.store %v, %out[%i, %j] : memref<64x64xf32, #hls.mem<dram>>
    }
  }
  return
}

// -----

// Lowering-proven scheduling bounds describe point iterations and must remain
// attached to the corresponding point loop when a surrounding band is tiled.

// CHECK-LABEL: func.func @minimum_ii
// CHECK: affine.for %{{.*}} = 0 to 8
// CHECK: affine.for %{{.*}} = 0 to 8
// CHECK: affine.for %{{.*}} = 0 to 8
// CHECK: affine.for %{{.*}} = 0 to 8
// CHECK: } {hls.min_ii = 8
func.func @minimum_ii(%in: memref<64x64xf32>,
                      %out: memref<64x64xf32>) {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %bias = affine.load %in[0, 0] : memref<64x64xf32>
      %v = affine.load %in[%i, %j] : memref<64x64xf32>
      %sum = arith.addf %v, %bias : f32
      affine.store %sum, %out[%i, %j] : memref<64x64xf32>
    } {hls.min_ii = 8 : i64}
  }
  return
}
