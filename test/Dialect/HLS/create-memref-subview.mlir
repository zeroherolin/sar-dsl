// RUN: sar-opt %s --hls-create-memref-subview | FileCheck %s

// A tiled band accessing `src[ti*16+i][tj*16+j]` reads one 16x16 tile per
// tile iteration: the point loops move to a subview at the tile offset,
// and the buffer records the tile layout for later staging decisions.

// CHECK: #map = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: func.func @tiled
// CHECK-SAME: {hls.tile_layout = #hls.tile<[16, 16], [1, 1]>}
func.func @tiled(%src: memref<64x64xf32>) attributes {top_func} {
  %c0 = arith.constant 0.0 : f32
  affine.for %ti = 0 to 4 {
    affine.for %tj = 0 to 4 {
      // CHECK: %[[I:.*]] = affine.apply #map(%arg1)
      // CHECK: %[[J:.*]] = affine.apply #map(%arg2)
      // CHECK: %[[SV:.*]] = memref.subview %arg0[%[[I]], %[[J]]] [16, 16] [1, 1]
      // CHECK: affine.store %{{.*}}, %[[SV]]
      affine.for %i = 0 to 16 {
        affine.for %j = 0 to 16 {
          affine.store %c0, %src[%ti * 16 + %i, %tj * 16 + %j] : memref<64x64xf32>
        } {point}
      } {point}
    }
  }
  return
}

// A band whose access does not decompose into a tile offset plus point
// indices is left untouched.

// CHECK-LABEL: func.func @not_tiled
func.func @not_tiled(%src: memref<64x64xf32>) attributes {top_func} {
  // CHECK-NOT: memref.subview
  %c0 = arith.constant 0.0 : f32
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.store %c0, %src[%i, %j] : memref<64x64xf32>
    }
  }
  return
}
