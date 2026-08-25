// RUN: sar-opt %s --hls-array-partition="bram-bytes=1000000 uram-bytes=36864" | FileCheck %s

// Re-binding a tier retypes the array. A top-level argument carries its type
// in the function signature as well, so the signature has to be re-derived
// from the entry block; walking the callees cannot repair it, because an
// argument the top function only reads itself reaches no call at all.

// CHECK-LABEL: func.func @rebalanced_arg
// CHECK-SAME:    #hls.mem<bram_t2p>
func.func @rebalanced_arg(%arg0: memref<8192xf32, #hls.mem<uram_t2p>>)
    attributes {top_func} {
  %c0 = arith.constant 0.0 : f32
  affine.for %i = 0 to 2048 {
    affine.store %c0, %arg0[%i * 4] : memref<8192xf32, #hls.mem<uram_t2p>>
    affine.store %c0, %arg0[%i * 4 + 1] : memref<8192xf32, #hls.mem<uram_t2p>>
    affine.store %c0, %arg0[%i * 4 + 2] : memref<8192xf32, #hls.mem<uram_t2p>>
    affine.store %c0, %arg0[%i * 4 + 3] : memref<8192xf32, #hls.mem<uram_t2p>>
  }
  return
}
