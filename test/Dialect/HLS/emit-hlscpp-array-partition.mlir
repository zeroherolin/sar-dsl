// RUN: sar-translate --hls-emit-hlscpp -emit-vitis-directives %s | FileCheck %s

// Partition pragmas have to match what Vitis 2022.2 accepts. A `cyclic` or
// `block` partition is sized by a factor; a `complete` one splits every
// element into its own register and takes none. Emitting `complete factor=N`
// makes the tool warn (HLS 207-5529) and drop the directive, which silently
// costs the banking the pass decided the array needed.

// CHECK-LABEL: void top(
// CHECK:       #pragma HLS array_partition variable=inout0 complete dim=1
// CHECK-NEXT:  #pragma HLS array_partition variable=inout0 cyclic factor=4 dim=2
// CHECK-NOT:   complete factor
module {
  func.func @top(%arg0: memref<8x64xf32, #hls.partition<[complete, cyclic], [8, 4]>,
                                        #hls.mem<bram_t2p>>) attributes {top_func} {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 64 {
        %v = affine.load %arg0[%i, %j]
            : memref<8x64xf32, #hls.partition<[complete, cyclic], [8, 4]>,
                     #hls.mem<bram_t2p>>
        %w = arith.mulf %v, %v : f32
        affine.store %w, %arg0[%i, %j]
            : memref<8x64xf32, #hls.partition<[complete, cyclic], [8, 4]>,
                     #hls.mem<bram_t2p>>
      }
    }
    return
  }

}
