// RUN: sar-opt %s --hls-array-partition="lutram-max-bits=0 lutram-bytes=0 bram-bytes=999999999 uram-bytes=999999999" | FileCheck %s

// A lowering that shaped an access pattern pins its banking through the
// hls.partition_* hint attributes. The pass applies them verbatim, consumes
// the attributes, and its automatic search must not override the layout.

// CHECK-LABEL: func.func @hinted
func.func @hinted() attributes {top_func} {
  // The hint becomes the layout; the attributes are consumed.
  // CHECK: hls.dataflow.buffer
  // CHECK-NOT: hls.partition_kinds
  // CHECK-SAME: memref<8x64xf32, #hls.partition<[complete, none], [8, 1]>, #hls.mem<bram_t2p>>
  %buf = hls.dataflow.buffer {depth = 1 : i32,
      hls.partition_kinds = ["complete", "none"],
      hls.partition_factors = [8, 1]}
      : memref<8x64xf32, #hls.mem<bram_t2p>>
  %zero = arith.constant 0.0 : f32
  // The single-lane access pattern would earn no lane banking on its own;
  // the hint is what carries the compact unrolled lanes' requirement.
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 64 {
      affine.store %zero, %buf[%i, %j]
          : memref<8x64xf32, #hls.mem<bram_t2p>>
    }
  }
  return
}
