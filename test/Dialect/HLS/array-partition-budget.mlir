// RUN: ! sar-opt %s --hls-array-partition="lutram-max-bits=0 lutram-bytes=0 bram-bytes=4608 uram-bytes=0" 2>&1 | FileCheck %s

// Banking four ways consumes four BRAM primitives even though the logical
// payload fits in two. The post-partition ledger must reject that physical
// overrun instead of trusting the pre-banking placement estimate.
func.func @banked_budget() attributes {top_func} {
  // CHECK: final banked memories exceed the resource budgets
  %buf = hls.dataflow.buffer {depth = 1 : i32}
      : memref<1024xf64, #hls.mem<bram_t2p>>
  %zero = arith.constant 0.0 : f64
  affine.for %i = 0 to 256 {
    affine.store %zero, %buf[%i * 4] : memref<1024xf64, #hls.mem<bram_t2p>>
    affine.store %zero, %buf[%i * 4 + 1] : memref<1024xf64, #hls.mem<bram_t2p>>
    affine.store %zero, %buf[%i * 4 + 2] : memref<1024xf64, #hls.mem<bram_t2p>>
    affine.store %zero, %buf[%i * 4 + 3] : memref<1024xf64, #hls.mem<bram_t2p>>
  }
  return
}
