// RUN: sar-opt %s --hls-place-dataflow-buffer="threshold=4096 bram-bytes=9216 uram-bytes=0 lutram-bytes=0 lutram-max-bytes=64 rebalance-only=true" --verify-diagnostics

// The rebalance run recharges buffers the fork/balance passes copied.
// Nothing may newly stream this late, so an overflow is a hard failure
// -- and this diagnostic's wording is the HLS backend's retry trigger
// (`sar/backends/hls/compiler.py`); changing one means changing both.

// expected-error @below {{on-chip working set exceeds the memory budgets}}
func.func @rebalance_overflow(%v: f32) {
  %a = hls.dataflow.buffer {depth = 1 : i32}
      : memref<1024xf32, #hls.mem<bram_t2p>>
  %b = hls.dataflow.buffer {depth = 1 : i32}
      : memref<1024xf32, #hls.mem<bram_t2p>>
  affine.for %i = 0 to 1024 {
    affine.store %v, %a[%i] : memref<1024xf32, #hls.mem<bram_t2p>>
    affine.store %v, %b[%i] : memref<1024xf32, #hls.mem<bram_t2p>>
  }
  return
}
