// RUN: sar-opt %s --hls-place-dataflow-buffer="threshold=4096 bram-bytes=0 uram-bytes=0 lutram-bytes=0 lutram-max-bytes=64 uram-min-bytes=36864" | FileCheck %s
// RUN: sar-opt %s --hls-place-dataflow-buffer="threshold=4096 bram-bytes=4608 uram-bytes=0 lutram-bytes=0 lutram-max-bytes=64 uram-min-bytes=36864" | FileCheck %s --check-prefix=TIGHT

// Tier per measured size: lutram at or below `lutram-max-bytes`, uram at
// or above `uram-min-bytes`, bram between, DRAM past `threshold` elements.

// CHECK-LABEL: func.func @tiers
func.func @tiers(%v: f32) {
  // 16 x f32 = 64 B: at the lutram-max boundary, distributed RAM.
  // CHECK: hls.dataflow.buffer {depth = 1 : i32} : memref<16xf32, #hls.mem<lutram_s2p>>
  %small = hls.dataflow.buffer {depth = 1 : i32} : memref<16xf32>
  // 1024 x f32 = 4 KiB: too big for a LUT bank, too small for a URAM
  // block, so block RAM.
  // CHECK: hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32, #hls.mem<bram_t2p>>
  %mid = hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32>
  // 96x96 x f64 = 9216 elements >= threshold 4096: streamed from DRAM.
  // CHECK: hls.dataflow.buffer {depth = 1 : i32} : memref<96x96xf64, #hls.mem<dram>>
  %plane = hls.dataflow.buffer {depth = 1 : i32} : memref<96x96xf64>
  affine.for %i = 0 to 16 {
    affine.store %v, %small[%i] : memref<16xf32>
    affine.store %v, %mid[%i] : memref<1024xf32>
    %w = arith.extf %v : f32 to f64
    affine.store %w, %plane[%i, %i] : memref<96x96xf64>
  }
  return
}

// Constant buffers are the ROM tables the design needs on entry; they
// never stream, whatever their size (8192 x f64 is past the threshold).

// CHECK-LABEL: func.func @const_stays_resident
// CHECK: hls.dataflow.const_buffer
// CHECK-SAME: memref<8192xf64, #hls.mem<uram_t2p>>
// TIGHT-LABEL: func.func @const_stays_resident
// TIGHT-NOT: #hls.mem<dram>
func.func @const_stays_resident() -> f64 {
  %table = hls.dataflow.const_buffer {value = dense<1.0> : tensor<8192xf64>}
      : memref<8192xf64>
  %c0 = arith.constant 0 : index
  %v = memref.load %table[%c0] : memref<8192xf64>
  return %v : f64
}

// Budgets are charged in whole primitives: with one 36 Kb block of BRAM,
// the first 4 KiB buffer fills it (rounded up to the block) and the second
// spills to DRAM.

// TIGHT-LABEL: func.func @budget_spills
// TIGHT: hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32, #hls.mem<bram_t2p>>
// TIGHT: hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32, #hls.mem<dram>>
func.func @budget_spills(%v: f32) {
  %a = hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32>
  %b = hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32>
  affine.for %i = 0 to 1024 {
    affine.store %v, %a[%i] : memref<1024xf32>
    affine.store %v, %b[%i] : memref<1024xf32>
  }
  return
}

// A declaration has nothing to place and must pass through untouched.

// CHECK-LABEL: func.func private @extern_decl
// CHECK-SAME: (memref<64xf32>)
func.func private @extern_decl(memref<64xf32>)
