// RUN: sar-opt %s --hls-place-dataflow-buffer="threshold=4096 bram-bytes=6193152 uram-bytes=23592960 lutram-bytes=901120 lutram-max-bytes=64" | FileCheck %s
// RUN: sar-opt %s --hls-place-dataflow-buffer="threshold=4096 bram-bytes=9216 uram-bytes=73728 lutram-bytes=0 lutram-max-bytes=64" | FileCheck %s --check-prefix=TIGHT
// RUN: sar-opt %s --hls-place-dataflow-buffer="threshold=4096 bram-bytes=6193152 uram-bytes=23592960 lutram-bytes=901120 lutram-max-bytes=64 allow-dram=false" | FileCheck %s --check-prefix=NODRAM

// The budgets are hard caps charged in whole primitives, twice per
// dataflow buffer (Vitis double-buffers every channel); 0 forbids a
// tier. Tier per measured size: lutram at or below `lutram-max-bytes`,
// URAM at or above one physical URAM block (288 Kb), block RAM between,
// DRAM past `threshold` elements -- and either block tier overflows
// into the other before anything is left behind. A buffer that cannot
// stream and fits no tier fails the pass; see
// place-dataflow-buffer-invalid.mlir.

// CHECK-LABEL: func.func @tiers
// NODRAM-LABEL: func.func @tiers
func.func @tiers(%v: f32) {
  // 16 x f32 = 64 B: at the lutram-max boundary, distributed RAM.
  // CHECK: hls.dataflow.buffer {depth = 1 : i32} : memref<16xf32, #hls.mem<lutram_s2p>>
  %small = hls.dataflow.buffer {depth = 1 : i32} : memref<16xf32>
  // 1024 x f32 = 4 KiB: too big for a LUT bank, below a URAM block, so
  // block RAM.
  // CHECK: hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32, #hls.mem<bram_t2p>>
  %mid = hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32>
  // 96x96 x f64 = 9216 elements >= threshold 4096: streamed from DRAM.
  // CHECK: hls.dataflow.buffer {depth = 1 : i32} : memref<96x96xf64, #hls.mem<dram>>
  //
  // An `ap_memory` design has no external master to stream through, so the
  // same plane has to fit a block tier or the design is refused. The
  // threshold does not apply when there is nowhere to put what it moves.
  // NODRAM: hls.dataflow.buffer {depth = 1 : i32} : memref<96x96xf64, #hls.mem<uram_t2p>>
  // NODRAM-NOT: #hls.mem<dram>
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
// never stream, whatever their size (8192 x f64 is past the threshold),
// and 64 KiB is above one URAM block, so UltraRAM.

// CHECK-LABEL: func.func @const_stays_resident
// CHECK: hls.dataflow.const_buffer
// CHECK-SAME: memref<8192xf64, #hls.mem<uram_t2p>>
func.func @const_stays_resident() -> f64 {
  %table = hls.dataflow.const_buffer {value = dense<1.0> : tensor<8192xf64>}
      : memref<8192xf64>
  %c0 = arith.constant 0 : index
  %v = memref.load %table[%c0] : memref<8192xf64>
  return %v : f64
}

// A 4 KiB f32 buffer costs two 36 Kb blocks (9216 bytes) in block RAM;
// with exactly that block RAM the first buffer fills it, the second
// overflows into URAM (part of the block is wasted, but leaving block
// RAM over budget would waste the design), and with URAM spent too the
// third streams from DRAM.

// TIGHT-LABEL: func.func @budget_spills
// TIGHT: hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32, #hls.mem<bram_t2p>>
// TIGHT: hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32, #hls.mem<uram_t2p>>
// TIGHT: hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32, #hls.mem<dram>>
func.func @budget_spills(%v: f32) {
  %a = hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32>
  %b = hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32>
  %c = hls.dataflow.buffer {depth = 1 : i32} : memref<1024xf32>
  affine.for %i = 0 to 1024 {
    affine.store %v, %a[%i] : memref<1024xf32>
    affine.store %v, %b[%i] : memref<1024xf32>
    affine.store %v, %c[%i] : memref<1024xf32>
  }
  return
}

// A declaration has nothing to place and must pass through untouched.

// CHECK-LABEL: func.func private @extern_decl
// CHECK-SAME: (memref<64xf32>)
func.func private @extern_decl(memref<64xf32>)

// A buffer allocated inside a compiled loop lives for one iteration: the
// DRAM threshold must not apply (an AXI port cannot be carved through an
// scf.for region), so it places on chip like a constant buffer would.

// CHECK-LABEL: func.func @per_iteration_buffer
func.func @per_iteration_buffer(%v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    // Over the 4096-element DRAM threshold, but per-iteration: on chip.
    // CHECK: hls.dataflow.buffer {depth = 1 : i32} : memref<8192xf32, #hls.mem<bram_t2p>>
    %scratch = hls.dataflow.buffer {depth = 1 : i32} : memref<8192xf32>
    affine.for %j = 0 to 8192 {
      affine.store %v, %scratch[%j] : memref<8192xf32>
    }
  }
  return
}
