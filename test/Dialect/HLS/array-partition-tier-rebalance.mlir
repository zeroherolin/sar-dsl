// RUN: sar-opt %s --hls-array-partition="bram-bytes=1000000 uram-bytes=36864" | FileCheck %s
// RUN: ! sar-opt %s --hls-array-partition="bram-bytes=8000 uram-bytes=36864" 2>&1 | FileCheck %s --check-prefix=TIGHT
// RUN: sar-opt %s --hls-array-partition="bram-bytes=1000000 uram-bytes=37748736" | FileCheck %s --check-prefix=ROOMY

// A memory primitive is claimed whole, and banking is decided after
// placement has already chosen a tier from the unbanked array. An UltraRAM
// (36864 bytes) holding an 8 KiB bank wastes most of itself, so when the
// UltraRAM tier is over budget such an array is re-bound to block RAM,
// whose 4608-byte granularity the bank does fill.
//
// Block RAM is the smaller budget, so the move is only made while it has
// room: under TIGHT it has none, the array stays where placement put it,
// and the design is refused rather than emitted over budget. Under ROOMY
// the UltraRAM tier is already inside its budget, so there is nothing to
// relieve and the binding placement chose stands.

// TIGHT: SAR_HLS_RETRYABLE_PARTITION_OVERFLOW{{.*}}URAM 294912/36864 bytes

// CHECK-LABEL: func.func @banked_uram
// ROOMY-LABEL: func.func @banked_uram
func.func @banked_uram() attributes {top_func} {
  // CHECK: hls.dataflow.buffer {{.*}} #hls.mem<bram_t2p>
  // ROOMY: hls.dataflow.buffer {{.*}} #hls.mem<uram_t2p>
  %buf = hls.dataflow.buffer {depth = 1 : i32}
      : memref<8192xf32, #hls.mem<uram_t2p>>
  %c0 = arith.constant 0.0 : f32
  affine.for %i = 0 to 2048 {
    affine.store %c0, %buf[%i * 4] : memref<8192xf32, #hls.mem<uram_t2p>>
    affine.store %c0, %buf[%i * 4 + 1] : memref<8192xf32, #hls.mem<uram_t2p>>
    affine.store %c0, %buf[%i * 4 + 2] : memref<8192xf32, #hls.mem<uram_t2p>>
    affine.store %c0, %buf[%i * 4 + 3] : memref<8192xf32, #hls.mem<uram_t2p>>
  }
  return
}
