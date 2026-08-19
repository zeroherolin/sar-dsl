// RUN: sar-opt %s --split-input-file --hls-array-partition="lutram-max-bits=512 lutram-bytes=256" | FileCheck %s
// RUN: sar-opt %s --split-input-file --hls-array-partition="lutram-max-bits=512 lutram-bytes=0" | FileCheck %s --check-prefix=DISABLED

// The banking tier threshold is inclusive: a bank holding exactly one bus
// beat (here 8 x f64 = 512 bits after cyclic-4 partitioning) is the
// canonical distributed-RAM case. A strict compare would burn one block
// RAM primitive per bank on it.

// CHECK-LABEL: func.func @bank_at_threshold
// DISABLED-LABEL: func.func @bank_at_threshold
func.func @bank_at_threshold() attributes {top_func} {
  // CHECK: memref.alloc() : memref<32xf64, #hls.partition<[cyclic], [4]>, #hls.mem<lutram_2p>>
  // DISABLED: memref.alloc() : memref<32xf64, #hls.partition<[cyclic], [4]>>
  // DISABLED-NOT: #hls.mem<lutram
  %buf = memref.alloc() : memref<32xf64>
  %c0 = arith.constant 0.0 : f64
  affine.for %i = 0 to 8 {
    affine.store %c0, %buf[%i * 4] : memref<32xf64>
    affine.store %c0, %buf[%i * 4 + 1] : memref<32xf64>
    affine.store %c0, %buf[%i * 4 + 2] : memref<32xf64>
    affine.store %c0, %buf[%i * 4 + 3] : memref<32xf64>
  }
  return
}

// -----

// One element past the threshold (16 x f64 = 1024-bit banks) keeps the
// tier placement chose; only the partition layout is applied.

// CHECK-LABEL: func.func @bank_past_threshold
func.func @bank_past_threshold() attributes {top_func} {
  // CHECK: memref.alloc() : memref<64xf64, #hls.partition<[cyclic], [4]>>
  // CHECK-NOT: lutram
  %buf = memref.alloc() : memref<64xf64>
  %c0 = arith.constant 0.0 : f64
  affine.for %i = 0 to 16 {
    affine.store %c0, %buf[%i * 4] : memref<64xf64>
    affine.store %c0, %buf[%i * 4 + 1] : memref<64xf64>
    affine.store %c0, %buf[%i * 4 + 2] : memref<64xf64>
    affine.store %c0, %buf[%i * 4 + 3] : memref<64xf64>
  }
  return
}
