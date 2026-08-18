// RUN: sar-opt %s --hls-place-dataflow-buffer="threshold=99999 bram-bytes=50000 uram-bytes=0 lutram-bytes=0" | FileCheck %s

// The 4 KiB payload would fit one BRAM, but cyclic-32 banking consumes at
// least 32 primitives. Placement must charge the bank fragmentation and
// spill instead of emitting a design that exceeds its physical budget.

// CHECK: #map = affine_map<(d0) -> (d0 mod 32, d0 floordiv 32)>
// CHECK-LABEL: func.func @banked_cost
// CHECK: memref<1024xf32, #map, #hls.mem<dram>>
func.func @banked_cost(%value: f32) {
  %buf = hls.dataflow.buffer {depth = 1 : i32}
      : memref<1024xf32, #hls.partition<[cyclic], [32]>>
  affine.store %value, %buf[0]
      : memref<1024xf32, #hls.partition<[cyclic], [32]>>
  return
}
