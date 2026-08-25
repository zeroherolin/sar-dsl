// RUN: sar-opt %s "--hls-create-axi-interface=max-scratch-arenas=2" | FileCheck %s

// Three simultaneously written buffers require three colors. The two-master
// cap keeps the ABI bounded and records the predicted bandwidth penalty used
// to choose the least costly merge.

// CHECK-LABEL: func.func @overflow(
// CHECK-SAME: !hls.axi<memref<64xf32, #hls.mem<dram>>>
// CHECK-SAME: !hls.axi<memref<128xf32, #hls.mem<dram>>>
// CHECK-NOT: !hls.axi<memref<
// CHECK-SAME: hls.scratch_arena_overflow
// CHECK-SAME: hls.scratch_arena_penalty = {{[1-9][0-9]*}}
module {
  func.func @write3(%a: memref<64xf32, #hls.mem<dram>>,
                    %b: memref<64xf32, #hls.mem<dram>>,
                    %c: memref<64xf32, #hls.mem<dram>>) {
    affine.for %i = 0 to 64 {
      %zero = arith.constant 0.0 : f32
      affine.store %zero, %a[%i] : memref<64xf32, #hls.mem<dram>>
      affine.store %zero, %b[%i] : memref<64xf32, #hls.mem<dram>>
      affine.store %zero, %c[%i] : memref<64xf32, #hls.mem<dram>>
    }
    return
  }

  func.func @overflow() attributes {top_func} {
    %a = hls.dataflow.buffer {depth = 1 : i32}
        : memref<64xf32, #hls.mem<dram>>
    %b = hls.dataflow.buffer {depth = 1 : i32}
        : memref<64xf32, #hls.mem<dram>>
    %c = hls.dataflow.buffer {depth = 1 : i32}
        : memref<64xf32, #hls.mem<dram>>
    call @write3(%a, %b, %c)
        : (memref<64xf32, #hls.mem<dram>>,
           memref<64xf32, #hls.mem<dram>>,
           memref<64xf32, #hls.mem<dram>>) -> ()
    return
  }
}
