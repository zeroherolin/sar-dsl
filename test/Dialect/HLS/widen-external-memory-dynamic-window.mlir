// RUN: sar-opt %s --hls-widen-external-memory="bus-bits=256 max-lanes=4 min-elements=1" \
// RUN:   | FileCheck %s

// Dynamic clamped indices model a data-dependent SVA/tap window. Unrelated
// loads from a second plane are interleaved, but each plane's taps still share
// two adjacent packed words.
// CHECK-LABEL: func.func @dynamic_window_node(
// CHECK-SAME: memref<4xvector<4xf32>, #hls.mem<dram>>
// CHECK-COUNT-4: memref.load %{{.*}} : memref<4xvector<4xf32>, #hls.mem<dram>>
// CHECK: vector.extract
func.func @dynamic_window_node(
    %a: memref<16xf32, #hls.mem<dram>>,
    %b: memref<16xf32, #hls.mem<dram>>,
    %i: i64) {
  %c0 = arith.constant 0 : i64
  %c7 = arith.constant 7 : i64
  %c1 = arith.constant 1 : i64
  %x0 = arith.addi %i, %c0 : i64
  %x1 = arith.addi %i, %c1 : i64
  %y0 = arith.maxsi %x0, %c0 : i64
  %y1 = arith.maxsi %x1, %c0 : i64
  %z0 = arith.minsi %y0, %c7 : i64
  %z1 = arith.minsi %y1, %c7 : i64
  %a0 = arith.index_cast %z0 : i64 to index
  %a1 = arith.index_cast %z1 : i64 to index
  %va = memref.load %a[%a0] : memref<16xf32, #hls.mem<dram>>
  %vb = memref.load %b[%a0] : memref<16xf32, #hls.mem<dram>>
  %wa = memref.load %a[%a1] : memref<16xf32, #hls.mem<dram>>
  %wb = memref.load %b[%a1] : memref<16xf32, #hls.mem<dram>>
  %sum0 = arith.addf %va, %vb : f32
  %sum1 = arith.addf %wa, %wb : f32
  return
}

func.func @dynamic_window(
    %a: !hls.axi<memref<16xf32, #hls.mem<dram>>>,
    %b: !hls.axi<memref<16xf32, #hls.mem<dram>>>, %i: i64) {
  %ba = hls.axi.bundle "a" : <f32, mm>
  %ma = hls.axi.port %ba, %a {hls.scratch}
      : <f32, mm>, (!hls.axi<memref<16xf32, #hls.mem<dram>>>)
      -> memref<16xf32, #hls.mem<dram>>
  %bb = hls.axi.bundle "b" : <f32, mm>
  %mb = hls.axi.port %bb, %b {hls.scratch}
      : <f32, mm>, (!hls.axi<memref<16xf32, #hls.mem<dram>>>)
      -> memref<16xf32, #hls.mem<dram>>
  call @dynamic_window_node(%ma, %mb, %i)
      : (memref<16xf32, #hls.mem<dram>>, memref<16xf32, #hls.mem<dram>>, i64)
      -> ()
  return
}
