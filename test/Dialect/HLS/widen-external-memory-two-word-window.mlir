// RUN: sar-opt %s --hls-widen-external-memory="bus-bits=256 max-lanes=4 min-elements=1" \
// RUN:   | FileCheck %s

// Three taps at base, base+2 and base+4 cross a packed-word boundary for
// some base values. The packed representation must issue two vector loads,
// then select/extract lanes, rather than one vector load per tap.
// CHECK-LABEL: func.func @window_node(
// CHECK-SAME: memref<4xvector<4xf32>, #hls.mem<dram>>
// CHECK-COUNT-2: affine.load %{{.*}} : memref<4xvector<4xf32>, #hls.mem<dram>>
// CHECK-COUNT-3: vector.extract
func.func @window_node(%arena: memref<16xf32, #hls.mem<dram>>) {
  affine.for %i = 0 to 4 {
    %a = affine.load %arena[%i * 2] : memref<16xf32, #hls.mem<dram>>
    %b = affine.load %arena[%i * 2 + 2] : memref<16xf32, #hls.mem<dram>>
    %c = affine.load %arena[%i * 2 + 4] : memref<16xf32, #hls.mem<dram>>
    %sum = arith.addf %a, %b : f32
    %out = arith.addf %sum, %c : f32
  }
  return
}

func.func @window(
    %arena: !hls.axi<memref<16xf32, #hls.mem<dram>>>) {
  %bundle = hls.axi.bundle "window" : <f32, mm>
  %memref = hls.axi.port %bundle, %arena {hls.scratch}
      : <f32, mm>, (!hls.axi<memref<16xf32, #hls.mem<dram>>>)
      -> memref<16xf32, #hls.mem<dram>>
  call @window_node(%memref) : (memref<16xf32, #hls.mem<dram>>) -> ()
  return
}
