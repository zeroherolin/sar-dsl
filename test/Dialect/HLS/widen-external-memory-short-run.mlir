// RUN: sar-opt %s --hls-widen-external-memory="bus-bits=256 min-elements=1" \
// RUN:   | FileCheck %s

// A two-element sweep is not a complete eight-lane word. It must remain
// scalar instead of being selected as a packed loop that the rewrite cannot
// legally compact.
// CHECK-LABEL: func.func @short_node(
// CHECK-SAME: memref<2x8xf32, #hls.mem<dram>>
// CHECK-NOT: vector<8xf32>
func.func @short_node(%input: memref<2x8xf32, #hls.mem<dram>>) {
  affine.for %row = 0 to 2 {
    affine.for %col = 0 to 2 {
      %value = affine.load %input[%row, %col]
          : memref<2x8xf32, #hls.mem<dram>>
    }
  }
  return
}

func.func @short(%input: !hls.axi<memref<2x8xf32, #hls.mem<dram>>>)
    attributes {top_func} {
  %bundle = hls.axi.bundle "short" : <f32, mm>
  %memref = hls.axi.port %bundle, %input
      : <f32, mm>, (!hls.axi<memref<2x8xf32, #hls.mem<dram>>>)
      -> memref<2x8xf32, #hls.mem<dram>>
  call @short_node(%memref) : (memref<2x8xf32, #hls.mem<dram>>) -> ()
  return
}
