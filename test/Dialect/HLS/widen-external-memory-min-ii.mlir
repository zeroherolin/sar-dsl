// RUN: sar-opt %s --hls-widen-external-memory="bus-bits=512 min-elements=1" | FileCheck %s

// An initiation-interval floor records a structural hazard -- a gather whose
// taps contend for one banked window, say. Packing puts eight original
// iterations in one body, so eight of those hazards land in one interval;
// dual-ported banks retire two per cycle, leaving a floor of `8 * 8 / 2`.
// Carrying the original floor across the rewrite would ask Vitis for an
// interval no binding can reach, and the modulo scheduler spends a long time
// proving that before settling anyway.

module {
  // CHECK-LABEL: func.func @scaled_node
  // CHECK: affine.for %{{.*}} = 0 to 2 {
  // CHECK: } {hls.min_ii = 32 : i64
  // CHECK-SAME: target_ii = 32
  func.func @scaled_node(%input: memref<2x16xf32, #hls.mem<dram>>,
                         %output: memref<2x16xf32, #hls.mem<dram>>) {
    affine.for %row = 0 to 2 {
      affine.for %col = 0 to 16 {
        %value = affine.load %input[%row, %col]
            : memref<2x16xf32, #hls.mem<dram>>
        %two = arith.constant 2.0 : f32
        %scaled = arith.mulf %value, %two : f32
        affine.store %scaled, %output[%row, %col]
            : memref<2x16xf32, #hls.mem<dram>>
      } {hls.min_ii = 8 : i64,
          loop_directive = #hls.loop<pipeline = true, target_ii = 8>}
    }
    return
  }

  func.func @scaled(
      %input: !hls.axi<memref<2x16xf32, #hls.mem<dram>>>,
      %output: !hls.axi<memref<2x16xf32, #hls.mem<dram>>>)
      attributes {top_func} {
    %input_bundle = hls.axi.bundle "input" : <f32, mm>
    %input_memref = hls.axi.port %input_bundle, %input
        : <f32, mm>, (!hls.axi<memref<2x16xf32, #hls.mem<dram>>>)
        -> memref<2x16xf32, #hls.mem<dram>>
    %output_bundle = hls.axi.bundle "output" : <f32, mm>
    %output_memref = hls.axi.port %output_bundle, %output
        : <f32, mm>, (!hls.axi<memref<2x16xf32, #hls.mem<dram>>>)
        -> memref<2x16xf32, #hls.mem<dram>>
    call @scaled_node(%input_memref, %output_memref)
        : (memref<2x16xf32, #hls.mem<dram>>,
           memref<2x16xf32, #hls.mem<dram>>) -> ()
    return
  }
}
