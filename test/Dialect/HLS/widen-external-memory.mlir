// RUN: sar-opt %s --hls-widen-external-memory="bus-bits=512 min-elements=1" | FileCheck %s

module {
  // CHECK-LABEL: func.func @pack_node(
  // CHECK-SAME: memref<2x2xvector<8xf32>, #hls.mem<dram>>
  // CHECK-SAME: memref<2x16xf32, #hls.mem<dram>>
  // CHECK: affine.for
  // CHECK: %[[WORD:.*]] = affine.load
  // CHECK-SAME: memref<2x2xvector<8xf32>
  // CHECK: vector.extract %[[WORD]][0] : f32 from vector<8xf32>
  // CHECK: affine.store {{.*}} : memref<2x16xf32
  func.func @pack_node(%input: memref<2x16xf32, #hls.mem<dram>>,
                       %output: memref<2x16xf32, #hls.mem<dram>>) {
    affine.for %row = 0 to 2 {
      affine.for %col = 0 to 16 {
        %value = affine.load %input[%row, %col]
            : memref<2x16xf32, #hls.mem<dram>>
        %two = arith.constant 2.0 : f32
        %scaled = arith.mulf %value, %two : f32
        affine.store %scaled, %output[%row, %col]
            : memref<2x16xf32, #hls.mem<dram>>
      } {loop_directive = #hls.loop<pipeline = true, target_ii = 1>}
    }
    return
  }

  // CHECK-LABEL: func.func @pack(
  // CHECK-SAME: !hls.axi<memref<2x2xvector<8xf32>, #hls.mem<dram>>>
  // CHECK-SAME: !hls.axi<memref<2x16xf32, #hls.mem<dram>>>
  func.func @pack(
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
    call @pack_node(%input_memref, %output_memref)
        : (memref<2x16xf32, #hls.mem<dram>>,
           memref<2x16xf32, #hls.mem<dram>>) -> ()
    return
  }

  // CHECK-LABEL: func.func @main(
  // CHECK-SAME: memref<2x2xvector<8xf32>, #hls.mem<dram>>
  // CHECK-SAME: memref<2x16xf32, #hls.mem<dram>>
  func.func @main(%input: memref<2x16xf32, #hls.mem<dram>>,
                  %output: memref<2x16xf32, #hls.mem<dram>>)
      attributes {runtime} {
    %input_axi = hls.axi.pack %input
        : (memref<2x16xf32, #hls.mem<dram>>)
        -> !hls.axi<memref<2x16xf32, #hls.mem<dram>>>
    %output_axi = hls.axi.pack %output
        : (memref<2x16xf32, #hls.mem<dram>>)
        -> !hls.axi<memref<2x16xf32, #hls.mem<dram>>>
    call @pack(%input_axi, %output_axi)
        : (!hls.axi<memref<2x16xf32, #hls.mem<dram>>>,
           !hls.axi<memref<2x16xf32, #hls.mem<dram>>>) -> ()
    return
  }
}
