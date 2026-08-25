// RUN: sar-translate --hls-emit-hlscpp -emit-vitis-directives %s \
// RUN:   | FileCheck %s

// An m_axi argument remains one physical interface. A partition layout may
// describe internal consumers, but emitting it on the port would make Vitis
// split one declared master into several RTL masters.
// CHECK-LABEL: void axi_top(
// CHECK:       #pragma HLS interface m_axi
// CHECK-NEXT:  #pragma HLS interface s_axilite
// CHECK-NOT:   #pragma HLS array_partition variable=out0
module {
  func.func @axi_top(
      %arg0: !hls.axi<memref<8x64xf32,
          #hls.partition<[complete, cyclic], [8, 4]>,
          #hls.mem<bram_t2p>>>) attributes {top_func} {
    %bundle = hls.axi.bundle "data" : <f32, mm>
    %memref = hls.axi.port %bundle, %arg0
        : <f32, mm>,
          (!hls.axi<memref<8x64xf32,
              #hls.partition<[complete, cyclic], [8, 4]>,
              #hls.mem<bram_t2p>>>)
          -> memref<8x64xf32,
              #hls.partition<[complete, cyclic], [8, 4]>,
              #hls.mem<bram_t2p>>
    return
  }
}
