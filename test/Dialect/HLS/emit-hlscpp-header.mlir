// RUN: sar-translate --hls-emit-hlscpp -emit-vitis-directives %s | FileCheck %s

// The header is what a board integrator reads first, so it has to name the
// protocol the ports actually speak. An AXI design reports its master count
// and the bus width those masters use; a stream design reports neither,
// because its ports carry `axis` pragmas rather than a memory-mapped bus.

// CHECK:      Top function : top
// CHECK:      Interface    : 2 AXI masters
// CHECK:      AXI bus      : {{[0-9]+}}-bit
// CHECK:      Sub-functions: 0
module {
  func.func @top(%arg0: !hls.axi<memref<64xf32, #hls.mem<dram>>>,
                 %arg1: !hls.axi<memref<64xf32, #hls.mem<dram>>>)
      attributes {top_func} {
    %b0 = hls.axi.bundle "axi_0" : <f32, mm>
    %p0 = hls.axi.port %b0, %arg0 : <f32, mm>,
        (!hls.axi<memref<64xf32, #hls.mem<dram>>>)
        -> memref<64xf32, #hls.mem<dram>>
    %b1 = hls.axi.bundle "axi_1" : <f32, mm>
    %p1 = hls.axi.port %b1, %arg1 : <f32, mm>,
        (!hls.axi<memref<64xf32, #hls.mem<dram>>>)
        -> memref<64xf32, #hls.mem<dram>>
    affine.for %i = 0 to 64 {
      %v = affine.load %p0[%i] : memref<64xf32, #hls.mem<dram>>
      affine.store %v, %p1[%i] : memref<64xf32, #hls.mem<dram>>
    }
    return
  }
}
