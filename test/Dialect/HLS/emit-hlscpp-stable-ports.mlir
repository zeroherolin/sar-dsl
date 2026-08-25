// RUN: sar-translate --hls-emit-hlscpp -emit-vitis-directives %s | FileCheck %s

// Inside a dataflow region Vitis checks every array as a channel: one
// writer, one reader. A read-only port is not a channel -- its value is a
// run-long constant -- and several nodes reading it is the normal fan-out
// of an input raster (HLS 200-779 otherwise). The emitter states that
// contract with `#pragma HLS stable`. Written ports carry no such pragma:
// for them the check is real.

// CHECK:      void fanout(
// CHECK:        #pragma HLS interface s_axilite port=return bundle=ctrl
// CHECK-NEXT:   #pragma HLS interface bram port=in0
// CHECK-NEXT:   #pragma HLS stable variable=in0
// CHECK-NEXT:   #pragma HLS interface m_axi offset=slave{{.*}} port=in1
// CHECK-NEXT:   #pragma HLS interface s_axilite port=in1 bundle=ctrl
// CHECK-NEXT:   #pragma HLS stable variable=in1
// CHECK-NEXT:   #pragma HLS interface bram port=out0
// CHECK-NOT:    #pragma HLS stable variable=out0
// CHECK-NEXT:   #pragma HLS interface bram port=out1
// CHECK-NOT:    #pragma HLS stable variable=out1
// CHECK:        #pragma HLS dataflow
func.func @fanout(%f: memref<4x4xf64>,
                  %g: memref<4x4xf64, #hls.mem<dram>>,
                  %a: memref<4x4xf64>,
                  %b: memref<4x4xf64>)
    attributes {func_directive = #hls.func<pipeline = false,
                                           target_interval = 1,
                                           dataflow = true>,
                top_func} {
  call @node_a(%f, %g, %a) : (memref<4x4xf64>,
                              memref<4x4xf64, #hls.mem<dram>>,
                              memref<4x4xf64>) -> ()
  call @node_b(%f, %g, %b) : (memref<4x4xf64>,
                              memref<4x4xf64, #hls.mem<dram>>,
                              memref<4x4xf64>) -> ()
  return
}

func.func @node_a(%f: memref<4x4xf64>,
                  %g: memref<4x4xf64, #hls.mem<dram>>,
                  %out: memref<4x4xf64>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %x = affine.load %f[%i, %j] : memref<4x4xf64>
      %y = affine.load %g[%i, %j] : memref<4x4xf64, #hls.mem<dram>>
      %s = arith.addf %x, %y : f64
      affine.store %s, %out[%i, %j] : memref<4x4xf64>
    }
  }
  return
}

func.func @node_b(%f: memref<4x4xf64>,
                  %g: memref<4x4xf64, #hls.mem<dram>>,
                  %out: memref<4x4xf64>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %x = affine.load %f[%i, %j] : memref<4x4xf64>
      %y = affine.load %g[%i, %j] : memref<4x4xf64, #hls.mem<dram>>
      %m = arith.mulf %x, %y : f64
      affine.store %m, %out[%i, %j] : memref<4x4xf64>
    }
  }
  return
}
