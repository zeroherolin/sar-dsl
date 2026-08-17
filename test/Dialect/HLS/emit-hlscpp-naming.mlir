// RUN: sar-translate -hls-emit-hlscpp -emit-vitis-directives %s | FileCheck %s

// Ports and identifiers carry their role. Nothing in the emitted C++ should
// be named after a counter that ran across the whole design.

// A constant table belongs at file scope, where the tool can put it in ROM
// and a reader can skip over it. Its name says what the values are: this one
// is a constant-step ramp, i.e. a sampling axis.
// CHECK:      Constant tables
// CHECK:      static const double kAxis0_4[4] = {
// CHECK-NEXT:   0, 1, 2, 3
// CHECK-NEXT: };

// A table only ever read through a callee makes that parameter const.
// CHECK:      void named_s00(
// CHECK-NEXT:   const double in0[4]

// Loop counters read as counters; ports read as ports.
// CHECK:      void named(
// CHECK:        double in0[4][4]
// CHECK:        double out0[4][4]
// CHECK:        named_s00(kAxis0_4,

// CHECK:      void named_s00(
// CHECK:        for (int i0 = 0
// CHECK:          for (int i1 = 0

// No line-number comments pointing into an MLIR file the reader never sees.
// CHECK-NOT:  // L{{[0-9]}}

module {
  func.func @named_node0(%arg0: memref<4xf64, #hls.mem<bram_t2p>>,
                         %arg1: memref<4x4xf64, #hls.mem<bram_t2p>>,
                         %arg2: memref<4x4xf64, #hls.mem<bram_t2p>>)
      attributes {inline} {
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 {
        %t = affine.load %arg0[%j] : memref<4xf64, #hls.mem<bram_t2p>>
        %v = affine.load %arg1[%i, %j] : memref<4x4xf64, #hls.mem<bram_t2p>>
        %m = arith.mulf %v, %t : f64
        affine.store %m, %arg2[%i, %j] : memref<4x4xf64, #hls.mem<bram_t2p>>
      }
    }
    return
  }

  func.func @named(%arg0: memref<4x4xf64, #hls.mem<bram_t2p>>,
                   %arg1: memref<4x4xf64, #hls.mem<bram_t2p>>)
      attributes {top_func} {
    %0 = hls.dataflow.const_buffer {value = dense<[0.0, 1.0, 2.0, 3.0]>
        : tensor<4xf64>} : memref<4xf64, #hls.mem<bram_t2p>>
    call @named_node0(%0, %arg0, %arg1)
        : (memref<4xf64, #hls.mem<bram_t2p>>,
           memref<4x4xf64, #hls.mem<bram_t2p>>,
           memref<4x4xf64, #hls.mem<bram_t2p>>) -> ()
    return
  }
}
