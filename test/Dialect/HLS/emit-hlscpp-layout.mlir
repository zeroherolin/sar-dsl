// RUN: sar-translate --hls-emit-hlscpp -emit-vitis-directives %s | FileCheck %s

// The emitted file has to read like something a human would hand to Vitis:
// a header saying where it came from, the top function up front, and every
// callee declared before it is used.

// CHECK:      SAR-DSL HLS emitter
// CHECK:      Top function : top
// CHECK:      Directives   : vitis
// CHECK:      Sub-functions: 2

// Only the headers the design needs. This one has no stream, no vector and
// no wide integer, so none of those may appear.
// CHECK:      #include <cmath>
// CHECK-NOT:  #include <hls_stream.h>
// CHECK-NOT:  #include <hls_vector.h>
// CHECK-NOT:  #include <ap_int.h>

// `using namespace std` in generated code lets an unqualified `abs` bind to
// the C integer overload and silently truncate a double.
// CHECK-NOT:  using namespace std;

// Prototypes come first, so the top function can be read before the parts.
// CHECK:      Sub-function prototypes
// CHECK:      void top_s00(
// CHECK:      );
// CHECK:      void top_s01(
// CHECK:      );

// The top function precedes both of its callees.
// CHECK:      Top function of the design.
// CHECK-NEXT: void top(
// CHECK:        float in0[8][8]
// CHECK:        float out0[8][8]
// CHECK:      #pragma HLS interface s_axilite port=return bundle=ctrl
// CHECK:        top_s00(
// CHECK:        top_s01(

// Definitions follow, in dataflow order.
// CHECK:      void top_s00(
// CHECK:      void top_s01(

module {
  func.func @top_node1(%arg0: memref<8x8xf32, #hls.mem<bram_t2p>>,
                       %arg1: memref<8x8xf32, #hls.mem<bram_t2p>>)
      attributes {inline} {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        %v = affine.load %arg0[%i, %j] : memref<8x8xf32, #hls.mem<bram_t2p>>
        %a = math.absf %v : f32
        affine.store %a, %arg1[%i, %j] : memref<8x8xf32, #hls.mem<bram_t2p>>
      }
    }
    return
  }

  func.func @top_node0(%arg0: memref<8x8xf32, #hls.mem<bram_t2p>>,
                       %arg1: memref<8x8xf32, #hls.mem<bram_t2p>>)
      attributes {inline} {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        %v = affine.load %arg0[%i, %j] : memref<8x8xf32, #hls.mem<bram_t2p>>
        %s = math.sqrt %v : f32
        affine.store %s, %arg1[%i, %j] : memref<8x8xf32, #hls.mem<bram_t2p>>
      }
    }
    return
  }

  func.func @top(%arg0: memref<8x8xf32, #hls.mem<bram_t2p>>,
                 %arg1: memref<8x8xf32, #hls.mem<bram_t2p>>)
      attributes {top_func} {
    %0 = hls.dataflow.buffer {depth = 1 : i32}
        : memref<8x8xf32, #hls.mem<bram_t2p>>
    call @top_node1(%arg0, %0) : (memref<8x8xf32, #hls.mem<bram_t2p>>,
                                  memref<8x8xf32, #hls.mem<bram_t2p>>) -> ()
    call @top_node0(%0, %arg1) : (memref<8x8xf32, #hls.mem<bram_t2p>>,
                                  memref<8x8xf32, #hls.mem<bram_t2p>>) -> ()
    return
  }
}
