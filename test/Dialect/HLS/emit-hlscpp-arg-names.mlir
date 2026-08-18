// RUN: sar-translate --hls-emit-hlscpp -emit-vitis-directives %s | FileCheck %s

// `sar.arg_names` carries the kernel's Python parameter names down to the
// emitted top signature. Arguments past the list keep the role-based
// scheme, and a name the emitter cannot use verbatim -- a C++ keyword, or
// the shape of a name it generates itself -- falls back.
//
// Functions emit in name order, so the CHECK blocks below follow that
// order rather than the order the functions appear in.

// A split name colliding with a sibling parameter would redeclare a port,
// so the whole attribute is ignored rather than emitting broken C++.
// CHECK:      void duplicate_names(
// CHECK-NEXT:   double in0[4],
// CHECK-NEXT:   double in1[4],
// CHECK-NEXT:   double out0[4]
func.func @duplicate_names(%a: memref<4xf64, #hls.mem<dram>>,
                           %b: memref<4xf64, #hls.mem<dram>>,
                           %out: memref<4xf64, #hls.mem<dram>>)
    attributes {sar.arg_names = ["raw_re", "raw_re"]} {
  affine.for %i = 0 to 4 {
    %x = affine.load %a[%i] : memref<4xf64, #hls.mem<dram>>
    %y = affine.load %b[%i] : memref<4xf64, #hls.mem<dram>>
    %s = arith.subf %x, %y : f64
    affine.store %s, %out[%i] : memref<4xf64, #hls.mem<dram>>
  }
  return
}

// CHECK:      void named_ports(
// CHECK-NEXT:   double raw_re[4][4],
// CHECK-NEXT:   double raw_im[4][4],
// CHECK-NEXT:   double out0[4][4]
func.func @named_ports(%re: memref<4x4xf64, #hls.mem<dram>>,
                       %im: memref<4x4xf64, #hls.mem<dram>>,
                       %out: memref<4x4xf64, #hls.mem<dram>>)
    attributes {sar.arg_names = ["raw_re", "raw_im"]} {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %a = affine.load %re[%i, %j] : memref<4x4xf64, #hls.mem<dram>>
      %b = affine.load %im[%i, %j] : memref<4x4xf64, #hls.mem<dram>>
      %s = arith.addf %a, %b : f64
      affine.store %s, %out[%i, %j] : memref<4x4xf64, #hls.mem<dram>>
    }
  }
  return
}

// CHECK:      void reserved_names(
// CHECK-NEXT:   double in0[4],
// CHECK-NEXT:   double in1[4],
// CHECK-NEXT:   double out0[4]
func.func @reserved_names(%a: memref<4xf64, #hls.mem<dram>>,
                          %b: memref<4xf64, #hls.mem<dram>>,
                          %out: memref<4xf64, #hls.mem<dram>>)
    attributes {sar.arg_names = ["double", "buf7"]} {
  affine.for %i = 0 to 4 {
    %x = affine.load %a[%i] : memref<4xf64, #hls.mem<dram>>
    %y = affine.load %b[%i] : memref<4xf64, #hls.mem<dram>>
    %s = arith.mulf %x, %y : f64
    affine.store %s, %out[%i] : memref<4xf64, #hls.mem<dram>>
  }
  return
}
