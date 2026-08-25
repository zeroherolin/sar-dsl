// RUN: sar-translate --hls-emit-hlscpp %s | FileCheck %s

func.func @state_and_unsigned(%lhs: memref<1xi8>, %rhs: memref<1xi8>,
                              %out: memref<8xi8>) attributes {top_func} {
  %buffer = hls.dataflow.buffer {
      depth = 1 : i32, init_value = -1 : i8} : memref<2xi8>
  // CHECK: ap_int<8> buf0[2];
  // CHECK: for (int64_t init_i0 = 0; init_i0 < 2; ++init_i0) {
  // CHECK-NEXT: buf0[init_i0] = (ap_int<8>)-1;

  %stream = hls.dataflow.stream {depth = 3 : i32} : <i8, 3>
  // CHECK: hls::stream<ap_int<8>> {{.*}};
  // CHECK-NEXT: #pragma HLS stream variable={{.*}} depth=3

  %c0 = arith.constant 0 : index
  %a = memref.load %lhs[%c0] : memref<1xi8>
  %b = memref.load %rhs[%c0] : memref<1xi8>
  // CHECK: (ap_uint<8>)({{.*}}) < (ap_uint<8>)({{.*}})
  %cmp = arith.cmpi ult, %a, %b : i8
  // CHECK: (ap_uint<8>)({{.*}}) / (ap_uint<8>)({{.*}})
  %div = arith.divui %a, %b : i8
  // CHECK: (ap_uint<8>)({{.*}}) % (ap_uint<8>)({{.*}})
  %rem = arith.remui %a, %b : i8
  // CHECK: (ap_uint<8>)({{.*}}) >> (ap_uint<8>)({{.*}})
  %shr = arith.shrui %a, %b : i8
  // CHECK: std::max((ap_uint<8>)({{.*}}), (ap_uint<8>)({{.*}}))
  %max = arith.maxui %a, %b : i8
  // CHECK: std::min((ap_uint<8>)({{.*}}), (ap_uint<8>)({{.*}}))
  %min = arith.minui %a, %b : i8
  // CHECK: = (ap_uint<8>)({{.*}});
  %as_float = arith.uitofp %a : i8 to f32
  // CHECK: = (ap_uint<8>)({{.*}});
  %as_uint = arith.fptoui %as_float : f32 to i8
  %cmp_i8 = arith.extui %cmp : i1 to i8
  affine.store %cmp_i8, %out[0] : memref<8xi8>
  affine.store %div, %out[1] : memref<8xi8>
  affine.store %rem, %out[2] : memref<8xi8>
  affine.store %shr, %out[3] : memref<8xi8>
  affine.store %max, %out[4] : memref<8xi8>
  affine.store %min, %out[5] : memref<8xi8>
  affine.store %as_uint, %out[6] : memref<8xi8>
  %initial = affine.load %buffer[0] : memref<2xi8>
  affine.store %initial, %out[7] : memref<8xi8>
  return
}

func.func private @vector_constant() {
  // CHECK: hls::vector<float, 4> {{.*}} = {(float)0, (float)0, (float)0, (float)0};
  %zero = arith.constant dense<0.0> : vector<4xf32>
  return
}
