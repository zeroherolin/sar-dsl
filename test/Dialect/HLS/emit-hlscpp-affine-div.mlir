// RUN: sar-translate --hls-emit-hlscpp %s | FileCheck %s

#floor = affine_map<(d0) -> (d0 floordiv 4)>
#ceil = affine_map<(d0) -> (d0 ceildiv 4)>
#mod = affine_map<(d0) -> (d0 mod 4)>

// CHECK: static int64_t sar_hls_floor_div
// CHECK: return quotient - (remainder < 0)
// CHECK: static int64_t sar_hls_ceil_div
// CHECK: return quotient + (remainder > 0)
// CHECK: static int64_t sar_hls_mod
// CHECK: return remainder < 0 ? remainder + rhs : remainder

func.func @affine_div(%value: index, %out: memref<3xindex>)
    attributes {top_func} {
  // CHECK: sar_hls_floor_div(arg0, 4)
  %floor = affine.apply #floor(%value)
  // CHECK: sar_hls_ceil_div(arg0, 4)
  %ceil = affine.apply #ceil(%value)
  // CHECK: sar_hls_mod(arg0, 4)
  %mod = affine.apply #mod(%value)
  affine.store %floor, %out[0] : memref<3xindex>
  affine.store %ceil, %out[1] : memref<3xindex>
  affine.store %mod, %out[2] : memref<3xindex>
  return
}
