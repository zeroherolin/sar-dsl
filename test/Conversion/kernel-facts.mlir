// RUN: sar-translate --sar-emit-kernel-facts %s | FileCheck %s

// CHECK: "plane_elements":64
// CHECK-SAME: "element_bytes":4
// CHECK-SAME: "transposes":4
// CHECK-SAME: "transforms":
// CHECK-SAME: 8,4
// CHECK-SAME: "transform_strided":[false]
// CHECK-SAME: "buffers":
func.func @facts(%x: tensor<8x8xcomplex<f32>>, %positions: tensor<8x8xf64>)
    -> tensor<8x8xcomplex<f32>> {
  %0 = sar.fft %x {dim = 1 : i64} : tensor<8x8xcomplex<f32>>
  %1 = sar.transpose %0
      : tensor<8x8xcomplex<f32>> -> tensor<8x8xcomplex<f32>>
  %2 = sar.interp1d %1, %positions {dim = 0 : i64}
      : (tensor<8x8xcomplex<f32>>, tensor<8x8xf64>)
      -> tensor<8x8xcomplex<f32>>
  return %2 : tensor<8x8xcomplex<f32>>
}
