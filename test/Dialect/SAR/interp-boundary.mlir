// RUN: sar-opt %s | FileCheck %s

// CHECK-LABEL: func.func @interp_default_boundary
func.func @interp_default_boundary(%data: tensor<8x16xcomplex<f64>>, %pos: tensor<8x16xf64>)
    -> tensor<8x16xcomplex<f64>> {
  // The default boundary is "zero" but it won't print when it's the default value.
  // CHECK: sar.interp1d
  %0 = sar.interp1d %data, %pos
      : (tensor<8x16xcomplex<f64>>, tensor<8x16xf64>)
      -> (tensor<8x16xcomplex<f64>>)
  return %0 : tensor<8x16xcomplex<f64>>
}

// CHECK-LABEL: func.func @interp_edge_boundary
func.func @interp_edge_boundary(%data: tensor<8x16xcomplex<f64>>, %pos: tensor<8x16xf64>)
    -> tensor<8x16xcomplex<f64>> {
  // CHECK: sar.interp1d %{{.*}}, %{{.*}} {boundary = "edge"}
  %0 = sar.interp1d %data, %pos {boundary = "edge"}
      : (tensor<8x16xcomplex<f64>>, tensor<8x16xf64>)
      -> (tensor<8x16xcomplex<f64>>)
  return %0 : tensor<8x16xcomplex<f64>>
}

// CHECK-LABEL: func.func @interp_reflect_boundary
func.func @interp_reflect_boundary(%data: tensor<8x16xcomplex<f64>>, %pos: tensor<8x16xf64>)
    -> tensor<8x16xcomplex<f64>> {
  // CHECK: sar.interp1d %{{.*}}, %{{.*}} {boundary = "reflect"}
  %0 = sar.interp1d %data, %pos {boundary = "reflect"}
      : (tensor<8x16xcomplex<f64>>, tensor<8x16xf64>)
      -> (tensor<8x16xcomplex<f64>>)
  return %0 : tensor<8x16xcomplex<f64>>
}

// CHECK-LABEL: func.func @interp_split_boundary
func.func @interp_split_boundary(%re: tensor<4x8xf64>, %im: tensor<4x8xf64>, %pos: tensor<4x8xf64>)
    -> (tensor<4x8xf64>, tensor<4x8xf64>) {
  // CHECK: sar.interp1d_split %{{.*}}, %{{.*}}, %{{.*}} {boundary = "edge"}
  %0, %1 = sar.interp1d_split %re, %im, %pos {boundary = "edge"}
      : (tensor<4x8xf64>, tensor<4x8xf64>, tensor<4x8xf64>)
      -> (tensor<4x8xf64>, tensor<4x8xf64>)
  return %0, %1 : tensor<4x8xf64>, tensor<4x8xf64>
}
