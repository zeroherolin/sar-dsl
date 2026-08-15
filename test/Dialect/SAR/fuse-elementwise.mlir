// RUN: sar-opt %s --sar-fuse-elementwise="min-elements=32" | FileCheck %s
// RUN: sar-opt %s --sar-fuse-elementwise="min-elements=128" \
// RUN:   | FileCheck %s --check-prefix=BELOW

// Upstream's elementwise fusion declines to fuse a producer with several
// consumers, since each would recompute it. At scene scale the alternative
// materialises a whole raster, so above `min-elements` the producer is
// fused into every consumer however many there are.

// The producer (add) feeds two consumers (mul, sub). With the threshold at
// or below the 64-element width, it is recomputed into each: two generics
// remain, each carrying the add inline, and the shared producer is gone.

// CHECK-LABEL: func.func @two_consumers
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: arith.subf

// Below the threshold upstream's judgement stands: the producer is
// materialised once and read twice, so three generics remain.

// BELOW-LABEL: func.func @two_consumers
// BELOW: linalg.generic
// BELOW: linalg.generic
// BELOW: linalg.generic
#map = affine_map<(d0) -> (d0)>
func.func @two_consumers(%a: tensor<64xf64>, %b: tensor<64xf64>)
    -> (tensor<64xf64>, tensor<64xf64>) {
  %r = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel"]}
      ins(%a : tensor<64xf64>) outs(%b : tensor<64xf64>) {
  ^bb0(%in: f64, %out: f64):
    %v = arith.addf %in, %in : f64
    linalg.yield %v : f64
  } -> tensor<64xf64>
  %c1 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel"]}
      ins(%r : tensor<64xf64>) outs(%b : tensor<64xf64>) {
  ^bb0(%in: f64, %out: f64):
    %v = arith.mulf %in, %in : f64
    linalg.yield %v : f64
  } -> tensor<64xf64>
  %c2 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel"]}
      ins(%r : tensor<64xf64>) outs(%b : tensor<64xf64>) {
  ^bb0(%in: f64, %out: f64):
    %v = arith.subf %in, %in : f64
    linalg.yield %v : f64
  } -> tensor<64xf64>
  return %c1, %c2 : tensor<64xf64>, tensor<64xf64>
}
