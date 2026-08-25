// RUN: sar-opt %s --hls-widen-external-memory="bus-bits=512 min-elements=1" \
// RUN:   | FileCheck %s

module {
  func.func @rmw(%source: memref<16xf32, #hls.mem<dram>>,
                 %arena: memref<16xf32, #hls.mem<dram>>) {
    affine.for %word = 0 to 2 {
      %v0 = affine.load %source[%word * 8] : memref<16xf32, #hls.mem<dram>>
      affine.store %v0, %arena[%word * 8] : memref<16xf32, #hls.mem<dram>>
      %v1 = affine.load %source[%word * 8 + 1]
          : memref<16xf32, #hls.mem<dram>>
      affine.store %v1, %arena[%word * 8 + 1]
          : memref<16xf32, #hls.mem<dram>>
      %v2 = affine.load %source[%word * 8 + 2]
          : memref<16xf32, #hls.mem<dram>>
      affine.store %v2, %arena[%word * 8 + 2]
          : memref<16xf32, #hls.mem<dram>>
      %v3 = affine.load %source[%word * 8 + 3]
          : memref<16xf32, #hls.mem<dram>>
      affine.store %v3, %arena[%word * 8 + 3]
          : memref<16xf32, #hls.mem<dram>>
      %v4 = affine.load %source[%word * 8 + 4]
          : memref<16xf32, #hls.mem<dram>>
      affine.store %v4, %arena[%word * 8 + 4]
          : memref<16xf32, #hls.mem<dram>>
      %v5 = affine.load %source[%word * 8 + 5]
          : memref<16xf32, #hls.mem<dram>>
      affine.store %v5, %arena[%word * 8 + 5]
          : memref<16xf32, #hls.mem<dram>>
      %v6 = affine.load %source[%word * 8 + 6]
          : memref<16xf32, #hls.mem<dram>>
      affine.store %v6, %arena[%word * 8 + 6]
          : memref<16xf32, #hls.mem<dram>>
      %v7 = affine.load %source[%word * 8 + 7]
          : memref<16xf32, #hls.mem<dram>>
      affine.store %v7, %arena[%word * 8 + 7]
          : memref<16xf32, #hls.mem<dram>>
    }
    return
  }

  func.func @top(%source: !hls.axi<memref<16xf32, #hls.mem<dram>>>,
                 %arena: !hls.axi<memref<16xf32, #hls.mem<dram>>>)
      attributes {top_func} {
    %source_bundle = hls.axi.bundle "source" : <f32, mm>
    %source_memref = hls.axi.port %source_bundle, %source
        : <f32, mm>, (!hls.axi<memref<16xf32, #hls.mem<dram>>>)
        -> memref<16xf32, #hls.mem<dram>>
    %arena_bundle = hls.axi.bundle "arena" : <f32, mm>
    %arena_memref = hls.axi.port %arena_bundle, %arena {hls.scratch}
        : <f32, mm>, (!hls.axi<memref<16xf32, #hls.mem<dram>>>)
        -> memref<16xf32, #hls.mem<dram>>
    call @rmw(%source_memref, %arena_memref)
        : (memref<16xf32, #hls.mem<dram>>,
           memref<16xf32, #hls.mem<dram>>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @rmw(
// CHECK: %[[WORD:.*]] = affine.load {{.*}} : memref<2xvector<8xf32>
// CHECK: %[[I0:.*]] = vector.insert {{.*}}, %[[WORD]]
// CHECK: %[[I1:.*]] = vector.insert {{.*}}, %[[I0]]
// CHECK: vector.insert
// CHECK: vector.insert
// CHECK: vector.insert
// CHECK: vector.insert
// CHECK: vector.insert
// CHECK: %[[I7:.*]] = vector.insert
// CHECK: affine.store %[[I7]], {{.*}} : memref<2xvector<8xf32>
// CHECK-NOT: affine.store {{.*}} : memref<2xvector<8xf32>
