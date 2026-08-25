// RUN: sar-opt %s --hls-widen-external-memory="bus-bits=512 min-elements=1" | FileCheck %s
// RUN: sar-opt %s --hls-widen-external-memory="bus-bits=512 min-elements=1 \
// RUN:   pack-public-outputs=false" | FileCheck %s --check-prefix=SCALAROUT
// RUN: sar-opt %s --hls-widen-external-memory="bus-bits=512 min-elements=1 \
// RUN:   compute-max-lanes=2" | FileCheck %s --check-prefix=LANES2

module {
  // CHECK-LABEL: func.func @pack_node(
  // CHECK-SAME: memref<2x2xvector<8xf32>, #hls.mem<dram>>
  // CHECK-SAME: memref<2x2xvector<8xf32>, #hls.mem<dram>>
  // CHECK: affine.for
  // CHECK: %[[WORD:.*]] = affine.load
  // CHECK-SAME: memref<2x2xvector<8xf32>
  // CHECK: vector.extract %[[WORD]][0] : f32 from vector<8xf32>
  // CHECK: %[[PACKED:.*]] = vector.from_elements
  // CHECK: affine.store %[[PACKED]]
  // CHECK-SAME: memref<2x2xvector<8xf32>
  func.func @pack_node(%input: memref<2x16xf32, #hls.mem<dram>>,
                       %output: memref<2x16xf32, #hls.mem<dram>>) {
    affine.for %row = 0 to 2 {
      affine.for %col = 0 to 16 {
        %value = affine.load %input[%row, %col]
            : memref<2x16xf32, #hls.mem<dram>>
        %two = arith.constant 2.0 : f32
        %scaled = arith.mulf %value, %two : f32
        affine.store %scaled, %output[%row, %col]
            : memref<2x16xf32, #hls.mem<dram>>
      } {loop_directive = #hls.loop<pipeline = true, target_ii = 1>}
    }
    return
  }

  // CHECK-LABEL: func.func @pack(
  // CHECK-SAME: !hls.axi<memref<2x2xvector<8xf32>, #hls.mem<dram>>>
  // CHECK-SAME: !hls.axi<memref<2x2xvector<8xf32>, #hls.mem<dram>>>

  // A public output ABI can be held scalar while the read-only input still
  // takes the packed word: a caller that has to keep its result layout is
  // not a reason to give up the input's burst width.
  // SCALAROUT-LABEL: func.func @pack(
  // SCALAROUT-SAME: !hls.axi<memref<2x2xvector<8xf32>, #hls.mem<dram>>>
  // SCALAROUT-SAME: !hls.axi<memref<2x16xf32, #hls.mem<dram>>>
  func.func @pack(
      %input: !hls.axi<memref<2x16xf32, #hls.mem<dram>>>,
      %output: !hls.axi<memref<2x16xf32, #hls.mem<dram>>>)
      attributes {top_func} {
    %input_bundle = hls.axi.bundle "input" : <f32, mm>
    %input_memref = hls.axi.port %input_bundle, %input
        : <f32, mm>, (!hls.axi<memref<2x16xf32, #hls.mem<dram>>>)
        -> memref<2x16xf32, #hls.mem<dram>>
    %output_bundle = hls.axi.bundle "output" : <f32, mm>
    %output_memref = hls.axi.port %output_bundle, %output
        : <f32, mm>, (!hls.axi<memref<2x16xf32, #hls.mem<dram>>>)
        -> memref<2x16xf32, #hls.mem<dram>>
    call @pack_node(%input_memref, %output_memref)
        : (memref<2x16xf32, #hls.mem<dram>>,
           memref<2x16xf32, #hls.mem<dram>>) -> ()
    return
  }

  // CHECK-LABEL: func.func @main(
  // CHECK-SAME: memref<2x2xvector<8xf32>, #hls.mem<dram>>
  // CHECK-SAME: memref<2x2xvector<8xf32>, #hls.mem<dram>>
  func.func @main(%input: memref<2x16xf32, #hls.mem<dram>>,
                  %output: memref<2x16xf32, #hls.mem<dram>>)
      attributes {runtime} {
    %input_axi = hls.axi.pack %input
        : (memref<2x16xf32, #hls.mem<dram>>)
        -> !hls.axi<memref<2x16xf32, #hls.mem<dram>>>
    %output_axi = hls.axi.pack %output
        : (memref<2x16xf32, #hls.mem<dram>>)
        -> !hls.axi<memref<2x16xf32, #hls.mem<dram>>>
    call @pack(%input_axi, %output_axi)
        : (!hls.axi<memref<2x16xf32, #hls.mem<dram>>>,
           !hls.axi<memref<2x16xf32, #hls.mem<dram>>>) -> ()
    return
  }

  // Compiler-owned scratch has no public scalar layout. Its complete aligned
  // reads and writes may therefore use packed words even though the arena is
  // flat and bidirectional.
  // CHECK-LABEL: func.func @scratch_node(
  // CHECK-SAME: memref<2xvector<8xf32>, #hls.mem<dram>>
  // CHECK: %[[WORD:.*]] = affine.load
  // CHECK-SAME: memref<2xvector<8xf32>
  // CHECK: vector.extract %[[WORD]][0]
  // CHECK: vector.from_elements
  // CHECK: affine.store
  // CHECK-SAME: memref<2xvector<8xf32>
  func.func @scratch_node(%arena: memref<16xf32, #hls.mem<dram>>) {
    affine.for %i = 0 to 16 {
      %value = affine.load %arena[%i] : memref<16xf32, #hls.mem<dram>>
      %one = arith.constant 1.0 : f32
      %next = arith.addf %value, %one : f32
      affine.store %next, %arena[%i] : memref<16xf32, #hls.mem<dram>>
    }
    return
  }

  // CHECK-LABEL: func.func @scratch_pack(
  // CHECK-SAME: !hls.axi<memref<2xvector<8xf32>, #hls.mem<dram>>>
  func.func @scratch_pack(
      %arena: !hls.axi<memref<16xf32, #hls.mem<dram>>>) {
    %bundle = hls.axi.bundle "scratch" : <f32, mm>
    %port = hls.axi.port %bundle, %arena {hls.scratch}
        : <f32, mm>, (!hls.axi<memref<16xf32, #hls.mem<dram>>>)
        -> memref<16xf32, #hls.mem<dram>>
    call @scratch_node(%port) : (memref<16xf32, #hls.mem<dram>>) -> ()
    return
  }

  // Lanes of one scratch word that arrive as separate stores are merged into
  // a single read-modify-write: one word load, the lane updates threaded
  // through it, one word store. Per lane instead, a pipelined body would
  // drive the same address eight times and serialize on the port.
  // CHECK-LABEL: func.func @lane_sweep_node(
  // CHECK: %[[WORD:.*]] = affine.load %{{.*}} : memref<2x2xvector<8xf32>
  // CHECK-NOT: affine.load %{{.*}} : memref<2x2xvector<8xf32>
  // CHECK: %[[I0:.*]] = vector.insert %{{.*}}, %[[WORD]]
  // CHECK: %[[I1:.*]] = vector.insert %{{.*}}, %[[I0]]
  // CHECK: vector.insert
  // CHECK: vector.insert
  // CHECK: vector.insert
  // CHECK: vector.insert
  // CHECK: vector.insert
  // CHECK: %[[LAST:.*]] = vector.insert
  // CHECK: affine.store %[[LAST]], %{{.*}} : memref<2x2xvector<8xf32>
  // CHECK-NOT: affine.store %{{.*}} : memref<2x2xvector<8xf32>
  func.func @lane_sweep_node(%in: memref<2x16xf32, #hls.mem<dram>>,
                             %arena: memref<2x16xf32, #hls.mem<dram>>) {
    affine.for %row = 0 to 2 {
      affine.for %word = 0 to 2 {
      %v0 = affine.load %in[%row, %word * 8 + 0]
          : memref<2x16xf32, #hls.mem<dram>>
      affine.store %v0, %arena[%row, %word * 8 + 0]
          : memref<2x16xf32, #hls.mem<dram>>
      %v1 = affine.load %in[%row, %word * 8 + 1]
          : memref<2x16xf32, #hls.mem<dram>>
      affine.store %v1, %arena[%row, %word * 8 + 1]
          : memref<2x16xf32, #hls.mem<dram>>
      %v2 = affine.load %in[%row, %word * 8 + 2]
          : memref<2x16xf32, #hls.mem<dram>>
      affine.store %v2, %arena[%row, %word * 8 + 2]
          : memref<2x16xf32, #hls.mem<dram>>
      %v3 = affine.load %in[%row, %word * 8 + 3]
          : memref<2x16xf32, #hls.mem<dram>>
      affine.store %v3, %arena[%row, %word * 8 + 3]
          : memref<2x16xf32, #hls.mem<dram>>
      %v4 = affine.load %in[%row, %word * 8 + 4]
          : memref<2x16xf32, #hls.mem<dram>>
      affine.store %v4, %arena[%row, %word * 8 + 4]
          : memref<2x16xf32, #hls.mem<dram>>
      %v5 = affine.load %in[%row, %word * 8 + 5]
          : memref<2x16xf32, #hls.mem<dram>>
      affine.store %v5, %arena[%row, %word * 8 + 5]
          : memref<2x16xf32, #hls.mem<dram>>
      %v6 = affine.load %in[%row, %word * 8 + 6]
          : memref<2x16xf32, #hls.mem<dram>>
      affine.store %v6, %arena[%row, %word * 8 + 6]
          : memref<2x16xf32, #hls.mem<dram>>
      %v7 = affine.load %in[%row, %word * 8 + 7]
          : memref<2x16xf32, #hls.mem<dram>>
      affine.store %v7, %arena[%row, %word * 8 + 7]
          : memref<2x16xf32, #hls.mem<dram>>
      } {loop_directive = #hls.loop<pipeline = true, target_ii = 1>}
    }
    return
  }

  // CHECK-LABEL: func.func @lane_sweep(
  func.func @lane_sweep(%in: !hls.axi<memref<2x16xf32, #hls.mem<dram>>>,
                        %arena: !hls.axi<memref<2x16xf32, #hls.mem<dram>>>) {
    %in_bundle = hls.axi.bundle "lane_in" : <f32, mm>
    %in_memref = hls.axi.port %in_bundle, %in
        : <f32, mm>, (!hls.axi<memref<2x16xf32, #hls.mem<dram>>>)
        -> memref<2x16xf32, #hls.mem<dram>>
    %arena_bundle = hls.axi.bundle "lane_arena" : <f32, mm>
    %arena_memref = hls.axi.port %arena_bundle, %arena {hls.scratch}
        : <f32, mm>, (!hls.axi<memref<2x16xf32, #hls.mem<dram>>>)
        -> memref<2x16xf32, #hls.mem<dram>>
    call @lane_sweep_node(%in_memref, %arena_memref)
        : (memref<2x16xf32, #hls.mem<dram>>,
           memref<2x16xf32, #hls.mem<dram>>) -> ()
    return
  }

  // A dynamic scratch read keeps the body compact: the packed output is
  // assembled through a carried vector instead of cloning the body 8 times.
  // CHECK-LABEL: func.func @compact_node(
  // CHECK: hls.dataflow.buffer {{.*}}hls.partition_kinds = ["complete"]
  // CHECK: affine.for %{{.*}} = 0 to 2
  // CHECK: affine.for %{{.*}} = 0 to 8
  // CHECK: memref.load %{{.*}}[%{{.*}}] : memref<2xvector<8xf32>
  // CHECK: vector.extract
  // CHECK: } {hls.unroll_factor = 8
  // CHECK: vector.from_elements

  // The transfer stays a full packed word while the compute lanes inside it
  // are capped: the burst width is a memory property, the lane count an
  // area one, and they are decided separately.
  // LANES2-LABEL: func.func @compact_node(
  // LANES2: memref.load %{{.*}}[%{{.*}}] : memref<2xvector<8xf32>
  // LANES2: } {hls.unroll_factor = 2
  func.func @compact_node(%input: memref<16xf32, #hls.mem<dram>>,
                          %output: memref<16xf32, #hls.mem<dram>>) {
    affine.for %i = 0 to 16 {
      %value = memref.load %input[%i] : memref<16xf32, #hls.mem<dram>>
      affine.store %value, %output[%i] : memref<16xf32, #hls.mem<dram>>
    } {loop_directive = #hls.loop<pipeline = true, target_ii = 1>}
    return
  }

  func.func @compact(
      %input: !hls.axi<memref<16xf32, #hls.mem<dram>>>,
      %output: !hls.axi<memref<16xf32, #hls.mem<dram>>>) {
    %input_bundle = hls.axi.bundle "compact_in" : <f32, mm>
    %input_memref = hls.axi.port %input_bundle, %input {hls.scratch}
        : <f32, mm>, (!hls.axi<memref<16xf32, #hls.mem<dram>>>)
        -> memref<16xf32, #hls.mem<dram>>
    %output_bundle = hls.axi.bundle "compact_out" : <f32, mm>
    %output_memref = hls.axi.port %output_bundle, %output {hls.scratch}
        : <f32, mm>, (!hls.axi<memref<16xf32, #hls.mem<dram>>>)
        -> memref<16xf32, #hls.mem<dram>>
    call @compact_node(%input_memref, %output_memref)
        : (memref<16xf32, #hls.mem<dram>>,
           memref<16xf32, #hls.mem<dram>>) -> ()
    return
  }

  // Aligned f64 movers use the full 512-bit word.
  // CHECK-LABEL: func.func @double_mover(
  // CHECK-SAME: memref<2xvector<8xf64>, #hls.mem<dram>>
  func.func @double_mover(%arena: memref<16xf64, #hls.mem<dram>>) {
    affine.for %i = 0 to 16 {
      %value = affine.load %arena[%i] : memref<16xf64, #hls.mem<dram>>
      affine.store %value, %arena[%i] : memref<16xf64, #hls.mem<dram>>
    }
    return
  }

  func.func @double_pack(
      %arena: !hls.axi<memref<16xf64, #hls.mem<dram>>>) {
    %bundle = hls.axi.bundle "double" : <f64, mm>
    %memref = hls.axi.port %bundle, %arena {hls.scratch}
        : <f64, mm>, (!hls.axi<memref<16xf64, #hls.mem<dram>>>)
        -> memref<16xf64, #hls.mem<dram>>
    call @double_mover(%memref) : (memref<16xf64, #hls.mem<dram>>) -> ()
    return
  }

  // A dynamic f64 write would require a wide read-modify-write adapter and is
  // kept scalar by the candidate cost model.
  // CHECK-LABEL: func.func @double_dynamic_store(
  // CHECK-SAME: memref<16xf64, #hls.mem<dram>>
  func.func @double_dynamic_store(%arena: memref<16xf64, #hls.mem<dram>>,
                                  %index: index, %value: f64) {
    memref.store %value, %arena[%index] : memref<16xf64, #hls.mem<dram>>
    return
  }

  func.func @double_dynamic_pack(
      %arena: !hls.axi<memref<16xf64, #hls.mem<dram>>>, %index: index,
      %value: f64) {
    %bundle = hls.axi.bundle "double_dynamic" : <f64, mm>
    %memref = hls.axi.port %bundle, %arena {hls.scratch}
        : <f64, mm>, (!hls.axi<memref<16xf64, #hls.mem<dram>>>)
        -> memref<16xf64, #hls.mem<dram>>
    call @double_dynamic_store(%memref, %index, %value)
        : (memref<16xf64, #hls.mem<dram>>, index, f64) -> ()
    return
  }
}
