// RUN: sar-opt %s --split-input-file --verify-diagnostics

func.func @buffer_zero_depth() {
  // expected-error @+1 {{buffer depth must be positive}}
  %0 = hls.dataflow.buffer {depth = 0 : i32} : memref<4xf32>
  return
}

// -----

func.func @stream_zero_depth() {
  // expected-error @+1 {{stream depth must be positive}}
  %0 = hls.dataflow.stream {depth = 0 : i32} : <f32, 0>
  return
}

// -----

#zero_vector = #hls.tile<[8], [0]>
// expected-error @-1 {{tile and vector dimensions must be positive}}

func.func @invalid_tile_layout() attributes {layout = #zero_vector} {
  return
}

// -----

func.func @pack_mismatched_type(%arg0: memref<4xf32>) {
  // expected-error @+1 {{source type doesn't align with AXI element type}}
  %0 = hls.axi.pack %arg0
      : (memref<4xf32>) -> !hls.axi<memref<4xf64>>
  return
}
