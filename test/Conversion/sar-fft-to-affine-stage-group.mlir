// RUN: sar-opt %s --convert-sar-fft-to-affine \
// RUN:   | FileCheck %s --check-prefix=UNROLL
// RUN: sar-opt %s --convert-sar-fft-to-affine=fft-stage-group=2 \
// RUN:   | FileCheck %s --check-prefix=GROUP2
// RUN: sar-opt %s --convert-sar-fft-to-affine=fft-stage-group=99 \
// RUN:   | FileCheck %s --check-prefix=SATURATE
// RUN: sar-opt %s --convert-sar-fft-to-affine=fft-parallel-rows=4 \
// RUN:   | FileCheck %s --check-prefix=PAR4

// A 64-point transform has three radix-4 stages, so two are intermediate
// and the full unroll allocates two (re, im) scratch pairs. Every variant
// also allocates the prefetch and write-back block pair, so four pairs =
// eight line allocations in total.

// The default is the full unroll: one scratch line per intermediate stage.
// UNROLL-LABEL: func.func @grouped
// UNROLL-COUNT-8: memref.alloc() : memref<64xf64>
// UNROLL-NOT: memref.alloc()

// Grouping two stages reaches the two-slot safety floor.
// GROUP2-LABEL: func.func @grouped
// GROUP2-COUNT-8: memref.alloc() : memref<64xf64>
// GROUP2-NOT: memref.alloc()

// A group larger than the stage count saturates at the two-line floor
// rather than collapsing to one: a Stockham butterfly reads and writes
// different lines, so a single shared buffer would make the transform
// in place and wrong.
// SATURATE-LABEL: func.func @grouped
// SATURATE-COUNT-8: memref.alloc() : memref<64xf64>
// SATURATE-NOT: memref.alloc()

// Four lanes in flight: each line buffer grows a leading lane dimension
// hinted complete (each unrolled lane owns its banks), and the block loop
// advances one four-line block per iteration.
// PAR4-LABEL: func.func @grouped
// PAR4-COUNT-4: memref.alloc() {hls.partition_factors = [4, 1], hls.partition_kinds = ["complete", "none"]} : memref<4x64xf64>
// PAR4: affine.for {{.*}} = 0 to 8 step 4

func.func @grouped(%re: tensor<8x64xf64>, %im: tensor<8x64xf64>)
    -> (tensor<8x64xf64>, tensor<8x64xf64>) {
  %r, %i = sar.fft_split %re, %im {dim = 1 : i64} : tensor<8x64xf64>
  return %r, %i : tensor<8x64xf64>, tensor<8x64xf64>
}
