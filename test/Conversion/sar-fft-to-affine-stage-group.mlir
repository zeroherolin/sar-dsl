// RUN: sar-opt %s --convert-sar-fft-to-affine \
// RUN:   | FileCheck %s --check-prefix=UNROLL
// RUN: sar-opt %s --convert-sar-fft-to-affine=fft-stage-group=2 \
// RUN:   | FileCheck %s --check-prefix=GROUP2
// RUN: sar-opt %s --convert-sar-fft-to-affine=fft-stage-group=99 \
// RUN:   | FileCheck %s --check-prefix=SATURATE

// A 64-point transform has log2(64) = 6 stages, so 5 of them are
// intermediate and the full unroll allocates 5 (re, im) scratch pairs.

// The default is the full unroll: one scratch line per intermediate stage,
// so 5 pairs = 10 line allocations, and no eleventh.
// UNROLL-LABEL: func.func @grouped
// UNROLL-COUNT-10: memref.alloc() : memref<64xf64>
// UNROLL-NOT: memref.alloc() : memref<64xf64>

// Grouping two stages per slot leaves ceil(6/2) - 1 = 2 pairs.
// GROUP2-LABEL: func.func @grouped
// GROUP2-COUNT-4: memref.alloc() : memref<64xf64>
// GROUP2-NOT: memref.alloc() : memref<64xf64>

// A group larger than the stage count saturates at the two-line floor
// rather than collapsing to one: a Stockham butterfly reads and writes
// different lines, so a single shared buffer would make the transform
// in place and wrong.
// SATURATE-LABEL: func.func @grouped
// SATURATE-COUNT-4: memref.alloc() : memref<64xf64>
// SATURATE-NOT: memref.alloc() : memref<64xf64>

func.func @grouped(%re: tensor<4x64xf64>, %im: tensor<4x64xf64>)
    -> (tensor<4x64xf64>, tensor<4x64xf64>) {
  %r, %i = sar.fft_split %re, %im {dim = 1 : i64} : tensor<4x64xf64>
  return %r, %i : tensor<4x64xf64>, tensor<4x64xf64>
}
