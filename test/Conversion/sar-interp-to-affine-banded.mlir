// RUN: sar-opt %s --convert-sar-interp-to-affine | FileCheck %s
// RUN: sar-opt %s --convert-sar-interp-to-affine='enable-banded-gather=0' \
// RUN:   | FileCheck %s --check-prefix=NOBAND
// RUN: sar-opt %s --convert-sar-interp-to-affine='banded-profit-threshold=64' \
// RUN:   | FileCheck %s --check-prefix=NOBAND
// RUN: sar-opt %s --convert-sar-interp-to-affine='full-row-max-bytes=1024' \
// RUN:   | FileCheck %s --check-prefix=FULLROW

// Displacement-range analysis proves |positions[i,j] - j| is bounded when the
// position field is an identity ramp plus a bounded perturbation. The band
// buffer is then a small on-chip line buffer instead of the full source row.

// positions[i, j] = j + 0.5  ->  displacement interval [0.5, 0.5].
// taps = 8 gives a raw width of 1 + 8 = 9, rounded to 16.

// CHECK-LABEL: func.func @interp_banded
// Output planes, then the two narrow band buffers.
// CHECK: memref.alloc() : memref<8x32xf64>
// CHECK: memref.alloc() : memref<8x32xf64>
// Small bands use complete banking so dynamic tap slots can issue in parallel.
// CHECK: memref.alloc(){{.*}}hls.gather_strategy = "band"{{.*}}hls.partition_kinds = ["complete"]{{.*}} : memref<16xf64>
// CHECK: memref.alloc(){{.*}}hls.gather_strategy = "band"{{.*}}hls.partition_kinds = ["complete"]{{.*}} : memref<16xf64>
// CHECK-NOT: hls.min_ii
// Row loop, then the prologue that primes the window, then the column loop.
// CHECK: affine.for %{{.*}} = 0 to 8
// CHECK: affine.for %{{.*}} = 0 to 15
// CHECK: affine.for %{{.*}} = 0 to 32
// The gather reads the band buffer, never the source plane.
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<16xf64>
// CHECK-NOT: sar.interp1d_split
// CHECK-NOT: scf.if

// NOBAND-LABEL: func.func @interp_banded
// NOBAND-NOT: memref<16xf64>
// NOBAND-NOT: sar.interp1d_split
func.func @interp_banded(%re: tensor<8x32xf64>, %im: tensor<8x32xf64>)
    -> (tensor<8x32xf64>, tensor<8x32xf64>) {
  %pos_1d = "sar.constant"() <{value = dense<[
       0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,
       8.5,  9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5,
      16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5,
      24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5]>
      : tensor<32xf64>}> : () -> tensor<32xf64>
  %pos = "sar.broadcast"(%pos_1d) <{dim = 1 : i64}>
      : (tensor<32xf64>) -> tensor<8x32xf64>
  %r, %i = sar.interp1d_split %re, %im, %pos
      : (tensor<8x32xf64>, tensor<8x32xf64>, tensor<8x32xf64>)
      -> (tensor<8x32xf64>, tensor<8x32xf64>)
  return %r, %i : tensor<8x32xf64>, tensor<8x32xf64>
}

// -----

// sqrt(x*x + q) preserves a positive coordinate ramp: its residual over x is
// bounded even when q selects at runtime between bounded constant planes.
// The unknown mask prevents whole-plane constant folding.

// CHECK-LABEL: func.func @sqrt_residual_banded
// CHECK: memref.alloc(){{.*}} : memref<16xf64>
// CHECK: memref.alloc(){{.*}} : memref<16xf64>
// CHECK-NOT: sar.interp1d_split
func.func @sqrt_residual_banded(
    %re: tensor<8x32xf64>, %im: tensor<8x32xf64>,
    %mask: tensor<8x32xf64>) -> (tensor<8x32xf64>, tensor<8x32xf64>) {
  %axis = sar.constant dense<[
      100.0, 101.0, 102.0, 103.0, 104.0000000001, 105.0, 106.0, 107.0,
      108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0,
      116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0,
      124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0
    ]> : tensor<32xf64>
  %x = sar.broadcast %axis {dim = 1 : i64}
      : tensor<32xf64> -> tensor<8x32xf64>
  %x2 = sar.mul %x, %x : tensor<8x32xf64>
  %q0 = sar.constant dense<4.0> : tensor<8x32xf64>
  %q1 = sar.constant dense<16.0> : tensor<8x32xf64>
  %q = sar.where %mask, %q0, %q1
      : (tensor<8x32xf64>, tensor<8x32xf64>, tensor<8x32xf64>)
      -> tensor<8x32xf64>
  %sum = sar.add %x2, %q : tensor<8x32xf64>
  %root = sar.sqrt %sum : tensor<8x32xf64>
  %positions = sar.add_scalar %root, -100.0 : tensor<8x32xf64>
  %or, %oi = sar.interp1d_split %re, %im, %positions
      : (tensor<8x32xf64>, tensor<8x32xf64>, tensor<8x32xf64>)
      -> (tensor<8x32xf64>, tensor<8x32xf64>)
  return %or, %oi : tensor<8x32xf64>, tensor<8x32xf64>
}

// -----

// A ramp reached through add_scalar / mul_scalar and a row-varying (dim = 0)
// perturbation still resolves: the row term contributes only to the interval,
// leaving the ramp coefficient at 1.

// CHECK-LABEL: func.func @interp_banded_perturbed
// Displacement spans [-2, 3]; with the tap span that rounds to a 16-wide band.
// CHECK: memref.alloc(){{.*}} : memref<16xf64>
// CHECK-NOT: sar.interp1d_split
func.func @interp_banded_perturbed(%re: tensor<4x64xf64>, %im: tensor<4x64xf64>)
    -> (tensor<4x64xf64>, tensor<4x64xf64>) {
  %ramp_1d = "sar.constant"() <{value = dense<[
       0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
       8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
      16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
      24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
      32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0,
      40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
      48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0,
      56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0]>
      : tensor<64xf64>}> : () -> tensor<64xf64>
  %ramp = "sar.broadcast"(%ramp_1d) <{dim = 1 : i64}>
      : (tensor<64xf64>) -> tensor<4x64xf64>
  // Row-varying shift in [-2, 3]: broadcast along dim = 0 carries no ramp.
  %shift_1d = "sar.constant"() <{value = dense<[-2.0, 0.0, 1.0, 3.0]>
      : tensor<4xf64>}> : () -> tensor<4xf64>
  %shift = "sar.broadcast"(%shift_1d) <{dim = 0 : i64}>
      : (tensor<4xf64>) -> tensor<4x64xf64>
  %pos = "sar.add"(%ramp, %shift)
      : (tensor<4x64xf64>, tensor<4x64xf64>) -> tensor<4x64xf64>
  %r, %i = sar.interp1d_split %re, %im, %pos
      : (tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64xf64>)
      -> (tensor<4x64xf64>, tensor<4x64xf64>)
  return %r, %i : tensor<4x64xf64>, tensor<4x64xf64>
}

// -----

// The band touches both row edges: the ramp starts at -4, so the window runs
// off the low edge at j = 0 and off the high edge at j = cols-1. Staging
// clamps the address; the tap mask still zeroes those contributions, so this
// must stay numerically identical to the full-plane path.

// CHECK-LABEL: func.func @interp_banded_edges
// The ramp is an exact shift, so the band is just the tap span.
// CHECK: memref.alloc(){{.*}} : memref<8xf64>
// Clamp on the staging address.
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK-NOT: sar.interp1d_split
func.func @interp_banded_edges(%re: tensor<2x32xf64>, %im: tensor<2x32xf64>)
    -> (tensor<2x32xf64>, tensor<2x32xf64>) {
  %pos_1d = "sar.constant"() <{value = dense<[
      -4.0, -3.0, -2.0, -1.0,  0.0,  1.0,  2.0,  3.0,
       4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0,
      12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
      20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]>
      : tensor<32xf64>}> : () -> tensor<32xf64>
  %pos = "sar.broadcast"(%pos_1d) <{dim = 1 : i64}>
      : (tensor<32xf64>) -> tensor<2x32xf64>
  %r, %i = sar.interp1d_split %re, %im, %pos
      : (tensor<2x32xf64>, tensor<2x32xf64>, tensor<2x32xf64>)
      -> (tensor<2x32xf64>, tensor<2x32xf64>)
  return %r, %i : tensor<2x32xf64>, tensor<2x32xf64>
}

// -----

// Unprovable: the positions plane is a kernel argument, so the analysis has
// no compile-time range and the full-plane gather is kept.

// CHECK-LABEL: func.func @interp_unbounded_arg
// CHECK: memref.alloc(){{.*}}hls.gather_strategy = "direct"
// CHECK: memref.alloc(){{.*}}hls.gather_strategy = "direct"
// CHECK-NOT: memref<16xf64>
// CHECK: memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<8x32xf64>
// CHECK-NOT: sar.interp1d_split
// FULLROW-LABEL: func.func @interp_unbounded_arg
// Two f64 planes of the 32-element row occupy 512 bytes and fit the cap.
// A full row larger than the complete-banking cutoff uses one bank per tap.
// FULLROW: memref.alloc(){{.*}}hls.gather_strategy = "full_row"{{.*}}hls.partition_factors = [8]{{.*}}hls.partition_kinds = ["cyclic"]{{.*}} : memref<32xf64>
// FULLROW: memref.alloc(){{.*}}hls.gather_strategy = "full_row"{{.*}}hls.partition_factors = [8]{{.*}}hls.partition_kinds = ["cyclic"]{{.*}} : memref<32xf64>
// FULLROW-NOT: sar.interp1d_split
func.func @interp_unbounded_arg(%re: tensor<8x32xf64>, %im: tensor<8x32xf64>,
                                %pos: tensor<8x32xf64>)
    -> (tensor<8x32xf64>, tensor<8x32xf64>) {
  %r, %i = sar.interp1d_split %re, %im, %pos
      : (tensor<8x32xf64>, tensor<8x32xf64>, tensor<8x32xf64>)
      -> (tensor<8x32xf64>, tensor<8x32xf64>)
  return %r, %i : tensor<8x32xf64>, tensor<8x32xf64>
}

// -----

// A coefficient far from one has a finite bound for this static shape, but the
// resulting band is wider than the row and fails the profitability gate.

// CHECK-LABEL: func.func @interp_scaled_ramp
// CHECK-NOT: memref<16xf64>
// CHECK-NOT: sar.interp1d_split
func.func @interp_scaled_ramp(%re: tensor<2x16xf64>, %im: tensor<2x16xf64>)
    -> (tensor<2x16xf64>, tensor<2x16xf64>) {
  %pos_1d = "sar.constant"() <{value = dense<[
      0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
      16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]>
      : tensor<16xf64>}> : () -> tensor<16xf64>
  %pos = "sar.broadcast"(%pos_1d) <{dim = 1 : i64}>
      : (tensor<16xf64>) -> tensor<2x16xf64>
  %r, %i = sar.interp1d_split %re, %im, %pos
      : (tensor<2x16xf64>, tensor<2x16xf64>, tensor<2x16xf64>)
      -> (tensor<2x16xf64>, tensor<2x16xf64>)
  return %r, %i : tensor<2x16xf64>, tensor<2x16xf64>
}
