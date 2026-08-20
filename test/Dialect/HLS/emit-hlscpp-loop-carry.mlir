// RUN: sar-translate --hls-emit-hlscpp %s | FileCheck %s

// Loop-carried values (`iter_args`) compile to ordinary variables: declared
// and initialized ahead of the loop, reassigned by the yield, read as the
// loop's results afterwards. A multi-carry yield stages through temporaries
// so a permuted carry reads this iteration's values.

// CHECK: void carries(
func.func @carries(%buf: memref<8xf64, #hls.mem<dram>>,
                   %out: memref<2xf64, #hls.mem<dram>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64

  // A single carry assigns directly: no temporary needed.
  // CHECK:      double [[ACC:v[0-9]+]] = (double)0;
  // CHECK-NEXT: for (
  // CHECK:        [[ACC]] = [[NEXT:v[0-9]+]];
  // CHECK-NEXT: }
  %sum = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> (f64) {
    %v = memref.load %buf[%i] : memref<8xf64, #hls.mem<dram>>
    %n = arith.addf %acc, %v : f64
    scf.yield %n : f64
  }

  // Two carries with a swap (Fibonacci): the yield stages both values
  // before either variable is overwritten.
  // CHECK:      double [[A:v[0-9]+]] = (double)0;
  // CHECK-NEXT: double [[B:v[0-9]+]] = (double)1;
  // CHECK-NEXT: for (
  // CHECK:        double [[T0:carry[0-9]+]] = [[B]];
  // CHECK-NEXT:   double [[T1:carry[0-9]+]] =
  // CHECK-NEXT:   [[A]] = [[T0]];
  // CHECK-NEXT:   [[B]] = [[T1]];
  // CHECK-NEXT: }
  %fib:2 = affine.for %j = 0 to 8 iter_args(%a = %zero, %b = %one)
      -> (f64, f64) {
    %s = arith.addf %a, %b : f64
    affine.yield %b, %s : f64, f64
  }

  // CHECK: out0[(int64_t)0] = [[ACC]];
  // CHECK: out0[(int64_t)1] = [[A]];
  memref.store %sum, %out[%c0] : memref<2xf64, #hls.mem<dram>>
  memref.store %fib#0, %out[%c1] : memref<2xf64, #hls.mem<dram>>
  return
}

// A carried array lives in its init buffer: the region argument reads it,
// the yield copies the fresh buffer back into it, and the loop result is
// that same buffer.
// CHECK: void carry_array(
func.func @carry_array(%f: memref<4xf64, #hls.mem<bram_t2p>>,
                       %out: memref<4xf64, #hls.mem<dram>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // CHECK: double [[INIT:buf[0-9]+]][4];
  %init = memref.alloc() : memref<4xf64, #hls.mem<bram_t2p>>
  // CHECK: for (
  %r = scf.for %i = %c0 to %c8 step %c1
      iter_args(%acc = %init) -> (memref<4xf64, #hls.mem<bram_t2p>>) {
    // CHECK: double [[NEXT:buf[0-9]+]][4];
    %next = memref.alloc() : memref<4xf64, #hls.mem<bram_t2p>>
    affine.for %a = 0 to 4 {
      // The region argument reads the init buffer directly.
      // CHECK: = [[INIT]][
      // CHECK: [[NEXT]][{{.*}}] =
      %x = affine.load %acc[%a] : memref<4xf64, #hls.mem<bram_t2p>>
      %y = affine.load %f[%a] : memref<4xf64, #hls.mem<bram_t2p>>
      %m = arith.mulf %x, %y : f64
      affine.store %m, %next[%a] : memref<4xf64, #hls.mem<bram_t2p>>
    }
    // The write-back sweep that stands in for the yield.
    // CHECK: [[INIT]][iv0] = [[NEXT]][iv0];
    scf.yield %next : memref<4xf64, #hls.mem<bram_t2p>>
  }
  affine.for %a = 0 to 4 {
    // CHECK: = [[INIT]][
    // CHECK: out0[{{.*}}] =
    %v = affine.load %r[%a] : memref<4xf64, #hls.mem<bram_t2p>>
    affine.store %v, %out[%a] : memref<4xf64, #hls.mem<dram>>
  }
  return
}
