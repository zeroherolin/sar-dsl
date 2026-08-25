// RUN: sar-opt %s --sar-demote-loop-carries --verify-diagnostics

// A yield naming another carry's argument would need its value staged
// before the first write-back clobbers it; the demotion refuses rather
// than miscompiles.

func.func @swapped(%a0: memref<4xf64>, %b0: memref<4xf64>)
    -> (memref<4xf64>, memref<4xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // expected-error @+1 {{swapped buffer carries are not supported}}
  %r:2 = scf.for %i = %c0 to %c8 step %c1
      iter_args(%a = %a0, %b = %b0) -> (memref<4xf64>, memref<4xf64>) {
    scf.yield %b, %a : memref<4xf64>, memref<4xf64>
  }
  return %r#0, %r#1 : memref<4xf64>, memref<4xf64>
}

// A chain rather than a swap: the second carry names the first. Every carry
// is checked before any is rewritten, because the rewrite replaces a
// carry's argument with its buffer -- checking as it goes would let this
// one through and order the copies wrong, which is a wrong image rather
// than an error.

func.func @chained(%a0: memref<4xf64>, %b0: memref<4xf64>)
    -> (memref<4xf64>, memref<4xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // expected-error @+1 {{chained or swapped buffer carries are not supported}}
  %r:2 = scf.for %i = %c0 to %c8 step %c1
      iter_args(%a = %a0, %b = %b0) -> (memref<4xf64>, memref<4xf64>) {
    %next = memref.alloc() : memref<4xf64>
    affine.for %j = 0 to 4 {
      %v = affine.load %a[%j] : memref<4xf64>
      affine.store %v, %next[%j] : memref<4xf64>
    }
    scf.yield %next, %a : memref<4xf64>, memref<4xf64>
  }
  return %r#0, %r#1 : memref<4xf64>, memref<4xf64>
}
