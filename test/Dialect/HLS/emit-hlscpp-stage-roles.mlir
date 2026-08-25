// RUN: sar-translate --hls-emit-hlscpp -emit-vitis-directives %s | FileCheck %s

// A stage's name carries the role the IR can still prove. The classification
// looks through calls and follows the twiddle tables, so a stage that drives
// its arithmetic from helpers is not mistaken for a plain copy.

// A stage handed a twiddle table is a transform, even though every butterfly
// sits in the callee and the stage body itself only moves data.
// CHECK: void roles_s00_fft(

// A stage that computes a sine and a cosine applies a phase factor.
// CHECK: void roles_s02_phase(

// A stage that only moves data keeps the copy role.
// CHECK: void roles_s03_copy(

module {
  // The butterfly: all of the transform's arithmetic lives here.
  func.func @roles_butterfly(%tw: memref<4xf64, #hls.mem<bram_t2p>>,
                             %io: memref<4x4xf64, #hls.mem<bram_t2p>>)
      attributes {inline} {
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 {
        %t = affine.load %tw[%j] : memref<4xf64, #hls.mem<bram_t2p>>
        %v = affine.load %io[%i, %j] : memref<4x4xf64, #hls.mem<bram_t2p>>
        %m = arith.mulf %v, %t : f64
        affine.store %m, %io[%i, %j] : memref<4x4xf64, #hls.mem<bram_t2p>>
      }
    }
    return
  }

  // The transform stage: a loop over butterfly calls, no arithmetic of its own.
  func.func @roles_transform(%tw: memref<4xf64, #hls.mem<bram_t2p>>,
                             %io: memref<4x4xf64, #hls.mem<bram_t2p>>) {
    call @roles_butterfly(%tw, %io)
        : (memref<4xf64, #hls.mem<bram_t2p>>,
           memref<4x4xf64, #hls.mem<bram_t2p>>) -> ()
    return
  }

  func.func @roles_phase(%io: memref<4x4xf64, #hls.mem<bram_t2p>>) {
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 {
        %v = affine.load %io[%i, %j] : memref<4x4xf64, #hls.mem<bram_t2p>>
        %s = math.sin %v : f64
        %c = math.cos %v : f64
        %a = arith.addf %s, %c : f64
        affine.store %a, %io[%i, %j] : memref<4x4xf64, #hls.mem<bram_t2p>>
      }
    }
    return
  }

  func.func @roles_move(%src: memref<4x4xf64, #hls.mem<bram_t2p>>,
                        %dst: memref<4x4xf64, #hls.mem<bram_t2p>>) {
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 {
        %v = affine.load %src[%i, %j] : memref<4x4xf64, #hls.mem<bram_t2p>>
        affine.store %v, %dst[%i, %j] : memref<4x4xf64, #hls.mem<bram_t2p>>
      }
    }
    return
  }

  func.func @roles(%arg0: memref<4x4xf64, #hls.mem<bram_t2p>>,
                   %arg1: memref<4x4xf64, #hls.mem<bram_t2p>>)
      attributes {top_func} {
    %tw = hls.dataflow.const_buffer
        {source_name = "__sar_fft_twiddle_cos_4_s0_f64",
         value = dense<[1.0, 0.0, -1.0, 0.0]> : tensor<4xf64>}
        : memref<4xf64, #hls.mem<bram_t2p>>
    call @roles_transform(%tw, %arg0)
        : (memref<4xf64, #hls.mem<bram_t2p>>,
           memref<4x4xf64, #hls.mem<bram_t2p>>) -> ()
    call @roles_phase(%arg0) : (memref<4x4xf64, #hls.mem<bram_t2p>>) -> ()
    call @roles_move(%arg0, %arg1)
        : (memref<4x4xf64, #hls.mem<bram_t2p>>,
           memref<4x4xf64, #hls.mem<bram_t2p>>) -> ()
    return
  }
}
