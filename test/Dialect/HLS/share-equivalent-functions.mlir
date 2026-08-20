// RUN: sar-opt %s --hls-share-equivalent-functions | FileCheck %s
// RUN: sar-opt %s --hls-share-equivalent-functions | sar-translate --hls-emit-hlscpp | FileCheck %s --check-prefix=CPP

module {
  // CPP: #pragma HLS allocation function instances={{.*}}_engine limit=1
  // CHECK-LABEL: func.func private @leaf0
  func.func private @leaf0(%input: memref<16xf32>,
                           %output: memref<16xf32>) attributes {inline} {
    affine.for %i = 0 to 16 {
      %value = affine.load %input[%i] : memref<16xf32>
      affine.store %value, %output[%i] : memref<16xf32>
    }
    return
  }

  // CHECK-NOT: func.func private @leaf1
  func.func private @leaf1(%input: memref<16xf32>,
                           %output: memref<16xf32>) attributes {inline} {
    affine.for %i = 0 to 16 {
      %value = affine.load %input[%i] : memref<16xf32>
      affine.store %value, %output[%i] : memref<16xf32>
    }
    return
  }

  // CHECK-LABEL: func.func private @engine0
  // CHECK-SAME: attributes {hls.shared_instance}
  func.func private @engine0(%input: memref<16xf32>,
                             %output: memref<16xf32>) attributes {inline} {
    %scratch = memref.alloc() : memref<16xf32>
    func.call @leaf0(%input, %scratch)
        : (memref<16xf32>, memref<16xf32>) -> ()
    func.call @leaf0(%scratch, %output)
        : (memref<16xf32>, memref<16xf32>) -> ()
    return
  }

  // CHECK-NOT: func.func private @engine1
  func.func private @engine1(%input: memref<16xf32>,
                             %output: memref<16xf32>) attributes {inline} {
    %scratch = memref.alloc() : memref<16xf32>
    func.call @leaf1(%input, %scratch)
        : (memref<16xf32>, memref<16xf32>) -> ()
    func.call @leaf1(%scratch, %output)
        : (memref<16xf32>, memref<16xf32>) -> ()
    return
  }

  // CHECK-LABEL: func.func @top
  // CHECK-COUNT-2: call @engine0
  func.func @top(%input: memref<16xf32>,
                 %output: memref<16xf32>) attributes {top_func} {
    func.call @engine0(%input, %output)
        : (memref<16xf32>, memref<16xf32>) -> ()
    func.call @engine1(%input, %output)
        : (memref<16xf32>, memref<16xf32>) -> ()
    return
  }
}
