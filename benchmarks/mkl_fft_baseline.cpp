//===- mkl_fft_baseline.cpp - optional MKL comparison ---------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
#include <mkl.h>
#include <mkl_dfti.h>

#include <cstdint>

extern "C" void *sar_mkl_plan_c128(int64_t rows, int64_t cols, int threads) {
  if (rows < 1 || cols < 2 || threads < 1)
    return nullptr;
  mkl_set_num_threads_local(threads);
  DFTI_DESCRIPTOR_HANDLE descriptor = nullptr;
  MKL_LONG status = DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX,
                                         1, (MKL_LONG)cols);
  if (!status)
    status =
        DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)rows);
  MKL_LONG strides[2] = {0, 1};
  if (!status)
    status = DftiSetValue(descriptor, DFTI_INPUT_STRIDES, strides);
  if (!status)
    status = DftiSetValue(descriptor, DFTI_OUTPUT_STRIDES, strides);
  if (!status)
    status = DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, (MKL_LONG)cols);
  if (!status)
    status = DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, (MKL_LONG)cols);
  if (!status)
    status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  if (!status)
    status = DftiCommitDescriptor(descriptor);
  if (status && descriptor)
    DftiFreeDescriptor(&descriptor);
  return status ? nullptr : descriptor;
}

extern "C" int sar_mkl_execute_c128(void *plan, const void *input,
                                    void *output) {
  if (!plan || !input || !output)
    return -1;
  return (int)DftiComputeForward((DFTI_DESCRIPTOR_HANDLE)plan,
                                 const_cast<void *>(input), output);
}

extern "C" void sar_mkl_plan_free(void *plan) {
  auto descriptor = (DFTI_DESCRIPTOR_HANDLE)plan;
  if (descriptor)
    DftiFreeDescriptor(&descriptor);
}
