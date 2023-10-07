// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranGpeKernel.h"

static inline __device__ void ncclKernelStallStream(int *flag) {
  volatile int* flag_d = flag;
  *flag_d = KERNEL_STARTED;
  while (*flag_d != KERNEL_TERMINATE) {}
}

__global__ void ncclKernelAllGatherCTD(int *flag) {
  ncclKernelStallStream(flag);
}

__global__ void ncclKernelAllGatherCTR(int *flag) {
  ncclKernelStallStream(flag);
}
