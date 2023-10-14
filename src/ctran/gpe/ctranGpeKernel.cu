// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranGpeKernel.h"

static inline __device__ void ncclKernelStallStream(int *flag) {
  volatile int* flag_d = flag;
  *flag_d = KERNEL_STARTED;
  while (*flag_d != KERNEL_TERMINATE) {}
}

__global__ void ncclKernelAllGatherCtranDirect(int *flag) {
  ncclKernelStallStream(flag);
}

__global__ void ncclKernelAllGatherCtranRing(int *flag) {
  ncclKernelStallStream(flag);
}

__global__ void ncclKernelAllGatherCtranRecDbl(int *flag) {
  ncclKernelStallStream(flag);
}

__global__ void ncclKernelSend(int *flag) {
  ncclKernelStallStream(flag);
}

__global__ void ncclKernelRecv(int *flag) {
  ncclKernelStallStream(flag);
}
