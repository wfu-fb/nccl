// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CtranGpeKernel.h"

__global__ void ncclKernelAllGatherCtranDirect(int *flag) {
  ncclKernelStallStream(flag);
}

__global__ void ncclKernelSend(int *flag) {
  ncclKernelStallStream(flag);
}

__global__ void ncclKernelRecv(int *flag) {
  ncclKernelStallStream(flag);
}

__global__ void ncclKernelSendRecv(int *flag) {
  ncclKernelStallStream(flag);
}
