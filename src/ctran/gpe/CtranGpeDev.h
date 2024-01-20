
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_DEV_H_
#define CTRAN_GPE_DEV_H_

#include "CtranAlgoDev.h"

struct CtranKernelAllGatherArgs {
  const void* sendbuff;
  void* recvbuff;
  size_t nbytes;
};

// TODO: Placeholder for sendrecv for now since we don't pass NVL sendrecv to
// ctran. Define actual arguments when supporting NVL sendrecv with p2pElems
// pool
struct CtranKernelSendArgs {
  int dummy;
};

struct CtranKernelRecvArgs {
  int dummy;
};

struct CtranKernelSendRecvArgs {
  int dummy;
};

struct CtranKernelAllToAllArgs {
  const void* sendbuff;
  void* recvbuff;
  size_t count;
};

struct CtranKernelArgs {
  CtranAlgoDeviceState* devState_d;
  union {
    CtranKernelAllGatherArgs allgather;
    CtranKernelSendArgs send;
    CtranKernelRecvArgs recv;
    CtranKernelSendRecvArgs sendrecv;
    CtranKernelAllToAllArgs alltoall;
  } collective;
};
#endif
