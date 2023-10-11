// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_H_
#define CTRAN_GPE_H_

#include "nccl.h"
#include <memory>

struct collOp {
  ncclResult_t (*func)(struct collOp *op);
  const void *ncclKernel;

  struct {
    const void *sendbuff;
    void *recvbuff;
    size_t sendcount;
    ncclDataType_t datatype;
    ncclComm_t comm;
  } allgather;
  struct {
    const void *sendbuff;
    size_t count;
    ncclDataType_t datatype;
    int peerRank;
    ncclComm_t comm;
  } send;
  struct {
    void *recvbuff;
    size_t count;
    ncclDataType_t datatype;
    int peerRank;
    ncclComm_t comm;
  } recv;
};

class ctranGpe {
  public:
    ctranGpe(int cudaDev);
    ~ctranGpe();
    ncclResult_t submit(std::unique_ptr<struct collOp> op, cudaStream_t stream);

  private:
    class impl;
    std::unique_ptr<impl> pimpl;
};

__global__ void ncclKernelAllGatherCTD(int *flag);
__global__ void ncclKernelAllGatherCTR(int *flag);
__global__ void ncclKernelAllGatherCTRD(int *flag);
__global__ void ncclKernelSend(int *flag);
__global__ void ncclKernelRecv(int *flag);

#endif
