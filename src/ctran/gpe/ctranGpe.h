// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_H_
#define CTRAN_GPE_H_

#include "nccl.h"
#include <memory>
#include <vector>

typedef ncclResult_t (*collOpFunc)(std::vector<std::unique_ptr<struct collOp>> opGroup);

struct collOp {
  enum opType {
    ALLGATHER,
    SEND,
    RECV,
  } type;
  cudaStream_t stream;
  ncclComm_t comm;

  struct {
    const void *sendbuff;
    void *recvbuff;
    size_t sendcount;
    ncclDataType_t datatype;
  } allgather;
  struct {
    const void *sendbuff;
    size_t count;
    ncclDataType_t datatype;
    int peerRank;
  } send;
  struct {
    void *recvbuff;
    size_t count;
    ncclDataType_t datatype;
    int peerRank;
  } recv;
};

class ctranGpe {
  public:
    ctranGpe(int cudaDev);
    ~ctranGpe();
    ncclResult_t submit(std::vector<std::unique_ptr<struct collOp>> opGroup, collOpFunc func,
        const void *ncclKernel);

  private:
    class impl;
    std::unique_ptr<impl> pimpl;
};

__global__ void ncclKernelAllGatherCtranDirect(int *flag);
__global__ void ncclKernelAllGatherCtranRing(int *flag);
__global__ void ncclKernelAllGatherCtranRecDbl(int *flag);
__global__ void ncclKernelSend(int *flag);
__global__ void ncclKernelRecv(int *flag);

#endif
