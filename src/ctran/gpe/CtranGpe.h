// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_H_
#define CTRAN_GPE_H_

#include <memory>
#include <vector>
#include "CtranAlgoDev.h"
#include "CtranGpeDev.h"
#include "nccl.h"

typedef ncclResult_t (*opFunc)(
    std::vector<std::unique_ptr<struct OpElem>> opGroup);

struct OpElem {
  enum opType {
    ALLGATHER,
    SEND,
    RECV,
  } type;
  cudaStream_t stream;
  ncclComm_t comm;

  union {
    struct {
      const void* sendbuff;
      void* recvbuff;
      size_t sendcount;
      ncclDataType_t datatype;
    } allgather;
    struct {
      const void* sendbuff;
      size_t count;
      ncclDataType_t datatype;
      int peerRank;
    } send;
    struct {
      void* recvbuff;
      size_t count;
      ncclDataType_t datatype;
      int peerRank;
    } recv;
  };

 public:
  OpElem(enum opType type, ncclComm_t comm);

  // Constructor used to store grouped operations (e.g., grouped send/recv)
  OpElem(enum opType type, cudaStream_t stream, ncclComm_t comm);
  ~OpElem();
};

struct KernelConfig {
  enum KernelType {
    ALLGATHER,
    SEND,
    RECV,
    SENDRECV,
  } type;
  unsigned int numBlocks{1};
  unsigned int numThreads{1};

  cudaStream_t stream;
  CtranKernelArgs args;

 public:
  KernelConfig(enum KernelType type, cudaStream_t stream)
      : type(type), stream(stream){};
  ~KernelConfig(){};
};

class CtranGpe {
 public:
  CtranGpe(int cudaDev);
  ~CtranGpe();

  ncclResult_t submit(
      std::vector<std::unique_ptr<struct OpElem>> opGroup,
      opFunc func,
      KernelConfig& kernelConfig,
      const void* ncclKernel);

  ncclResult_t submit(
      std::vector<std::unique_ptr<struct OpElem>> opGroup,
      opFunc func);

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

static inline void ctranKernelSetAllGatherArgs(
    const void* sendbuff,
    void* recvbuff,
    size_t nbytes,
    CtranAlgoDeviceState* devState_d,
    CtranKernelArgs* args) {
  args->devState_d = devState_d;
  args->collective.allgather.sendbuff = sendbuff;
  args->collective.allgather.recvbuff = recvbuff;
  args->collective.allgather.nbytes = nbytes;
}

__global__ void ncclKernelAllGatherCtranDirect(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllGatherArgs args);

__global__ void ncclKernelAllGatherCtranRing(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllGatherArgs args);

__global__ void ncclKernelAllGatherCtranRecDbl(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllGatherArgs args);

__global__ void ncclKernelSend(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelSendArgs args);

__global__ void ncclKernelRecv(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelRecvArgs args);

__global__ void ncclKernelSendRecv(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelSendRecvArgs args);

#endif
