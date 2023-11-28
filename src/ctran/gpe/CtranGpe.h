// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_H_
#define CTRAN_GPE_H_

#include <memory>
#include <vector>
#include "nccl.h"

typedef ncclResult_t (*opFunc)(
    std::vector<std::unique_ptr<struct OpElem>> opGroup);

struct OpElem {
  enum opType {
    SEND,
    RECV,
  } type;
  cudaStream_t stream;
  ncclComm_t comm;

  union {
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
};

class CtranGpe {
 public:
  CtranGpe(int cudaDev);
  ~CtranGpe();
  ncclResult_t submit(
      std::vector<std::unique_ptr<struct OpElem>> opGroup,
      opFunc func,
      const void* ncclKernel);

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

#endif
