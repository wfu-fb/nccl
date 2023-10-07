// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_ALGOS_H_
#define CTRAN_ALGOS_H_

#include "nccl.h"

typedef enum {
  ALLGATHER,
} ctranAlgoType;

typedef enum {
  UNKNOWN,

  ALLGATHER_ORIG,
  ALLGATHER_CTRAN_DIRECT,
  ALLGATHER_CTRAN_RING,
} ctranAlgo;

ctranAlgo ctranAlgoGet(ctranAlgoType type);

ncclResult_t ctranAllGatherDirect(const void* sendbuff, void* recvbuff,
    size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ctranAllGatherRing(const void* sendbuff, void* recvbuff,
    size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

#endif
