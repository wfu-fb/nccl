// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_ALGOS_H_
#define CTRAN_ALGOS_H_

#include "nccl.h"

typedef enum {
  ALLGATHER,
  SENDRECV,
} ctranAlgoType;

typedef enum {
  UNKNOWN,

  ALLGATHER_ORIG,
  ALLGATHER_CTRAN_DIRECT,
  ALLGATHER_CTRAN_RING,
  ALLGATHER_CTRAN_RD,

  SENDRECV_ORIG,
  SENDRECV_CTRAN,
} ctranAlgo;

ctranAlgo ctranAlgoGet(ctranAlgoType type);

ncclResult_t ctranAllGatherDirect(const void* sendbuff, void* recvbuff,
    size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ctranAllGatherRing(const void* sendbuff, void* recvbuff,
    size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ctranAllGatherRd(const void* sendbuff, void* recvbuff,
    size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ctranSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ctranRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

#endif
