// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_ALGOS_H_
#define CTRAN_ALGOS_H_

#include "nccl.h"

#define LOG_COLL_INFO(algoStr, sendbuff, recvbuff, sendcount, datatype, comm, stream) do {  \
    INFO(NCCL_COLL,                                                                         \
        "%s: opCount %lx sendbuff %p recvbuff %p sendcount %zi datatype %d comm %p commHash %lu [nranks=%d, localRanks=%d] stream=%p\n",  \
        algoStr, comm->opCount, sendbuff, recvbuff, sendcount, datatype, comm, comm->commHash, comm->nRanks, \
        comm->localRanks, stream);                                       \
    comm->opCount++;                                                                    \
} while (0)

#define LOG_SENDRECV_INFO(                                                                                         \
    algoStr, buffer, count, datatype, peer, comm, stream)                                                          \
  do {                                                                                                             \
    INFO(                                                                                                          \
        NCCL_COLL,                                                                                                 \
        "%s: opCount %lx buffer %p count %zi datatype %d peer %d comm %lu [nranks=%d, localRanks=%d] stream=%p\n", \
        algoStr,                                                                                                   \
        comm->opCount,                                                                                             \
        buffer,                                                                                                    \
        count,                                                                                                     \
        datatype,                                                                                                  \
        peer,                                                                                                      \
        comm->commHash,                                                                                            \
        comm->nRanks,                                                                                              \
        comm->localRanks,                                                                                          \
        stream);                                                                                                   \
    comm->opCount++;                                                                                               \
  } while (0)

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

ncclResult_t ctranGroupEndHook(void);

#endif
