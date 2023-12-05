// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_COMM_H_
#define CTRAN_COMM_H_

#include <memory>
#include "CtranGpe.h"
#include "CtranMapper.h"
#include "nccl.h"
#include "nccl_cvars.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_SENDRECV_ALGO
   type        : enum
   default     : orig
   choices     : orig, ctran
   description : |-
     The algorithm to use for sendrecv communication
     orig - Copy-based communication
     ctran - Ctran-based communication

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

#define CTRAN_COLL_INFO(                                                                                         \
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
  SENDRECV,
} CtranAlgoType;

typedef enum {
  UNKNOWN,

  SENDRECV_ORIG,
  SENDRECV_CTRAN,
} CtranAlgo;

struct ncclComm;

class Ctran {
 public:
  Ctran(ncclComm* comm);
  ~Ctran() = default;

  ncclResult_t commRegister(void* buff, size_t size, void** handle);
  ncclResult_t commDeregister(void* handle);

  std::unique_ptr<CtranMapper> mapper{nullptr};
  std::unique_ptr<CtranGpe> gpe{nullptr};
};

inline bool ctranIsUsed() {
  return (NCCL_SENDRECV_ALGO == NCCL_SENDRECV_ALGO::ctran);
}

ncclResult_t ctranInit(ncclComm* comm);
bool ctranInitialized(ncclComm* comm);
ncclResult_t ctranDestroy(ncclComm* comm);

bool ctranSendRecvSupport(int peer, ncclComm_t comm);
ncclResult_t ctranSend(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ctranRecv(
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream);

ncclResult_t ctranGroupEndHook(void);

#endif // CTRAN_COMM_H_
