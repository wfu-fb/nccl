// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_COMM_DDA_H_
#define NCCL_COMM_DDA_H_

#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "checks.h"
#include "ddaThreadSharedMd.h"
#include "ddaMemHandles.h"

int64_t ncclParamMaxDDAThreads(void);
int64_t ncclParamDDAAllreduceTmpbuffSize(void);
int64_t ncclParamDDAAllreduceTreeThresholdNVS(void);
int64_t ncclParamDDAAllreduceTreeThresholdHCM(void);

typedef enum {
  NCCL_DDA_TOPO_TYPE__NVS,
  NCCL_DDA_TOPO_TYPE__HCM,
  NCCL_DDA_TOPO_TYPE__UNKNOWN,
} ncclDDATopoType_t;

typedef enum {
  NCCL_DDA_ALLREDUCE_ALGO_DEFAULT,
  NCCL_DDA_ALLREDUCE_ALGO_DDA_IPC,
  NCCL_DDA_ALLREDUCE_ALGO_DDA_THREADED,
} ncclDDAAllReduceAlgo_t;

class ddaPrivateMd {
public:
  ddaPrivateMd(ddaThreadSharedMd *threadSharedMd, int rank, int cudaDev, int nRanks, int numCliques, ncclComm *comm) {
    this->barrierFlag = 0;
    this->barrierMboxId = 1;
    CUDACHECKIGNORE(cudaGetDeviceProperties(&this->devProp, cudaDev));

    CUDACHECKIGNORE(cudaMalloc(&this->tmpbuff, ncclParamDDAAllreduceTmpbuffSize()));

    /* add topology information */
    this->topoType = NCCL_DDA_TOPO_TYPE__UNKNOWN;
    if (numCliques == 1) {
      this->topoType = NCCL_DDA_TOPO_TYPE__NVS;
    } else if (numCliques == 2) {
      this->topoType = NCCL_DDA_TOPO_TYPE__HCM;
    }

    this->threadSharedMd = threadSharedMd;
    this->memHandles = std::unique_ptr<class ddaMemHandles>(new ddaMemHandles(threadSharedMd, comm));
  }

  ~ddaPrivateMd() {
    CUDACHECKIGNORE(cudaFree(this->tmpbuff));
  }

  // flag indicating that each rank has arrived at the barrier
  uintptr_t barrierFlag;

  // barrier mailbox ID to use
  int barrierMboxId;

  // device properties
  cudaDeviceProp devProp;

  // local tmpbuff
  void* tmpbuff{nullptr};

  // topology type
  ncclDDATopoType_t topoType;

  // thread-shared meta-data
  ddaThreadSharedMd *threadSharedMd;

  // ipc states
  // barrier mailboxes
  uintptr_t* barrierMbox[2];

  // all ranks' tmpbuff addresses
  void** allTmpSendbuffs{nullptr};

  std::unique_ptr<class ddaMemHandles> memHandles;
};

ncclDDAAllReduceAlgo_t getAllReduceAlgo(const void* sendbuff, void* recvbuff,
                                        size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                                        ncclComm* comm);
ncclResult_t allocDDAMd(ncclComm *comm, ncclUniqueId commId);
ncclResult_t freeDDAMd(ncclComm *comm);

static inline int typeSize(ncclDataType_t datatype)
{
  switch (datatype) {
    case ncclInt8:
      return sizeof(int8_t);

    case ncclUint8:
      return sizeof(uint8_t);

    case ncclInt32:
      return sizeof(int32_t);

    case ncclUint32:
      return sizeof(uint32_t);

    case ncclInt64:
      return sizeof(int64_t);

    case ncclUint64:
      return sizeof(uint64_t);

    case ncclFloat16:
      return sizeof(half);

  case ncclFloat32:
      return sizeof(float);

    case ncclFloat64:
      return sizeof(double);

#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
      return sizeof(__nv_bfloat16);
#endif

    default:
      return 0;
  }
}

#endif
